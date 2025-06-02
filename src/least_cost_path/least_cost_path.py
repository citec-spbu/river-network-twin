import math
import os
import time
from pathlib import Path
import tempfile
import networkit as nk
from osgeo import gdal
from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsFeature,
    QgsProject,
    QgsGeometry,
    QgsRasterLayer,
    QgsPointXY,
    QgsVectorLayer,
    QgsProcessingUtils,
    QgsField,
)
from PyQt5.QtCore import QVariant
from qgis.PyQt.QtWidgets import QMessageBox
from qgis.utils import iface

from src.least_cost_path.layers.output_least_cost_path import (
    build_output_least_cost_path,
)
from src.least_cost_path.layers.watershed_boundaries import build_watershed_boundaries
from src.progress_manager import ProgressManager
from src.river.layers.water_rasterized import build_water_rasterized
from .layers.polygon_from_path import flood_fill_areas

def least_cost_path_analysis(
        points_layer: QgsVectorLayer,
        dem_layer: QgsRasterLayer,
        water_raster_layer: QgsRasterLayer,
        feedback = None
    )-> tuple[QgsVectorLayer, QgsVectorLayer]:
    # Инициализация прогресса
    progress = ProgressManager(
        title="Анализ оптимальных путей", label="Инициализация..."
    )
    progress.init_progress(100)

    # Переменные для хранения сообщений
    height_message = None
    rivers_message = None

    try:
        # Получение необходимых слоев
        def log_message(message):
            if feedback:
                feedback.pushInfo(message)
            else:
                print(message)

        def create_temp_raster():
            """Создает временный растр"""
            return QgsProcessingUtils.generateTempFilename('temp_raster.tif')
        if not progress.update(5, "Поиск слоев..."):
            return


        src_crs = points_layer.crs()
        tgt_crs = QgsCoordinateReferenceSystem("EPSG:3857")
        transform_context = QgsProject.instance().transformContext()
        coord_transform = QgsCoordinateTransform(src_crs, tgt_crs, transform_context)

        #Подготовка DEM (перепроецирование и уменьшение разрешения)
        print("Создание слоя с путями...", flush=True)
        log_message("Подготовка DEM...")
        if not progress.update(10, "Подготовка DEM..."):
            return

        # Создаем временный файл для результата
        dem_reprojected = create_temp_raster()

        try:
            # Проверяем тип dem_layer
            if isinstance(dem_layer, str):
                # Если dem_layer - строка (путь к файлу)
                dem_file_path = dem_layer
            elif hasattr(dem_layer, 'source'):  # Если это QgsRasterLayer
                dem_file_path = dem_layer.source()
            else:
                raise TypeError("dem_layer должен быть строкой (путь) или QgsRasterLayer")

            print(f"Исходный файл DEM: {dem_file_path}")  # Для диагностики

            # Проверка существования файла
            if not os.path.isfile(dem_file_path):
                raise FileNotFoundError(f"Файл DEM не найден: {dem_file_path}")

            # Выполняем перепроецирование напрямую из файла
            print("Выполняю перепроецирование DEM...", flush=True)
            result = gdal.Warp(
                dem_reprojected,
                dem_file_path,  # Передаем путь как строку
                dstSRS="EPSG:3857",
                resampleAlg="bilinear",
                format="GTiff"
            )
            progress._keep_active()
            if not result:
                raise RuntimeError("Ошибка при выполнении gdal.Warp")
            if not progress.update(15, "Ресемплинг DEM..."):
                return
            print("Перепроецирование DEM завершено", flush=True)

        except Exception as e:
            # Удаляем временный файл при ошибке
            if os.path.exists(dem_reprojected):
                os.remove(dem_reprojected)
            raise Exception(f"Ошибка подготовки DEM: {str(e)}")


        ds3857 = gdal.Open(str(dem_reprojected))
        gt3857 = ds3857.GetGeoTransform()
        orig_xres = gt3857[1]
        orig_yres = abs(gt3857[5])
        ds3857 = None  # Закрываем файл

        dem_pooled = create_temp_raster()
        gdal.Warp(
            destNameOrDestDS=str(dem_pooled),
            srcDSOrSrcDSTab=str(dem_reprojected),
            xRes=orig_xres * 4,
            yRes=orig_yres * 4,
            resampleAlg="average",
            format="GTiff",
        )
        progress._keep_active()

        if not progress.update(20, "Загрузка слоя стоимости..."):
            return

        # 4. Подготовка растра воды (перепроецирование и ресемплинг)
        log_message("Построение графа стоимости...")

        t_paths_start = time.perf_counter()

        # строим граф из cost_layer
        g, gt, n_rows, n_cols = build_cost_graph(
            Path(dem_pooled.source()), Path(water_raster_layer.source())
        )

        terminal_nodes_set = set()
        ds_water = gdal.Open(Path(water_raster_layer.source()))
        arr_water = ds_water.GetRasterBand(1).ReadAsArray().astype(float)
        nodata_water = ds_water.GetRasterBand(1).GetNoDataValue()
        if nodata_water is not None:
            arr_water[arr_water == nodata_water] = 0

        # Создаем слой перемещенных источников (в памяти)
        moved_sources_layer = QgsVectorLayer(
            f"Point?crs=EPSG:3857", 
            "Moved sources", 
            "memory"
        )
        moved_sources_provider = moved_sources_layer.dataProvider()
        moved_sources_provider.addAttributes([QgsField("original_id", QVariant.Int)])
        moved_sources_layer.updateFields()

        terminal_nodes_set = set()
        for feat in points_layer.getFeatures():
            z = feat["z"]
            if z is None:
                continue

            pt = feat.geometry().asPoint()
            pt3857 = coord_transform.transform(pt)

            i, j = nearest_land(
                pt3857.x(), pt3857.y(), gt, n_rows, n_cols, arr_water, 2
            )
            if i == -1 or j == -1:
                continue
            node_idx = i * n_cols + j

            # Добавляем точку в слой перемещенных источников
            new_feat = QgsFeature(moved_sources_layer.fields())
            new_feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(*pixel_to_coord(i, j, gt))))
            new_feat.setAttribute("original_id", feat.id())
            moved_sources_provider.addFeature(new_feat)

            terminal_nodes_set.add(node_idx)

        terminal_nodes = list(terminal_nodes_set)
        QgsProject.instance().addMapLayer(moved_sources_layer)
        
        arr_water = None

        # Создаем временный файл с расширением .gpkg
        temp_file_lcp = tempfile.NamedTemporaryFile(suffix='.gpkg', delete=False)
        temp_file_lcp.close()  # Закрываем файл, чтобы его можно было использовать

        # Создаем временный файл в директории QGIS
        temp_dir = QgsProcessingUtils.tempFolder()
        temp_file = os.path.join(temp_dir, f"least_cost_path_{os.urandom(4).hex()}.gpkg")

        # Создаем слой через вашу функцию
        lcp_layer = build_output_least_cost_path(temp_file)
        
        # Проверяем валидность слоя
        if not lcp_layer.isValid():
            raise Exception("Не удалось создать слой оптимальных путей")

        # Устанавливаем понятное имя для отображения в QGIS
        lcp_layer.setName("Least Cost Paths")
        

        if not progress.update(50, "Расчет оптимальных путей..."):
                return

        dp = lcp_layer.dataProvider()
        total_pairs = len(terminal_nodes) * (len(terminal_nodes) - 1)
        processed_pairs = 0

        for i in range(len(terminal_nodes)):
            src_node = terminal_nodes[i]
            if progress.was_canceled():
                return

            progress.update(
                50 + int(20 * i / len(terminal_nodes)),
                f"Расчет путей из точки {i + 1}/{len(terminal_nodes)}",
            )

            src_node = terminal_nodes[i]
            dijk = nk.distance.Dijkstra(g, src_node)
            dijk.run()

            for dst in terminal_nodes[i + 1 :]:
                processed_pairs += 1
                if processed_pairs % 10 == 0:
                    progress.update(
                        50 + int(20 * processed_pairs / total_pairs),
                        f"Обработано {processed_pairs}/{total_pairs} пар",
                    )
                    if progress.was_canceled():
                        return

                node_path = dijk.getPath(dst)
                if not node_path:
                    continue

                # Конвертируем узлы в координаты
                path_pts = []
                for u in node_path:
                    if u != node_path[0] and u != node_path[-1]:
                        if u in terminal_nodes_set:
                            break
                    path_pts.append(
                        QgsPointXY(*pixel_to_coord(u // n_cols, u % n_cols, gt))
                    )
                else:
                    feat_out = QgsFeature(lcp_layer.fields())
                    feat_out.setGeometry(QgsGeometry.fromPolylineXY(path_pts))
                    dp.addFeature(feat_out)

        lcp_layer.updateExtents()
        QgsProject.instance().addMapLayer(lcp_layer)
        print("Создан слой с путями", flush=True)

        t_paths_end = time.perf_counter()
        print(
            f"Для построения слоя с путями потребовалось: {t_paths_end - t_paths_start:.3f} s",
            flush=True,
        )

        if not progress.update(75, "Фильтрация путей по высоте..."):
            return
        
        elevation_layer = QgsRasterLayer(
            str(dem_pooled), "SRTM DEM Layer Pooled (3857)"
        )

        # Фильтрация путей по критерию разницы высот
        paths_to_delete = []
        features = list(lcp_layer.getFeatures())
        total_features = len(features)

        for idx, feature in enumerate(features):
            if progress.was_canceled():
                break

            progress.update(
                75 + int(10 * idx / total_features),
                f"Фильтрация по высоте {idx + 1}/{total_features}",
            )

            geom = feature.geometry()
            min_elev = calculate_minimum_elevation(elevation_layer, geom)
            if min_elev is None:
                continue

            polyline = (
                geom.asMultiPolyline()[0] if geom.isMultipart() else geom.asPolyline()
            )
            if not polyline:
                continue

            first_point = polyline[0]
            last_point = polyline[-1]
            sample_start = elevation_layer.dataProvider().sample(
                QgsPointXY(first_point.x(), first_point.y()), 1
            )
            sample_end = elevation_layer.dataProvider().sample(
                QgsPointXY(last_point.x(), last_point.y()), 1
            )
            if sample_start and sample_end:
                z_start, valid_start = sample_start
                z_end, valid_end = sample_end
                if not (valid_start and valid_end):
                    continue

                z1 = min(z_start, z_end)
                if min_elev < z1 - 15:
                    paths_to_delete.append(feature.id())

        if paths_to_delete:
            lcp_layer.startEditing()
            for fid in paths_to_delete:
                lcp_layer.deleteFeature(fid)
            lcp_layer.commitChanges()
            
            # Используем feedback вместо QMessageBox
            if feedback:
                feedback.pushInfo(f"Удалено {len(paths_to_delete)} путей по критерию высоты.")
            else:
                print(f"Удалено {len(paths_to_delete)} путей по критерию высоты.", flush=True)

            height_message = f"Удалено {len(paths_to_delete)} путей по критерию высоты."

        if not progress.update(90, "Фильтрация путей по рекам..."):
            return

        progress.finish()

        reply = QMessageBox.question(
            iface.mainWindow(),
            "Построить слой водоразделов?",
            "Хотите построить слой замкнутых путей наибольших по площади?\n"
            "ВНИМАНИЕ: обработка может занять очень много времени!",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        # Создаем временный файл вместо сохранения в project_folder
        with tempfile.NamedTemporaryFile(suffix='.gpkg', delete=False) as temp_file:
            temp_file_path = temp_file.name

        watershed_layer = build_watershed_boundaries(
            lcp_layer, temp_file_path  # Используем временный файл вместо пути в project_folder
        )

        # Добавляем слой в проект (если нужно)
        QgsProject.instance().addMapLayer(watershed_layer)

            # Очистка временных файлов
        for temp_file in [dem_reprojected, temp_file_lcp]:
            try:
                os.remove(temp_file)
            except:
                pass

        return lcp_layer, moved_sources_layer
    finally:
        progress.finish()

        # Показываем сообщения после закрытия прогресса
        if height_message:
            QMessageBox.information(None, "Информация", height_message)
        if rivers_message:
            QMessageBox.information(None, "Информация", rivers_message)


def build_cost_graph(raster_path, water_layer, eps=1e-6):
    ds_cost = None
    ds_water = None
    try:
        # Открываем основной растр
        ds_cost = gdal.Open(str(raster_path))
        if ds_cost is None:
            raise RuntimeError(f"Не удалось открыть растр стоимости: {raster_path}")
        
        # Открываем водный растр
        ds_water = gdal.Open(water_layer)
        if ds_water is None:
            raise RuntimeError(f"Не удалось открыть водный растр: {water_layer}")

        arr = ds_cost.GetRasterBand(1).ReadAsArray().astype(float)
        arr_water = ds_water.GetRasterBand(1).ReadAsArray().astype(float)
        
        # Проверка совпадения размеров
        if arr_water.shape != arr.shape:
            raise RuntimeError("Размеры водного растра не совпадают с растром стоимости")

        nodata_water = ds_water.GetRasterBand(1).GetNoDataValue()
        if nodata_water is not None:
            arr_water[arr_water == nodata_water] = 0

        rows, cols = arr.shape
        g = nk.Graph(rows * cols, weighted=True, directed=False)

        # 4 направления от вниз-влево до вправо
        neigh = [
            (1, 0, 1.0),  # вниз
            (0, 1, 1.0),  # вправо
            (1, -1, math.sqrt(2)),
            (1, 1, math.sqrt(2)),
        ]

        def nid(i, j):
            return i * cols + j

        for i in range(rows):
            for j in range(cols):
                u = nid(i, j)
                hu = arr[i, j]
                if arr_water[i, j] != 0:
                    continue
                for di, dj, factor in neigh:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        hv = arr[ni, nj]
                        dh = abs(hu - hv)
                        w = factor * (dh + eps)
                        if arr_water[ni, nj] == 0:
                            g.addEdge(u, nid(ni, nj), w)

        gt = ds_cost.GetGeoTransform()
        return g, gt, rows, cols

    finally:
        # Гарантированное освобождение ресурсов
        if ds_cost is not None:
            ds_cost = None
        if ds_water is not None:
            ds_water = None


def coord_to_pixel(x, y, gt):
    inv = 1.0 / (gt[1] * gt[5] - gt[2] * gt[4])
    j_float = inv * (gt[5] * (x - gt[0]) - gt[2] * (y - gt[3]))
    i_float = inv * (-gt[4] * (x - gt[0]) + gt[1] * (y - gt[3]))
    i = int(i_float)
    j = int(j_float)
    return i, j


def pixel_to_coord(i, j, gt):
    x = gt[0] + (j + 0.5) * gt[1] + (i + 0.5) * gt[2]
    y = gt[3] + (j + 0.5) * gt[4] + (i + 0.5) * gt[5]
    return x, y


def calculate_minimum_elevation(raster_layer, line_geom):
    provider = raster_layer.dataProvider()
    min_elev = float("inf")
    lines = (
        line_geom.asMultiPolyline()
        if line_geom.isMultipart()
        else [line_geom.asPolyline()]
    )
    for line in lines:
        for pt in line:
            sample = provider.sample(QgsPointXY(pt.x(), pt.y()), 1)
            if sample:
                value, valid = sample
                if valid and value is not None:
                    min_elev = min(min_elev, value)
    return min_elev if min_elev != float("inf") else None


def nearest_land(x, y, gt, n_rows, n_cols, water, radius):
    i, j = coord_to_pixel(x, y, gt)
    if not 0 <= i < n_rows or not 0 <= j < n_cols:
        print(f"({i}, {j})", flush=True)
        return -1, -1

    row_start = max(0, i - radius)
    row_end = min(i + radius + 1, n_rows)
    col_start = max(0, j - radius)
    col_end = min(j + radius + 1, n_cols)

    nearest_point = (i, j)
    min_squared_distance = -1

    for row in range(row_start, row_end):
        for col in range(col_start, col_end):
            if water[row, col] == 0:
                coord_x, coord_y = pixel_to_coord(row, col, gt)
                dist = (x - coord_x) ** 2 + (y - coord_y) ** 2
                if min_squared_distance > dist or min_squared_distance == -1:
                    nearest_point = (row, col)
                    min_squared_distance = dist

    return nearest_point
