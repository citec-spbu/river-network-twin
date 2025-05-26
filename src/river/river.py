import os
from typing import List, Optional
from src.common import (
    add_dem_layer,
    add_opentopo_layer,
    download_dem,
    enable_processing_algorithms,
    get_coordinates,
    reproject_dem,
    set_project_crs,
    transform_coordinates,
)
from ..progress_manager import ProgressManager
from osgeo import gdal
import processing
from time import sleep
from qgis.core import (
    QgsProject,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsPointXY,
    QgsGeometry,
    QgsField,
    QgsRaster,
    QgsFeature,
)
from qgis.PyQt.QtCore import QVariant, QEventLoop
from qgis.PyQt.QtWidgets import QMessageBox, QInputDialog
from qgis.utils import iface
from .point_selection_tool import PointSelectionTool
from .layers.rivers_by_object_filtered import build_rivers_by_object_filtered
from .layers.max_height_points import build_max_height_points
from .layers.basins import build_basins_layer
from .layers.rivers_and_points import build_rivers_and_points_layer
from .layers.rivers_merged import build_merged_layer
from .layers.utils import load_quickosm_layer
from .layers.clustering import assign_clusters, preparing_data_for_clustering
RIVER_FILTERS = {
    "max_strahler_order": (">=", 2),
    "total_length": (">", 1000),
}

# ========== НАСТРОЙКА ПАРАМЕТРОВ ДЛЯ КЛАСТЕРИЗАЦИИ ==========
RESAMPLE_SCALE = 5  # Параметр масштаба для ресемплинга (2-10)
CONTOUR_INTERVAL = 20  # Интервал изолиний в метрах


# ============================================================


def transform_bbox(x_min, x_max, y_min, y_max, from_epsg, to_epsg):
    # Создаем объекты систем координат
    from_crs = QgsCoordinateReferenceSystem(f"EPSG:{from_epsg}")
    to_crs = QgsCoordinateReferenceSystem(f"EPSG:{to_epsg}")
    tr = QgsCoordinateTransform(from_crs, to_crs, QgsProject.instance())

    # Преобразуем все 4 угла bbox
    points = [
        QgsPointXY(x_min, y_min),
        QgsPointXY(x_min, y_max),
        QgsPointXY(x_max, y_max),
        QgsPointXY(x_max, y_min),
    ]

    transformed_points = [tr.transform(point) for point in points]

    # Находим новые границы
    x_coords = [p.x() for p in transformed_points]
    y_coords = [p.y() for p in transformed_points]

    return f"{min(x_coords)}, {max(x_coords)}, {min(y_coords)}, {max(y_coords)}"


def river(project_folder, with_clustering):
    # Инициализация проекта
    set_project_crs()
    enable_processing_algorithms()
    add_opentopo_layer()

    # Выбор территории анализа
    bbox = select_analysis_bbox()
    if bbox is None:
        return

    # Инициализация прогресса
    progress = ProgressManager(title="Анализ рек", label="Инициализация...")
    progress.init_progress(100)

    try:
        if not progress.update(5, "Настройка проекта"):
            return

        # Загрузка и подготовка DEM
        if not progress.update(10, "Загрузка цифровой модели рельефа"):
            return

        # Скачивание DEM
        dem_path = download_dem(bbox, project_folder)

        # Создание перепроецированного DEM в EPSG:3857
        dem_3857 = f"{project_folder}/river_dem_3857.tif"
        gdal.Warp(
            dem_3857,
            dem_path,
            dstSRS="EPSG:3857",
            resampleAlg="bilinear",
            format="GTiff",
        )

        # Добавление слоя DEM
        dem_layer = add_dem_layer(dem_path)
        reprojected_relief = reproject_dem(dem_path)

        sleep(0.1)
        progress._keep_active()

        # Анализ водосборных бассейнов
        if not progress.update(20, "Анализ водосборных бассейнов"):
            return
        basins_path = os.path.join(project_folder, "basins.sdat")
        basins = build_basins_layer(reprojected_relief, basins_path)
        QgsProject.instance().addMapLayer(basins)

        # Анализ речной сети
        if not progress.update(25, "Анализ речной сети"):
            return
        extent = transform_bbox(bbox[0], bbox[2], bbox[1], bbox[3], 4326, 3857)
        merged_path = os.path.join(project_folder, "merge_result.gpkg")
        rivers_path = os.path.join(project_folder, "rivers.gpkg")
        streams_path = os.path.join(project_folder, "streams.gpkg")
        rivers_merged = build_merged_layer(extent, merged_path, rivers_path, streams_path)
        QgsProject.instance().addMapLayer(rivers_merged)

        # Загрузка данных о водных объектах
        if not progress.update(30, "Загрузка данных о водных объектах"):
            return
        water_output = os.path.join(project_folder, "water.gpkg")
        load_quickosm_layer(
            "water",
            "natural",
            "water",
            extent,
            output_path=water_output,
            quickosm_layername="multipolygons",
        )

        # Расчет координат точек
        if not progress.update(35, "Расчет координат точек"):
            return
        start_x = processing.run(
            "native:fieldcalculator",
            {
                "INPUT": merged_path,
                "FIELD_NAME": "start_x",
                "FIELD_TYPE": 0,
                "FIELD_LENGTH": 0,
                "FIELD_PRECISION": 0,
                "FORMULA": "x(start_point($geometry))",
                "OUTPUT": "TEMPORARY_OUTPUT",
            },
        )["OUTPUT"]

        start_y = processing.run(
            "native:fieldcalculator",
            {
                "INPUT": start_x,
                "FIELD_NAME": "start_y",
                "FIELD_TYPE": 0,
                "FIELD_LENGTH": 0,
                "FIELD_PRECISION": 0,
                "FORMULA": "y(start_point($geometry))",
                "OUTPUT": "TEMPORARY_OUTPUT",
            },
        )["OUTPUT"]

        end_x = processing.run(
            "native:fieldcalculator",
            {
                "INPUT": start_y,
                "FIELD_NAME": "end_x",
                "FIELD_TYPE": 0,
                "FIELD_LENGTH": 0,
                "FIELD_PRECISION": 0,
                "FORMULA": "x(end_point($geometry))",
                "OUTPUT": "TEMPORARY_OUTPUT",
            },
        )["OUTPUT"]

        end_y = processing.run(
            "native:fieldcalculator",
            {
                "INPUT": end_x,
                "FIELD_NAME": "end_y",
                "FIELD_TYPE": 0,
                "FIELD_LENGTH": 0,
                "FIELD_PRECISION": 0,
                "FORMULA": "y(end_point($geometry))",
                "OUTPUT": "TEMPORARY_OUTPUT",
            },
        )["OUTPUT"]

        # Добавление высотных данных
        if not progress.update(40, "Добавление высотных данных"):
            return
        layer_provider = end_y.dataProvider()
        layer_provider.addAttributes(
            [QgsField("start_z", QVariant.Double), QgsField("end_z", QVariant.Double)]
        )
        end_y.updateFields()
        idx_start_z = layer_provider.fields().indexOf("start_z")
        idx_end_z = layer_provider.fields().indexOf("end_z")

        end_y.startEditing()
        line_provider = end_y.dataProvider()
        changes = {}

        features = list(end_y.getFeatures())
        total_features = len(features)

        for i, feature in enumerate(features):
            if progress.was_canceled():
                end_y.rollBack()
                return

            progress.update(40 + int(20 * i / total_features),
                            f"Обработка высотных данных ({i + 1}/{total_features})")

            geom = feature.geometry()
            if geom.isMultipart():
                polyline = geom.asMultiPolyline()[0]
            else:
                polyline = geom.asPolyline()

            start_point = QgsPointXY(polyline[0])
            end_point = QgsPointXY(polyline[-1])

            start_z = dem_layer.dataProvider().identify(
                start_point, QgsRaster.IdentifyFormatValue
            )
            end_z = dem_layer.dataProvider().identify(
                end_point, QgsRaster.IdentifyFormatValue
            )

            start_z_value = None
            end_z_value = None

            if start_z.isValid():
                start_z_value = start_z.results()[1]
                feature["start_z"] = start_z_value

            if end_z.isValid():
                end_z_value = end_z.results()[1]
                feature["end_z"] = end_z_value

            changes[feature.id()] = {idx_start_z: start_z_value, idx_end_z: end_z_value}

        line_provider.changeAttributeValues(changes)
        end_y.commitChanges()

        # Фильтрация рек
        if not progress.update(65, "Фильтрация рек"):
            return
        rivers_by_object_filtered_path = os.path.join(
            project_folder, "rivers_by_object_filtered.gpkg"
        )
        rivers_by_object_filtered = build_rivers_by_object_filtered(
            end_y, RIVER_FILTERS, rivers_by_object_filtered_path
        )
        QgsProject.instance().addMapLayer(rivers_by_object_filtered)

        # Определение максимальных высот
        if not progress.update(70, "Определение максимальных высот"):
            return
        rivers_and_points_path = os.path.join(project_folder, "rivers_with_points.gpkg")
        rivers_and_points = build_rivers_and_points_layer(end_y, rivers_and_points_path)
        QgsProject.instance().addMapLayer(rivers_and_points)

        start_points = set()
        end_points = set()

        for feat in rivers_and_points.getFeatures():
            sx, sy = feat["start_x"], feat["start_y"]
            ex, ey = feat["end_x"], feat["end_y"]
            if sx is not None and sy is not None:
                start_points.add((sx, sy))
            if ex is not None and ey is not None:
                end_points.add((ex, ey))

        # Создание точек максимальной высоты
        if not progress.update(80, "Создание точек максимальной высоты"):
            return
        point_layer_path = os.path.join(project_folder, "max_height_points.gpkg")
        point_layer = build_max_height_points(point_layer_path)
        fields = point_layer.fields()

        point_layer.startEditing()
        features = list(rivers_and_points.getFeatures())
        total_features = len(features)

        for i, feat in enumerate(features):
            if progress.was_canceled():
                point_layer.rollBack()
                return

            progress.update(80 + int(15 * i / total_features),
                            f"Обработка точек ({i + 1}/{total_features})")

            max_z = feat["max_z"]
            if max_z is None:
                continue

            sx, sy, start_z = feat["start_x"], feat["start_y"], feat["start_z"]
            ex, ey, end_z = feat["end_x"], feat["end_y"], feat["end_z"]

            if (
                    sx is not None
                    and sy is not None
                    and start_z is not None
                    and start_z == max_z
            ):
                if (sx, sy) not in end_points:
                    pt = QgsPointXY(sx, sy)
                    new_feat = QgsFeature()
                    new_feat.setFields(fields)
                    new_feat.setGeometry(QgsGeometry.fromPointXY(pt))
                    new_feat["x"] = sx
                    new_feat["y"] = sy
                    new_feat["z"] = start_z
                    point_layer.addFeature(new_feat)

            if ex is not None and ey is not None and end_z is not None and end_z == max_z:
                if (ex, ey) not in start_points:
                    pt = QgsPointXY(ex, ey)
                    new_feat = QgsFeature()
                    new_feat.setFields(fields)
                    new_feat.setGeometry(QgsGeometry.fromPointXY(pt))
                    new_feat["x"] = ex
                    new_feat["y"] = ey
                    new_feat["z"] = end_z
                    point_layer.addFeature(new_feat)

        point_layer.commitChanges(True)
        QgsProject.instance().addMapLayer(point_layer)

        # Кластеризация (если требуется)
        if with_clustering:
            if not progress.update(95, "Кластеризация точек"):
                return
            copied_point_layer = point_layer.clone()
            data_for_clustering_path = os.path.join(
                project_folder, "Изолинии.gpkg"
            )
            data_for_clustering = preparing_data_for_clustering(
                copied_point_layer,
                dem_layer,
                RESAMPLE_SCALE,
                CONTOUR_INTERVAL,
                data_for_clustering_path
            )
            QgsProject.instance().addMapLayer(data_for_clustering)

            points_and_clusters_path = os.path.join(
                project_folder, "Points_and_clusters.gpkg"
            )
            points_and_clusters = assign_clusters(
                data_for_clustering,
                copied_point_layer,
                points_and_clusters_path
            )
            QgsProject.instance().addMapLayer(points_and_clusters)

        progress.update(100, "Завершено!")

    finally:
        progress.finish()


def select_analysis_bbox() -> Optional[List[float]]:
    method, ok = QInputDialog.getItem(
        None,
        "Выбор метода",
        "Как определить область анализа?",
        ["Радиус вокруг точки", "Область по 4 точкам"],
        0,
        False,
    )
    if not ok:
        return None

    if method == "Радиус вокруг точки":
        center_method, ok = QInputDialog.getItem(
            None,
            "Способ определения центра",
            "Как определить центр области?",
            ["Ручной ввод координат", "Выбрать точку на карте"],
            0,
            False,
        )
        if not ok:
            return None

        if center_method == "Ручной ввод координат":
            x, y = get_coordinates()
            if x is None or y is None:
                return None
            lon, lat = transform_coordinates(x, y)
        else:
            canvas = iface.mapCanvas()
            tool = PointSelectionTool(canvas, points=1)
            canvas.setMapTool(tool)

            QMessageBox.information(
                None,
                "Выбор центра",
                "Выберите точку на карте, затем дождитесь завершения.",
            )

            loop = QEventLoop()
            tool.selection_completed.connect(loop.quit)
            loop.exec_()

            if len(tool.points) != 1:
                QMessageBox.warning(None, "Ошибка", "Не выбрана точка.")
                return None

            transform = QgsCoordinateTransform(
                QgsCoordinateReferenceSystem("EPSG:3857"),
                QgsCoordinateReferenceSystem("EPSG:4326"),
                QgsProject.instance(),
            )
            point_4326 = transform.transform(tool.points[0])
            lon, lat = point_4326.x(), point_4326.y()

        radius, ok = QInputDialog.getDouble(
            None,
            "Радиус вокруг точки",
            "Введите радиус (градусы):",
            value=0.5,
            min=0.1,
            max=5,
            decimals=5,
        )
        if not ok:
            return None

        return [lon - radius, lat - radius, lon + radius, lat + radius]

    elif method == "Область по 4 точкам":
        canvas = iface.mapCanvas()
        tool = PointSelectionTool(canvas, points=4)
        canvas.setMapTool(tool)

        QMessageBox.information(
            None,
            "Выбор территории",
            "Выберите 4 точки на карте, затем дождитесь завершения.",
        )

        loop = QEventLoop()
        tool.selection_completed.connect(loop.quit)
        loop.exec_()

        if len(tool.points) != 4:
            QMessageBox.warning(None, "Ошибка", "Выбрано не 4 точки.")
            return None

        transform = QgsCoordinateTransform(
            QgsCoordinateReferenceSystem("EPSG:3857"),
            QgsCoordinateReferenceSystem("EPSG:4326"),
            QgsProject.instance(),
        )
        points_4326 = [transform.transform(p) for p in tool.points]

        x_coords = [p.x() for p in points_4326]
        y_coords = [p.y() for p in points_4326]
        return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

    return None
