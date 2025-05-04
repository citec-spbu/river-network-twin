import math
from qgis.core import (
    QgsField,
    QgsProject,
    QgsPointXY,
    QgsGeometry,
    QgsSpatialIndex,
    QgsRasterLayer,
    QgsVectorLayer,
    QgsFeature,
)
from qgis.PyQt.QtCore import QVariant
from qgis.PyQt.QtWidgets import QMessageBox
import numpy as np
import networkit as nk
from scipy.ndimage import uniform_filter
from osgeo import gdal
from datetime import datetime


def mean_pool2d(arr, kernel_size, stride=None):
    """
    Применяет mean pooling к 2D массиву.

    Параметры:
        arr: 2D NumPy array
        kernel_size: размер ядра (int или tuple) (h, w)
        stride: шаг (int или tuple). Если None, равен kernel_size

    Возвращает:
        2D массив после max pooling
    """
    if isinstance(kernel_size, int):
        kh, kw = kernel_size, kernel_size
    else:
        kh, kw = kernel_size

    if stride is None:
        sh, sw = kh, kw
    elif isinstance(stride, int):
        sh, sw = stride, stride
    else:
        sh, sw = stride

    # Вычисляем размеры выходного массива
    H, W = arr.shape
    oh = (H - kh) // sh + 1
    ow = (W - kw) // sw + 1

    # строим «скользящее окно»
    shape = (oh, ow, kh, kw)
    strides = (arr.strides[0] * sh, arr.strides[1] * sw, arr.strides[0], arr.strides[1])
    windows = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

    # усредняем сразу по последним двум осям
    return windows.mean(axis=(2, 3))


def generate_cost_layer(dem_path, project_folder, window_size=3, pool_size=3) -> str:
    # читаем DEM
    src_ds = gdal.Open(dem_path)

    # извлекаем геотрансформ и размеры
    gt = src_ds.GetGeoTransform()
    crs = src_ds.GetProjection()
    width = src_ds.RasterXSize
    height = src_ds.RasterYSize

    # вычисляем центр пикселя
    center_longitude = gt[0] + gt[1] * (width / 2.0)
    center_latitude = gt[3] + gt[5] * (height / 2.0)

    utm_zone = int((center_longitude + 180.0) / 6.0) + 1
    if center_latitude >= 0:
        target_crs = f"EPSG:{32600 + utm_zone}"
    else:
        target_crs = f"EPSG:{32700 + utm_zone}"

    # 1. перепроецируем DEM в UTM в памяти
    mem_utm = gdal.Warp(
        "", src_ds, format="MEM", dstSRS=target_crs, resampleAlg=gdal.GRA_Bilinear
    )

    # 2. на этом промежуточном объекте считаем уклон
    mem_slope_utm = gdal.DEMProcessing(
        "",
        mem_utm,
        "slope",
        format="MEM",
        scale=1.0,
        slopeFormat="degree",
        alg="ZevenbergenThorne",
    )

    # Warp уклона обратно в исходный CRS
    mem_slope = gdal.Warp(
        "", mem_slope_utm, format="MEM", dstSRS=crs, resampleAlg=gdal.GRA_Bilinear
    )
    # Загружаем растр через GDAL
    slope_arr = mem_slope.GetRasterBand(1).ReadAsArray().astype(np.float32)

    # Быстрый расчёт локальной дисперсии через фильтры
    mean1 = uniform_filter(slope_arr, size=window_size, mode="reflect")
    mean2 = uniform_filter(slope_arr * slope_arr, size=window_size, mode="reflect")
    variance_array = mean2 - mean1 * mean1

    # 3. Пороги
    variance_array[variance_array < 0] = 0
    if np.any(variance_array > 0):
        median_var = np.nanmedian(variance_array)
        mad = 1.4826 * np.nanmedian(np.abs(variance_array - median_var))
        threshold = median_var + 3 * mad
        print(f"Threshold: {threshold}", flush=True)
        variance_array[variance_array > threshold] = threshold

    variance_array = np.nan_to_num(variance_array, nan=10)
    print(variance_array.shape, flush=True)

    # 4. Пулинг
    var_pooled = mean_pool2d(variance_array, pool_size)
    print(var_pooled.shape, flush=True)

    # 5. Сохраняем результат
    out_path = f"{project_folder}/slope_disp.tif"
    drv = gdal.GetDriverByName("GTiff")
    out = drv.Create(
        out_path, var_pooled.shape[1], var_pooled.shape[0], 1, gdal.GDT_Float32
    )

    # скорректировать геотрансформацию для нового размера
    old_h, old_w = slope_arr.shape
    nh, nw = var_pooled.shape
    gt_slope = mem_slope.GetGeoTransform()
    new_gt = (
        gt_slope[0],
        gt_slope[1] * (old_w / nw),
        gt_slope[2],
        gt_slope[3],
        gt_slope[4],
        gt_slope[5] * (old_h / nh),
    )
    out.SetGeoTransform(new_gt)
    out.SetProjection(crs)

    band = out.GetRasterBand(1)
    band.SetNoDataValue(-1)
    band.WriteArray(var_pooled)
    band.FlushCache()
    band.ComputeStatistics(False)
    band.SetStatistics(
        float(np.nanmin(var_pooled)),
        float(np.nanmax(var_pooled)),
        float(np.nanmean(var_pooled)),
        float(np.std(var_pooled)),
    )

    out.SetMetadataItem("STATISTICS_MINIMUM", "0")
    out.SetMetadataItem("STATISTICS_MAXIMUM", str(float(np.nanmax(var_pooled))))
    out.SetMetadataItem("STATISTICS_MEAN", str(float(np.nanmean(var_pooled))))

    # Закрываем
    src_ds = None
    mem_utm = None
    mem_slope = None
    out = None

    return out_path


def least_cost_path_analysis(project_folder):
    # Получение необходимых слоев
    try:
        points_layer = QgsProject.instance().mapLayersByName("MaxHeightPoints")[0]
    except IndexError:
        QMessageBox.warning(None, "Ошибка", "Слой 'MaxHeightPoints' не найден.")
        return

    disp_layer_path = generate_cost_layer(
        f"{project_folder}/srtm_output.tif", project_folder, 3, 3
    )
    cost_layer = QgsRasterLayer(disp_layer_path, "Slope Disp Layer")
    if cost_layer.isValid():
        QgsProject.instance().addMapLayer(cost_layer)
        print("Создан слой уклона", flush=True)
    else:
        QMessageBox.warning(None, "Ошибка", "Не удалось загрузить слой крутизны.")
        return

    print(f"Старт: {datetime.now()}", flush=True)

    # строим граф из cost_layer
    G, gt, n_rows, n_cols = build_cost_graph(cost_layer.source())

    end_nodes = {}
    for feat in points_layer.getFeatures():
        if feat["z"] is None:
            continue

        x = feat.geometry().asPoint().x()
        y = feat.geometry().asPoint().y()
        i, j = coord_to_pixel(x, y, gt, n_rows, n_cols)
        end_nodes[feat.id()] = i * n_cols + j

    # Собираем список терминалов и их узловых индексов
    fid_to_node = {}
    terminal_fids = []
    for feat in points_layer.getFeatures():
        if feat["z"] is None:
            continue
        fid = feat.id()
        x, y = feat.geometry().asPoint().x(), feat.geometry().asPoint().y()
        i, j = coord_to_pixel(x, y, gt, n_rows, n_cols)
        node = i * n_cols + j
        fid_to_node[fid] = node
        terminal_fids.append(fid)

    uri = f"LineString?crs={points_layer.crs().authid()}"
    paths_layer = QgsVectorLayer(uri, "Output least cost path", "memory")
    dp = paths_layer.dataProvider()
    dp.addAttributes(
        [
            QgsField("start_id", QVariant.Int),
            QgsField("end_id", QVariant.Int),
        ]
    )
    paths_layer.updateFields()

    for src_fid in terminal_fids:
        src_node = fid_to_node[src_fid]
        dijk = nk.distance.Dijkstra(G, src_node)
        dijk.run()

        for dst_fid in terminal_fids:
            if dst_fid == src_fid:
                continue

            dst_node = fid_to_node[dst_fid]
            node_path = dijk.getPath(dst_node)
            if not node_path:
                continue

            # Конвертируем узлы в координаты
            path_pts = [
                QgsPointXY(*pixel_to_coord(u // n_cols, u % n_cols, gt))
                for u in node_path
            ]

            feat_out = QgsFeature(paths_layer.fields())
            feat_out.setGeometry(QgsGeometry.fromPolylineXY(path_pts))
            feat_out["start_id"] = src_fid
            feat_out["end_id"] = dst_fid
            dp.addFeature(feat_out)

    paths_layer.updateExtents()
    QgsProject.instance().addMapLayer(paths_layer)
    print("Создан слой с путями", flush=True)

    print(f"Конец: {datetime.now()}", flush=True)

    # Вспомогательная функция для расчёта минимальной высоты вдоль линии
    def calculate_minimum_elevation(raster_layer, line_geom):
        provider = raster_layer.dataProvider()
        min_elev = float("inf")
        if line_geom.isMultipart():
            lines = line_geom.asMultiPolyline()
        else:
            lines = [line_geom.asPolyline()]
        for line in lines:
            for pt in line:
                sample = provider.sample(QgsPointXY(pt.x(), pt.y()), 1)
                if sample:
                    value, valid = sample
                    if valid and value is not None:
                        min_elev = min(min_elev, value)
        return min_elev if min_elev != float("inf") else None

    try:
        elevation_layer = QgsProject.instance().mapLayersByName("SRTM DEM Layer")[0]
    except IndexError:
        QMessageBox.warning(None, "Ошибка", "Слой 'SRTM DEM Layer' не найден.")
        return

    # Фильтрация путей по критерию разницы высот
    paths_to_delete = []
    for feature in paths_layer.getFeatures():
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
        paths_layer.startEditing()
        for fid in paths_to_delete:
            paths_layer.deleteFeature(fid)
        paths_layer.commitChanges()
        QMessageBox.information(
            None,
            "Информация",
            f"Удалено {len(paths_to_delete)} путей по критерию высоты.",
        )

    # Фильтрация путей, пересекающих реки (исключая совпадающие начала/концы)
    try:
        rivers_layer = QgsProject.instance().mapLayersByName("rivers_and_points")[0]
    except IndexError:
        QMessageBox.warning(None, "Ошибка", "Слой 'rivers_and_points' не найден.")
        return

    spatial_index = QgsSpatialIndex(rivers_layer.getFeatures())
    paths_to_delete = []
    for feature in paths_layer.getFeatures():
        geom = feature.geometry()
        if geom.isEmpty():
            continue

        pts = geom.asPolyline()
        if not pts:
            continue

        start_geom = QgsGeometry.fromPointXY(pts[0])
        end_geom = QgsGeometry.fromPointXY(pts[-1])

        candidate_ids = spatial_index.intersects(geom.boundingBox())
        for cid in candidate_ids:
            candidate = rivers_layer.getFeature(cid)
            if geom.intersects(candidate.geometry()):
                if not (
                    start_geom.intersects(candidate.geometry())
                    or end_geom.intersects(candidate.geometry())
                ):
                    paths_to_delete.append(feature.id())
                break

    if paths_to_delete:
        paths_layer.startEditing()
        for fid in paths_to_delete:
            paths_layer.deleteFeature(fid)
        paths_layer.commitChanges()
        QMessageBox.information(
            None,
            "Информация",
            f"Удалено {len(paths_to_delete)} путей, пересекающих реки.",
        )


def build_cost_graph(raster_path, eps=1e-6):
    ds = gdal.Open(raster_path)
    arr = ds.GetRasterBand(1).ReadAsArray().astype(float)
    rows, cols = arr.shape
    G = nk.Graph(rows * cols, weighted=True, directed=False)

    # все 8 направлений: 4 ортогональных + 4 диагональных
    neigh = [
        (-1, 0, 1.0),  # вверх
        (1, 0, 1.0),  # вниз
        (0, -1, 1.0),  # влево
        (0, 1, 1.0),  # вправо
        (-1, -1, math.sqrt(2)),  # диагонали
        (-1, 1, math.sqrt(2)),
        (1, -1, math.sqrt(2)),
        (1, 1, math.sqrt(2)),
    ]

    def nid(i, j):
        return i * cols + j

    for i in range(rows):
        for j in range(cols):
            u = nid(i, j)
            hu = arr[i, j]
            for di, dj, factor in neigh:
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    hv = arr[ni, nj]
                    dh = abs(hu - hv)
                    w = factor * (dh + eps)
                    G.addEdge(u, nid(ni, nj), w)

    gt = ds.GetGeoTransform()
    return G, gt, rows, cols


def coord_to_pixel(x, y, gt, n_rows, n_cols):
    inv = 1.0 / (gt[1] * gt[5] - gt[2] * gt[4])
    j_float = inv * (gt[5] * (x - gt[0]) - gt[2] * (y - gt[3]))
    i_float = inv * (-gt[4] * (x - gt[0]) + gt[1] * (y - gt[3]))
    i = int(round(i_float))
    j = int(round(j_float))
    i = min(max(i, 0), n_rows - 1)
    j = min(max(j, 0), n_cols - 1)
    return i, j


def pixel_to_coord(i, j, gt):
    x = gt[0] + j * gt[1] + i * gt[2]
    y = gt[3] + j * gt[4] + i * gt[5]
    return x, y
