import math
import time
from qgis.core import (
    QgsField,
    QgsProject,
    QgsPointXY,
    QgsGeometry,
    QgsSpatialIndex,
    QgsRasterLayer,
    QgsVectorLayer,
    QgsFeature,
    QgsCoordinateTransform,
    QgsCoordinateReferenceSystem,
)
from qgis.PyQt.QtCore import QVariant
from qgis.PyQt.QtWidgets import QMessageBox
import networkit as nk
from osgeo import gdal


def least_cost_path_analysis(project_folder):
    # Получение необходимых слоев
    try:
        points_layer = QgsProject.instance().mapLayersByName("MaxHeightPoints")[0]
    except IndexError:
        QMessageBox.warning(None, "Ошибка", "Слой 'MaxHeightPoints' не найден.")
        return

    src_crs = points_layer.crs()
    tgt_crs = QgsCoordinateReferenceSystem("EPSG:3857")
    transform_context = QgsProject.instance().transformContext()
    coord_transform = QgsCoordinateTransform(src_crs, tgt_crs, transform_context)

    print("Используется DEM в качестве слоя стоимости", flush=True)
    dem_src = f"{project_folder}/srtm_output.tif"
    dem_3857 = f"{project_folder}/srtm_output_3857.tif"
    gdal.Warp(
        dem_3857, dem_src, dstSRS="EPSG:3857", resampleAlg="bilinear", format="GTiff"
    )

    dem_pooled = f"{project_folder}/srtm_output_3857_pooled.tif"
    ds3857 = gdal.Open(dem_3857)
    gt3857 = ds3857.GetGeoTransform()
    orig_xres = gt3857[1]
    orig_yres = abs(gt3857[5])
    gdal.Warp(
        destNameOrDestDS=dem_pooled,
        srcDSOrSrcDSTab=dem_3857,
        xRes=orig_xres * 4,
        yRes=orig_yres * 4,
        resampleAlg="average",
        format="GTiff",
    )

    cost_layer = QgsRasterLayer(dem_pooled, "DEM Cost Layer")
    if not cost_layer.isValid():
        QMessageBox.warning(
            None, "Ошибка", "Не удалось загрузить перепроецированный DEM."
        )
        return

    t_paths_start = time.perf_counter()

    # строим граф из cost_layer
    G, gt, n_rows, n_cols = build_cost_graph(cost_layer.source())

    fid_to_node = {}
    terminal_fids = []
    for feat in points_layer.getFeatures():
        z = feat["z"]
        if z is None:
            continue

        pt = feat.geometry().asPoint()
        pt3857 = coord_transform.transform(pt)
        i, j = coord_to_pixel(pt3857.x(), pt3857.y(), gt, n_rows, n_cols)
        node_idx = i * n_cols + j

        fid_to_node[feat.id()] = node_idx
        terminal_fids.append(feat.id())

    uri = f"LineString?crs=EPSG:3857"
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

    t_paths_end = time.perf_counter()
    print(
        f"Для построения слоя с путями потребовалось: {t_paths_end - t_paths_start:.3f} s",
        flush=True,
    )

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

    t_filter1_start = time.perf_counter()

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

    t_filter1_end = time.perf_counter()
    print(
        f"Время фильтрации путей по критерию разницы высот: {t_filter1_end - t_filter1_start:.3f} s",
        flush=True,
    )

    t_filter2_start = time.perf_counter()

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

    t_filter2_end = time.perf_counter()
    print(
        f"Время фильтрации путей, пересекающих реки: {t_filter2_end - t_filter2_start:.3f} s",
        flush=True,
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
