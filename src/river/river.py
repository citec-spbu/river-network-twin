import os
from typing import List, Optional
import processing
from qgis.core import (
    QgsProject,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsPointXY,
    QgsVectorLayer,
    QgsGeometry,
    QgsField,
    QgsRaster,
    QgsFeature,
)
from qgis.PyQt.QtCore import QVariant, QEventLoop
from qgis.PyQt.QtWidgets import (
    QMessageBox,
    QInputDialog
)
from qgis.utils import iface
from .point_selection_tool import PointSelectionTool
from .layers.rivers_by_object_filtered import build_rivers_by_object_filtered
from ..river.layers.basins import build_basins_layer
from ..river.layers.rivers_and_points import build_rivers_and_points_layer
from ..river.layers.rivers_merged import build_merged_layer
from ..river.layers.utils import load_quickosm_layer
from ..common import *

RIVER_FILTERS = {
    "max_strahler_order": (">=", 2),
    "total_length": (">", 1000),
}

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


def river(project_folder):
    set_project_crs()
    enable_processing_algorithms()
    add_opentopo_layer()

    bbox = select_analysis_bbox()
    if bbox is None:
        return

    dem_path = download_dem(bbox, project_folder)
    dem_layer = add_dem_layer(dem_path)
    reprojected_relief = reproject_dem(dem_path)

    # Добавить заполненные области водосбора в проект
    basins_path = os.path.join(project_folder, "basins.sdat")
    basins = build_basins_layer(reprojected_relief, basins_path)
    QgsProject.instance().addMapLayer(basins)

    # Добавить объединенный слой рек и ручьев в проект
    # Использовать QuickOSM для запроса данных о водных путях на заданной территории
    extent = transform_bbox(bbox[0], bbox[2], bbox[1], bbox[3], 4326, 3857)
    merged_path = os.path.join(project_folder, "merge_result.gpkg")
    rivers_merged = build_merged_layer(extent, merged_path)
    QgsProject.instance().addMapLayer(rivers_merged)

    # Загрузить полигональные данные о водных объектах
    water_output = os.path.join(project_folder, "water.gpkg")
    load_quickosm_layer(
        "water",
        "natural",
        "water",
        extent,
        output_path=water_output,
        quickosm_layername="multipolygons",
    )

    # Рассчитать координаты начальных и конечных точек линий
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

    # Добавить новые поля для хранения высотных данных
    layer_provider = end_y.dataProvider()
    layer_provider.addAttributes(
        [QgsField("start_z", QVariant.Double), QgsField("end_z", QVariant.Double)]
    )
    end_y.updateFields()
    idx_start_z = layer_provider.fields().indexOf("start_z")
    idx_end_z = layer_provider.fields().indexOf("end_z")

    # Начать редактирование и заполнение значений высоты
    end_y.startEditing()
    line_provider = end_y.dataProvider()
    changes = {}
    for feature in end_y.getFeatures():
        geom = feature.geometry()
        if geom.isMultipart():
            polyline = geom.asMultiPolyline()[0]
        else:
            polyline = geom.asPolyline()

        start_point = QgsPointXY(polyline[0])
        end_point = QgsPointXY(polyline[-1])

        # Высотные данные начальной точки
        start_z = dem_layer.dataProvider().identify(
            start_point, QgsRaster.IdentifyFormatValue
        )
        # Высотные данные конечной точки
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

    # Добавим слой рек, где объекты - сами реки, а не сегменты.
    rivers_by_object_filtered_path = os.path.join(
        project_folder, "rivers_by_object_filtered.gpkg"
    )
    rivers_by_object_filtered = build_rivers_by_object_filtered(
        end_y, RIVER_FILTERS, rivers_by_object_filtered_path
    )
    QgsProject.instance().addMapLayer(rivers_by_object_filtered)

    # Определить максимальную высоту для каждой линии
    rivers_and_points_path = os.path.join(project_folder, "rivers_with_points.gpkg")
    rivers_and_points = build_rivers_and_points_layer(end_y, rivers_and_points_path)
    QgsProject.instance().addMapLayer(rivers_and_points)

    # Создать слой точек максимальной высоты
    point_layer = QgsVectorLayer("Point?crs=epsg:4326", "MaxHeightPoints", "memory")
    QgsProject.instance().addMapLayer(point_layer)
    layer_provider = point_layer.dataProvider()
    layer_provider.addAttributes(
        [
            QgsField("x", QVariant.Double),
            QgsField("y", QVariant.Double),
            QgsField("z", QVariant.Double),
        ]
    )
    point_layer.updateFields()
    fields = point_layer.fields()

    # Сначала собираем все конечные и начальные точки
    start_points = set()
    end_points = set()

    for feat in rivers_and_points.getFeatures():
        sx, sy = feat["start_x"], feat["start_y"]
        ex, ey = feat["end_x"], feat["end_y"]
        if sx is not None and sy is not None:
            start_points.add((sx, sy))
        if ex is not None and ey is not None:
            end_points.add((ex, ey))

    point_layer.startEditing()
    for feat in rivers_and_points.getFeatures():
        max_z = feat["max_z"]
        if max_z is None:
            continue

        sx, sy, start_z = feat["start_x"], feat["start_y"], feat["start_z"]
        ex, ey, end_z = feat["end_x"], feat["end_y"], feat["end_z"]

        # Проверка начальной точки
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

        # Проверка конечной точки
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

    # Завершение редактирования и сохранение изменений
    point_layer.commitChanges(True)
    QgsProject.instance().addMapLayer(point_layer)


def select_analysis_bbox() -> Optional[List[float]]:
    method, ok = QInputDialog.getItem(
        None,
        "Выбор метода",
        "Как определить область анализа?",
        ["Радиус вокруг точки", "Область по 4 точкам"],
        0, False
    )
    if not ok:
        return None

    if method == "Радиус вокруг точки":
        radius, ok = QInputDialog.getDouble(
            None,
            "Радиус вокруг точки",
            "Введите радиус (градусы):",
            value=0.5, min=0.1, max=5, decimals=1
        )
        if not ok:
            return None

        x, y = get_coordinates()
        if x is None or y is None:
            return None

        lon, lat = transform_coordinates(x, y)
        return [lon - radius, lat - radius, lon + radius, lat + radius]

    elif method == "Область по 4 точкам":
        canvas = iface.mapCanvas()
        tool = PointSelectionTool(canvas)
        canvas.setMapTool(tool)

        QMessageBox.information(
            None,
            "Выбор территории",
            "Выберите 4 точки на карте, затем дождитесь завершения."
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
            QgsProject.instance()
        )
        points_4326 = [transform.transform(p) for p in tool.points]

        x_coords = [p.x() for p in points_4326]
        y_coords = [p.y() for p in points_4326]
        return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

    return None