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
    QgsSpatialIndex,
)
from qgis.PyQt.QtCore import QVariant, QEventLoop
from qgis.PyQt.QtWidgets import QMessageBox, QInputDialog
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

    data_for_clustering = preparing_data_for_clustering(point_layer, dem_layer)

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

def preparing_data_for_clustering(point_layer, dem_layer):
    # ========== НАСТРОЙКИ ПАРАМЕТРОВ ==========
    RESAMPLE_SCALE = 5                # Параметр масштаба для ресемплинга (2-10)(Для более гладких изолиний увеличить значение)
    CONTOUR_INTERVAL = 20             # Интервал изолиний в метрах
    # ==========================================

    # Создание ID для точек
    point_layer.startEditing()
    if "point_id" not in [f.name() for f in point_layer.fields()]:
        point_layer.dataProvider().addAttributes(
            [QgsField("point_id", QVariant.Int)]
        )
        point_layer.updateFields()  
    point_id_idx = point_layer.fields().indexOf("point_id")

    for i, feat in enumerate(point_layer.getFeatures(), start=1):
        point_layer.changeAttributeValue(feat.id(), point_id_idx, i)
    point_layer.commitChanges()

    
    # Применение Resampling Filter
    output_dem = processing.run("sagang:resamplingfilter", {
        'GRID': dem_layer.source(),
        'LOPASS': 'TEMPORARY_OUTPUT',
        'HIPASS': 'TEMPORARY_OUTPUT',
        'SCALE': RESAMPLE_SCALE
    })

    # Создание изолиний
    contours = processing.run("gdal:contour_polygon", {
        'INPUT': output_dem['LOPASS'],
        'BAND': 1,
        'INTERVAL': CONTOUR_INTERVAL,
        'FIELD_NAME_MIN': 'ELEV_MIN',
        'FIELD_NAME_MAX': 'ELEV_MAX',
        'OUTPUT': 'TEMPORARY_OUTPUT'
    })['OUTPUT']    

    # Разделение составной геометрии
    single_parts = processing.run("native:multiparttosingleparts", {
        'INPUT': contours,
        'OUTPUT': 'TEMPORARY_OUTPUT'
    })['OUTPUT']

    # Подсчет точек в каждом полигоне
    count_result = processing.run("native:countpointsinpolygon", {
        'POLYGONS': single_parts,
        'POINTS': point_layer,
        'OUTPUT': 'TEMPORARY_OUTPUT'
    })['OUTPUT']

    # Удаление полигонов без точек
    selection = count_result.getFeatures("\"NUMPOINTS\" = 0 OR \"NUMPOINTS\" IS NULL")
    count_result.selectByIds([f.id() for f in selection])

    eliminated = processing.run("qgis:eliminateselectedpolygons", {
        'INPUT': count_result,
        'MODE': 0,
        'OUTPUT': 'TEMPORARY_OUTPUT'
    })['OUTPUT']
    
    # Обработка уровней высот
    elev_values = sorted({f['ELEV_MAX'] for f in eliminated.getFeatures()})
    merged_layers = []

    for z_value, current_elev in enumerate(elev_values, 1):
        selected = [f.id() for f in eliminated.getFeatures(f'"ELEV_MAX" >= {current_elev}')]
        eliminated.selectByIds(selected)

        # Создаем временный слой с выбранными объектами
        selected_layer = processing.run("native:saveselectedfeatures", {
            'INPUT': eliminated,
            'OUTPUT': 'memory:'
        })['OUTPUT']
        
        # Проверка валидности геометрии
        processing.run("native:fixgeometries", {
            'INPUT': selected_layer,
            'OUTPUT': 'memory:'
        })
        
        # Слияние через coverageunion
        merged = processing.run("native:coverageunion", {
            'INPUT': selected_layer,
            'OUTPUT': 'TEMPORARY_OUTPUT'
        })['OUTPUT']
        
        merged.startEditing()
        # Получаем список индексов полей для удаления
        fields_to_delete = [
            idx for idx, field in enumerate(merged.fields())
            if field.name() != "z"
        ]

        # Удаляем поля
        for idx in reversed(fields_to_delete):
            merged.deleteAttribute(idx)
        merged.updateFields() 

        if "z" not in [f.name() for f in merged.fields()]:
            merged.dataProvider().addAttributes(
                [QgsField("z", QVariant.Int)]
            )
            merged.updateFields()
        z_idx = merged.fields().indexOf("z")

        for feat in merged.getFeatures():
            merged.changeAttributeValue(feat.id(), z_idx, z_value)
        merged.commitChanges()

        merged_layers.append(merged)

    # Объединение слоев
    final_layer = processing.run("native:mergevectorlayers", {
        'LAYERS': merged_layers,
        'CRS': None,
        'OUTPUT': 'TEMPORARY_OUTPUT',
        'ADD_SOURCE_FIELDS': False
    })['OUTPUT']

    # Разделение составной геометрии
    result_layer = processing.run("native:multiparttosingleparts", {
        'INPUT': final_layer,
        'OUTPUT': 'TEMPORARY_OUTPUT'
    })['OUTPUT']
    
    # Добавляем поле ID_self (если отсутствует)
    result_layer.startEditing()
    if "fid" not in [f.name() for f in result_layer.fields()]:
        result_layer.dataProvider().addAttributes(
            [QgsField("fid", QVariant.Int)]
        )
        result_layer.updateFields()    
    fid_idx = result_layer.fields().indexOf("fid")
    for i, feat in enumerate(result_layer.getFeatures(), start=1):
        result_layer.changeAttributeValue(feat.id(), fid_idx, i)

    # Добавляем поле ID_child (если отсутствует)
    if "id_child" not in [f.name() for f in result_layer.fields()]:
        result_layer.dataProvider().addAttributes(
            [QgsField("id_child", QVariant.String, len=255)]
        )
        result_layer.updateFields()
    id_child_idx = result_layer.fields().indexOf("id_child")


    ## Создаем структуры для быстрого поиска
    z_dict = {}  # {z: [список объектов]}
    feature_map = {}  # {id объекта: объект}


    ## Заполнение структур данных
    for feat in result_layer.getFeatures():
        z = feat["z"]
        z_dict.setdefault(z, []).append(feat)
        feature_map[feat.id()] = feat


    attrs_to_update = {}

    ## Обрабатываем каждый полигон
    for current_feat in result_layer.getFeatures():
        current_z = current_feat['z']
        target_z = current_z + 1
        current_geom = current_feat.geometry()
        child_ids = []
        
        if target_z in z_dict:
            for candidate_feat in z_dict[target_z]:
                if candidate_feat.geometry().intersects(current_geom):
                    child_ids.append(str(candidate_feat["fid"]))
        
        child_ids_str = ','.join(child_ids) if child_ids else None
        attrs_to_update[current_feat.id()] = {id_child_idx: child_ids_str}

    result_layer.dataProvider().changeAttributeValues(attrs_to_update)
    
    # Проверка и добавление поля
    arr_point_idx = result_layer.fields().lookupField("arr_point")
    if arr_point_idx == -1:
        result_layer.dataProvider().addAttributes([QgsField("arr_point", QVariant.String)])
        result_layer.updateFields()
        arr_point_idx = result_layer.fields().lookupField("arr_point")

    ## Создаем структуры данных
    polygon_dict = {feat["fid"]: [] for feat in result_layer.getFeatures()}
    polygon_index = QgsSpatialIndex()
    feature_map = {}

    ## Заполняем пространственный индекс
    for poly_feat in result_layer.getFeatures():
        polygon_index.addFeature(poly_feat)
        feature_map[poly_feat.id()] = poly_feat

    ## Пакетное обновление атрибутов
    attrs_to_update = {}

    ## Обработка точек
    for point_feat in point_layer.getFeatures():
        point_geom = point_feat.geometry()
        if not point_geom.isGeosValid():
            continue

        ## Поиск пересечений
        candidate_ids = polygon_index.intersects(point_geom.boundingBox())
        containing_polys = []
        
        for fid in candidate_ids:
            poly_feat = feature_map[fid]
            if poly_feat.geometry().contains(point_geom):
                containing_polys.append(poly_feat)

        if containing_polys:
            ## Выбор полигона с максимальным z
            selected = max(containing_polys, key=lambda x: x['z'])
            polygon_dict[selected['fid']].append(str(point_feat['point_id']))

    ## Формируем атрибуты для обновления
    for poly_feat in result_layer.getFeatures():
        fid = poly_feat['fid']
        points = polygon_dict.get(fid, [])
        attrs_to_update[poly_feat.id()] = {
            arr_point_idx: ','.join(points) if points else None
        }

    # Применяем изменения
    result_layer.dataProvider().changeAttributeValues(attrs_to_update)

    result_layer.commitChanges()
    
    return result_layer

            

