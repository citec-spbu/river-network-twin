from qgis.core import (
    QgsVectorLayer,
    QgsField,
    QgsFeature,
    QgsGeometry,
    QgsSpatialIndex,
    QgsProject,
)
import processing
from qgis.PyQt.QtCore import QVariant
import os

def preparing_data_for_clustering(point_layer, dem_layer, RESAMPLE_SCALE, CONTOUR_INTERVAL, project_folder):
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

def assign_clusters(data_for_clustering, point_layer):
    # Создаем поле cluster, если его нет
    if point_layer.fields().indexFromName('cluster') == -1:
        point_layer.startEditing()
        point_layer.addAttribute(QgsField('cluster', QVariant.Int))
        point_layer.commitChanges()

    # Подготовка словаря полигонов (все ID как строки)
    polygons = {}
    for feat in data_for_clustering.getFeatures():
        poly_id = str(feat["fid"])  # Приводим к строке
        polygons[poly_id] = {
            'feature': feat,
            'z': feat['z'],
            'children': [],
            'points': []
        }

    # Заполняем дочерние полигоны и точки (ID как строки)
    for poly_id, data in polygons.items():
        feat = data['feature']
        
        # Обрабатываем ID_child
        if feat['id_child']:
            children = [
                c.strip() for c in str(feat['id_child']).split(',') 
                if c.strip() in polygons  # Только существующие ID
            ]
            data['children'] = children
        
        # Обрабатываем Arr_point
        if feat['arr_point']:
            data['points'] = [
                p.strip() for p in str(feat['arr_point']).split(',')
            ]

    # Рекурсивная функция для определения кластера
    def get_final_cluster(current_poly_id, point_geom):
        current_poly = polygons.get(current_poly_id)
        if not current_poly:
            return None

        children = current_poly['children']
        
        if not children:
            return int(current_poly_id)  # Возвращаем число, если cluster должен быть int
        elif len(children) == 1:
            return get_final_cluster(children[0], point_geom)
        else:
            # Ищем ближайший дочерний полигон
            min_dist = float('inf')
            closest_child = None
            for child_id in children:
                child_poly = polygons.get(child_id)
                if not child_poly:
                    continue
                child_geom = child_poly['feature'].geometry()
                dist = point_geom.distance(child_geom)
                if dist < min_dist:
                    min_dist = dist
                    closest_child = child_id
            return get_final_cluster(closest_child, point_geom) if closest_child else None

    # Обновляем точки
    point_layer.startEditing()
    # Группируем полигоны по z
    z_groups = {}
    for poly_id, data in polygons.items():
        z = data['z']
        if z not in z_groups:
            z_groups[z] = []
        z_groups[z].append(poly_id)

    # Обрабатываем полигоны от максимального z к минимальному
    for z in sorted(z_groups.keys(), reverse=True):
        for poly_id in z_groups[z]:
            points_ids = polygons[poly_id]['points']
            for point_id in points_ids:
                # Ищем точку (ID_point как строка)
                point_feat = next(
                    point_layer.getFeatures(f"point_id = '{point_id}'"), 
                    None
                )
                if not point_feat:
                    continue

                # Определяем кластер
                cluster_id = get_final_cluster(poly_id, point_feat.geometry())
                if cluster_id is not None:
                    point_layer.changeAttributeValue(
                        point_feat.id(),
                        point_layer.fields().indexFromName('cluster'),
                        cluster_id
                    )

    point_layer.commitChanges()

    return point_layer     