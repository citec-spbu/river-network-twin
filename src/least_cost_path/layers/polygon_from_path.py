import numpy as np
from osgeo import gdal
import queue
import os
from qgis.core import (
    QgsProject,
    QgsRasterLayer,
    QgsVectorLayer,
)
import processing

def flood_fill_areas(project_folder):
    # Загрузка необходимых слоев
    water_raster_path = f"{project_folder}/water_rasterized.tif"
    lcp_vector_path = os.path.join(project_folder, "output_least_cost_path.gpkg")
    
    # Загрузка растра воды
    ds_water = gdal.Open(water_raster_path)
    band_water = ds_water.GetRasterBand(1)
    arr_water = band_water.ReadAsArray()
    gt = ds_water.GetGeoTransform()
    nodata_water = band_water.GetNoDataValue()
    rows, cols = arr_water.shape
    
    # Создание маски путей (1 - путь, 0 - нет пути)
    mask_path = np.zeros_like(arr_water, dtype=np.uint8)
    lcp_layer = QgsVectorLayer(lcp_vector_path, "Least Cost Paths", "ogr")
    
    if lcp_layer.isValid():
        # Растеризация путей
        for feature in lcp_layer.getFeatures():
            geom = feature.geometry()
            if not geom.isMultipart():
                polyline = geom.asPolyline()
                for point in polyline:
                    px = int((point.x() - gt[0]) / gt[1])
                    py = int((point.y() - gt[3]) / gt[5])
                    if 0 <= px < cols and 0 <= py < rows:
                        mask_path[py, px] = 1
            else:
                for part in geom.asMultiPolyline():
                    for point in part:
                        px = int((point.x() - gt[0]) / gt[1])
                        py = int((point.y() - gt[3]) / gt[5])
                        if 0 <= px < cols and 0 <= py < rows:
                            mask_path[py, px] = 1

    # Создание массивов для результатов
    filled_areas = np.zeros_like(arr_water, dtype=np.uint8)
    visited = np.zeros_like(arr_water, dtype=bool)
    region_counter = 1

    # Направления соседей (4-связность)
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    # Поиск всех водных пикселей
    water_pixels = np.argwhere((arr_water != nodata_water) & (arr_water > 0))
    
    for y, x in water_pixels:
        if visited[y, x] or filled_areas[y, x] > 0:
            continue
            
        # Запуск заливки из текущего пикселя воды
        q = queue.Queue()
        q.put((y, x))
        visited[y, x] = True
        current_region = []
        hit_path = False

        while not q.empty():
            cy, cx = q.get()
            current_region.append((cy, cx))
            
            # Проверка на пересечение с путем
            if mask_path[cy, cx] == 1:
                hit_path = True
                continue
                
            # Проверка соседей
            for dy, dx in directions:
                ny, nx = cy + dy, cx + dx
                
                # Проверка границ растра
                if ny < 0 or ny >= rows or nx < 0 or nx >= cols:
                    continue
                
                if not visited[ny, nx]:
                    visited[ny, nx] = True
                    
                    # Пропускаем пиксели путей и уже заполненные области
                    if mask_path[ny, nx] == 1 or filled_areas[ny, nx] > 0:
                        continue
                    
                    q.put((ny, nx))
        
        # Заполнение области если не достигли пути
        if not hit_path:
            for y_fill, x_fill in current_region:
                filled_areas[y_fill, x_fill] = 1  # Используем 1 для всех закрашенных областей

    # Создаем бинарный растр: 1 - закрашенные области, 0 - остальное
    binary_filled = np.where(filled_areas > 0, 1, 0).astype(np.uint8)
    
    # Инвертируем растр: меняем 0 и 1 местами
    inverted_binary = np.where(binary_filled == 1, 0, 1).astype(np.uint8)
    
    # Сохранение инвертированного результата
    driver = gdal.GetDriverByName("GTiff")
    output_path = f"{project_folder}/inverted_flood_fill.tif"
    ds_out = driver.Create(
        output_path,
        ds_water.RasterXSize,
        ds_water.RasterYSize,
        1,
        gdal.GDT_Byte
    )
    ds_out.SetGeoTransform(gt)
    ds_out.SetProjection(ds_water.GetProjection())
    band_out = ds_out.GetRasterBand(1)
    band_out.WriteArray(inverted_binary)
    band_out.SetNoDataValue(0)
    
    ds_water = None
    ds_out = None
    

    # Добавление слоя в проект QGIS
    layer = QgsRasterLayer(output_path, "Inverted Flood Fill Areas")
    if layer.isValid():
        QgsProject.instance().addMapLayer(layer)
        # Обработка растра
        process_flood_fill(project_folder)
        return layer
    else:
        raise RuntimeError("Failed to load inverted flood fill areas layer")

def process_flood_fill(project_folder):
    # Путь к инвертированному растру
    input_raster = f"{project_folder}/inverted_flood_fill.tif"
    
    # Шаг 1: Растеризация -> Полигоны
    polygonized = processing.run("gdal:polygonize", {
        'INPUT': input_raster,
        'BAND': 1,
        'FIELD': 'DN',
        'EIGHT_CONNECTEDNESS': False,
        'OUTPUT': 'TEMPORARY_OUTPUT'
    })['OUTPUT']
    
    # Шаг 2: Границы полигонов
    boundary = processing.run("native:boundary", {
        'INPUT': polygonized,
        'OUTPUT': 'TEMPORARY_OUTPUT'
    })['OUTPUT']
    
    # Шаг 3: Полигонизация границ
    repolygonized = processing.run("native:polygonize", {
        'INPUT': boundary,
        'KEEP_FIELDS': False,
        'OUTPUT': 'TEMPORARY_OUTPUT'
    })['OUTPUT']
    
    # Шаг 4: Объединение полигонов
    dissolved = processing.run("native:dissolve", {
        'INPUT': repolygonized,
        'FIELD': [],
        'SEPARATE_DISJOINT': False,
        'OUTPUT': 'TEMPORARY_OUTPUT'
    })['OUTPUT']
    
    # Шаг 5: Разделение мультиполигонов
    single_parts = processing.run("native:multiparttosingleparts", {
        'INPUT': dissolved,
        'OUTPUT': 'TEMPORARY_OUTPUT'
    })['OUTPUT']
    
    # Шаг 6: Добавление поля с площадью
    with_area = processing.run("qgis:exportaddgeometrycolumns", {
        'INPUT': single_parts,
        'CALC_METHOD': 0,  # Площадь в CRS слоя
        'OUTPUT': 'TEMPORARY_OUTPUT'
    })['OUTPUT']
    
    # Шаг 7: Фильтрация объектов
    final_path = f"{project_folder}/final_areas.gpkg"
    processing.run("native:extractbyexpression", {
        'INPUT': with_area,
        'EXPRESSION': f'"area" >= {33111 * 15}',
        'OUTPUT': final_path
    })
    
    # Загрузка финального слоя
    final_layer = QgsVectorLayer(final_path, "Final Flood Fill Areas", "ogr")
    if final_layer.isValid():
        QgsProject.instance().addMapLayer(final_layer)
        return final_layer
    else:
        raise RuntimeError("Failed to load final areas layer")