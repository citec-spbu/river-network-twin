import queue
import numpy as np
from osgeo import gdal
from qgis.core import (
    QgsVectorLayer,
    QgsProcessing
)
import processing

def flood_fill_areas(raster_path, vector_path, output_raster, feedback=None):
    """
    Основная функция заливки областей
    """
    # Загрузка растра воды
    ds_water = gdal.Open(raster_path)
    band_water = ds_water.GetRasterBand(1)
    arr_water = band_water.ReadAsArray()
    gt = ds_water.GetGeoTransform()
    nodata_water = band_water.GetNoDataValue()
    rows, cols = arr_water.shape
    
    # Создание маски путей (1 - путь, 0 - нет пути)
    mask_path = np.zeros_like(arr_water, dtype=np.uint8)
    lcp_layer = QgsVectorLayer(vector_path, "Least Cost Paths", "ogr")
    
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

    # Направления соседей (4-связность)
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    # Поиск всех водных пикселей
    water_pixels = np.argwhere((arr_water != nodata_water) & (arr_water > 0))
    total = len(water_pixels)

    for i, (y, x) in enumerate(water_pixels):
        if feedback and feedback.isCanceled():
            break
            
        if feedback:
            feedback.setProgress(int(i / total * 50))
        
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
    ds_out = driver.Create(output_raster, cols, rows, 1, gdal.GDT_Byte)
    ds_out.SetGeoTransform(gt)
    ds_out.SetProjection(ds_water.GetProjection())
    band_out = ds_out.GetRasterBand(1)
    band_out.WriteArray(inverted_binary)
    band_out.SetNoDataValue(0)
    
    ds_water = None
    ds_out = None
    

    return output_raster

def process_flood_fill(input_raster, output_vector, min_area, feedback=None):
    """
    Функция постобработки растра
    """
    
    # Шаг 1: Растеризация -> Полигоны
    polygonized = processing.run("gdal:polygonize", {
        'INPUT': input_raster,
        'BAND': 1,
        'FIELD': 'DN',
        'EIGHT_CONNECTEDNESS': False,
        'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT
    }, feedback=feedback)['OUTPUT']
    
    # Шаг 2: Границы полигонов
    boundary = processing.run("native:boundary", {
        'INPUT': polygonized,
        'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT
    }, feedback=feedback)['OUTPUT']
    
    # Шаг 3: Полигонизация границ
    repolygonized = processing.run("native:polygonize", {
        'INPUT': boundary,
        'KEEP_FIELDS': False,
        'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT
    }, feedback=feedback)['OUTPUT']
    
    # Шаг 4: Объединение полигонов
    dissolved = processing.run("native:dissolve", {
        'INPUT': repolygonized,
        'FIELD': [],
        'SEPARATE_DISJOINT': False,
        'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT
    }, feedback=feedback)['OUTPUT']
    
    # Шаг 5: Разделение мультиполигонов
    single_parts = processing.run("native:multiparttosingleparts", {
        'INPUT': dissolved,
        'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT
    }, feedback=feedback)['OUTPUT']
    
    # Шаг 6: Добавление поля с площадью
    with_area = processing.run("qgis:exportaddgeometrycolumns", {
        'INPUT': single_parts,
        'CALC_METHOD': 0,  # Площадь в CRS слоя
        'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT
    }, feedback=feedback)['OUTPUT']
    
    # Шаг 7: Фильтрация объектов
    processing.run("native:extractbyexpression", {
        'INPUT': with_area,
        'EXPRESSION': f'"area" >= {min_area}',
        'OUTPUT': output_vector
    }, feedback=feedback)
    
    
    return output_vector