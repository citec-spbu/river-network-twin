from qgis.core import (
    QgsProject,
    QgsPointXY,
    QgsGeometry,
    QgsSpatialIndex,
    QgsRasterLayer,
    QgsVectorLayer,
    QgsMessageLog
)
from qgis.PyQt.QtWidgets import QMessageBox
import processing
import os
import numpy as np
from osgeo import gdal
from datetime import datetime

def mean_pool2d(arr, kernel_size, stride=None):
    """
    Применяет max pooling к 2D массиву.
    
    Параметры:
        arr: 2D NumPy array
        kernel_size: размер ядра (int или tuple) (h, w)
        stride: шаг (int или tuple). Если None, равен kernel_size
    
    Возвращает:
        2D массив после max pooling
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if stride is None:
        stride = kernel_size
    elif isinstance(stride, int):
        stride = (stride, stride)
    
    # Вычисляем размеры выходного массива
    h_out = (arr.shape[0] - kernel_size[0]) // stride[0] + 1
    w_out = (arr.shape[1] - kernel_size[1]) // stride[1] + 1
    
    result = np.zeros((h_out, w_out))
    
    for i in range(h_out):
        for j in range(w_out):
            # Вычисляем окно
            h_start = i * stride[0]
            h_end = h_start + kernel_size[0]
            w_start = j * stride[1]
            w_end = w_start + kernel_size[1]
            
            # Применяем max pooling к окну
            window = arr[h_start:h_end, w_start:w_end]
            result[i, j] = np.mean(window)
    
    return result

# Возвращает путь к полученному файлу
def generate_slope_layer(dem_path: str, project_folder: str) -> str:
    os.makedirs(f"{project_folder}/tmp/", exist_ok=True)

    dem_layer = QgsRasterLayer(dem_path, "DEM")
    crs = dem_layer.crs().authid()
    extent = dem_layer.extent()
    center_longitude = (extent.xMinimum() + extent.xMaximum()) / 2
    utm_zone = int((center_longitude + 180) / 6) + 1
    target_crs = f"EPSG:{32600 + utm_zone}" if extent.yMinimum() >= 0 else f"EPSG:{32700 + utm_zone}"

    # 2. Перепроектируем DEM в UTM
    params_reproject = {
        'INPUT': dem_path,
        'TARGET_CRS': target_crs,
        'RESAMPLING': 1,  # 0 = Nearest Neighbour (для категорийных), 1 = Bilinear (лучше для DEM)
        'OUTPUT': 'TEMPORARY_OUTPUT'
    }
    output_reprojected = processing.run("gdal:warpreproject", params_reproject)['OUTPUT']

    # 3. Рассчитываем уклон (в градусах)
    params_slope = {
        'INPUT': output_reprojected,
        'BAND': 1,
        'SCALE': 1,  # Масштаб (оставьте 1, если пиксели в метрах)
        'AS_PERCENT': False,  # False = градусы, True = проценты
        'ZEVENBERGEN': True,  # False = стандартный алгоритм, True = более гладкий
        'OUTPUT': 'TEMPORARY_OUTPUT'
    }
    output_slope_utm = processing.run("gdal:slope", params_slope)['OUTPUT']

    # возвращаем в исходные координаты
    output_slope = f"{project_folder}/slope_layer.tif"
    params = {
        'INPUT': output_slope_utm,
        'TARGET_CRS': crs,  # Целевая система координат
        'RESAMPLING': 1,  # 1 = Bilinear (лучше для DEM)
        'OUTPUT': output_slope
    }
    processing.run("gdal:warpreproject", params)

    return output_slope

def calculate_local_variance(input_raster_path, output_path, window_size=3, pool_size = 3):
    """
    Рассчитывает локальную дисперсию для растра
    :param input_raster_path: путь к входному растру
    :param output_path: путь для сохранения результата
    :param window_size: размер окна (нечетное число)
    """
    # 1. Загружаем растр через GDAL
    dataset = gdal.Open(input_raster_path)
    band = dataset.GetRasterBand(1)
    raster_array = band.ReadAsArray().astype(np.float32)
    
    # 2. Создаем массив для результатов
    variance_array = np.zeros_like(raster_array)
    pad_size = window_size // 2
    
    # 3. Добавляем границы для обработки краев
    padded = np.pad(raster_array, pad_size, mode='reflect')
    
    # 4. Расчет дисперсии в скользящем окне
    for i in range(pad_size, padded.shape[0] - pad_size):
        for j in range(pad_size, padded.shape[1] - pad_size):
            window = padded[i-pad_size:i+pad_size+1, j-pad_size:j+pad_size+1]
            variance_array[i-pad_size, j-pad_size] = np.var(window)
    
    variance_array[variance_array < 0] = 0
    if np.any(variance_array > 0):
        median_var = np.nanmedian(variance_array)
        mad = 1.4826 * np.nanmedian(np.abs(variance_array - median_var))  # Median Absolute Deviation
        threshold = median_var + 3 * mad
        print(f"Threshold: {threshold}", flush=True)
        variance_array[variance_array > threshold] = threshold
    
    variance_array = np.nan_to_num(variance_array, nan=10)

    print(variance_array.shape, flush=True)
    old_shape = variance_array.shape
    variance_array = mean_pool2d(variance_array, pool_size)
    print(variance_array.shape, flush=True)
    
    # 5. Сохраняем результат
    driver = gdal.GetDriverByName('GTiff')
    out_dataset = driver.Create(
        output_path,
        variance_array.shape[1],
        variance_array.shape[0],
        1,
        gdal.GDT_Float32
    )
    geotransform = dataset.GetGeoTransform()
    new_geotransform = (
        geotransform[0],                 # верхний левый X (не меняем)
        geotransform[1] * old_shape[1] / variance_array.shape[1],   # размер пикселя по X увеличиваем
        geotransform[2],                 # вращение по X (обычно 0)
        geotransform[3],                 # верхний левый Y (не меняем)
        geotransform[4],                 # вращение по Y (обычно 0)
        geotransform[5] * old_shape[0] / variance_array.shape[0]    # размер пикселя по Y увеличиваем (отрицательное значение)
    )
    out_dataset.SetGeoTransform(new_geotransform)
    out_dataset.SetProjection(dataset.GetProjection())
    out_band: gdal.Band = out_dataset.GetRasterBand(1)
    out_band.SetNoDataValue(-1)
    out_band.WriteArray(variance_array)
    out_band.ComputeStatistics(False)
    out_band.SetStatistics(
        float(np.nanmin(variance_array)),       # Минимальное значение
        float(np.nanmax(variance_array)),      # Максимальное значение
        float(np.nanmean(variance_array)),     # Среднее
        float(np.std(variance_array))
    )
    out_band.FlushCache()

    out_dataset.SetMetadataItem("STATISTICS_MINIMUM", "0")  # Явно задаем минимум
    out_dataset.SetMetadataItem("STATISTICS_MAXIMUM", str(np.nanmax(variance_array)))  # Реальный максимум
    out_dataset.SetMetadataItem("STATISTICS_MEAN", str(np.nanmean(variance_array)))
    
    # 6. Закрываем файлы
    dataset = None
    out_dataset = None


def least_cost_path_analysis(project_folder):
    # Получение необходимых слоев
    try:
        points_layer = QgsProject.instance().mapLayersByName('MaxHeightPoints')[0]
    except IndexError:
        QMessageBox.warning(None, "Ошибка", "Слой 'MaxHeightPoints' не найден.")
        return

    try:
        cost_layer = QgsProject.instance().mapLayersByName('Slope Layer')[0]
    except IndexError:
        slope_layer_path = generate_slope_layer(f"{project_folder}/srtm_output.tif", project_folder)
        disp_layer_path = f"{project_folder}/slope_disp.tif"
        calculate_local_variance(slope_layer_path, disp_layer_path, 3, 3)
        cost_layer = QgsRasterLayer(disp_layer_path, 'Slope Disp Layer')
        if cost_layer.isValid():
            QgsProject.instance().addMapLayer(cost_layer)
            print("Создан слой уклона", flush=True)
        else:
            QMessageBox.warning(None, "Ошибка", "Не удалось загрузить слой крутизны.")
            return

    print(f"Старт: {datetime.now()}", flush=True)

    paths_layer = QgsVectorLayer("paths", "Output least cost path", "memory")
    paths_layer.setCrs(points_layer.crs())
    for p in points_layer.getFeatures():
        if p['z'] != None:
            point = p

            start_layer = QgsVectorLayer("Point", "point", "memory")
            start_layer.setCrs(points_layer.sourceCrs()) # set the coordinate system from layer to the new layer
            pr1 = start_layer.dataProvider() # get the layers data provider (can't explain what it is exactly)
            pr1.addFeature(point) # add the feature to the layer via data provider
            start_layer.updateFeature(point)

            print("Построен слой с точкой", flush=True)

            # Параметры для алгоритма Least Cost Path
            params = {
                'INPUT_COST_RASTER': cost_layer,
                'INPUT_RASTER_BAND': 1,
                'INPUT_START_LAYER': start_layer,
                'INPUT_END_LAYER': points_layer,
                'BOOLEAN_FIND_LEAST_PATH_TO_ALL_ENDS': False,
                'BOOLEAN_OUTPUT_LINEAR_REFERENCE': False,
                'OUTPUT': 'TEMPORARY_OUTPUT'
            }
            try:
                result = processing.run("Cost distance analysis:Least Cost Path", params)
                layer = result['OUTPUT']

                params = {
                    'LAYERS': [paths_layer, layer],  # Список слоёв
                    'CRS': paths_layer.crs(),         # Используем CRS первого слоя (можно задать вручную, например, 'EPSG:4326')
                    'OUTPUT': 'memory:'          # Результат в памяти (можно сохранить в файл: 'C:/путь/к/файлу.shp')
                }

                # Запуск обработки
                result = processing.run("native:mergevectorlayers", params)
                paths_layer = result['OUTPUT']
            except Exception as e:
                print(f"Ошибка при выполнении анализа: {e}", flush=True)
            
    QgsProject.instance().addMapLayer(paths_layer)
    print("Создан слой с путями", flush=True)
    
    print(f"Конец: {datetime.now()}", flush=True)

    # Вспомогательная функция для расчёта минимальной высоты вдоль линии
    def calculate_minimum_elevation(raster_layer, line_geom):
        provider = raster_layer.dataProvider()
        min_elev = float('inf')
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
        return min_elev if min_elev != float('inf') else None

    try:
        elevation_layer = QgsProject.instance().mapLayersByName('SRTM DEM Layer')[0]
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
        if geom.isMultipart():
            polyline = geom.asMultiPolyline()[0]
        else:
            polyline = geom.asPolyline()
        if not polyline:
            continue
        first_point = polyline[0]
        last_point = polyline[-1]
        sample_start = elevation_layer.dataProvider().sample(QgsPointXY(first_point.x(), first_point.y()), 1)
        sample_end = elevation_layer.dataProvider().sample(QgsPointXY(last_point.x(), last_point.y()), 1)
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
        QMessageBox.information(None, "Информация", f"Удалено {len(paths_to_delete)} путей по критерию высоты.")

    # Фильтрация путей, пересекающих реки (исключая совпадающие начала/концы)
    try:
        rivers_layer = QgsProject.instance().mapLayersByName('rivers_and_points')[0]
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
                if start_geom.intersects(candidate.geometry()) or end_geom.intersects(candidate.geometry()):
                    continue
                paths_to_delete.append(feature.id())
                break
    if paths_to_delete:
        paths_layer.startEditing()
        for fid in paths_to_delete:
            paths_layer.deleteFeature(fid)
        paths_layer.commitChanges()
        QMessageBox.information(None, "Информация", f"Удалено {len(paths_to_delete)} путей, пересекающих реки.")