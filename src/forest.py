import os
import math
import numpy as np
import processing
from osgeo import gdal
from pyproj import Transformer
from qgis.core import (
    QgsProject,
    QgsCoordinateReferenceSystem,
    QgsPointXY,
    QgsRasterLayer,
    QgsVectorLayer,
    QgsGeometry,
    QgsField,
    QgsFields,
    QgsFeature,
    QgsSymbol,
    QgsRendererCategory,
    QgsCategorizedSymbolRenderer,
    QgsProcessingFeatureSourceDefinition,
)
from qgis.gui import QgsMapToolEmitPoint
from qgis.PyQt.QtCore import QVariant, QEventLoop, pyqtSignal
from qgis.PyQt.QtGui import QColor
from PyQt5.QtWidgets import QPushButton
from qgis.utils import iface
from .common import *


def start_point_collection():
    """
    Инициализирует сбор точек пользователем.
    """
    collector = PointCollector(iface.mapCanvas())
    iface.mapCanvas().setMapTool(collector)
    print("Щелкните по карте, чтобы выбрать точки.", flush=True)
    return collector


def create_polygon_from_points(points):
    """
    Преобразует список точек в замкнутый полигон.
    """
    # Преобразуем список точек в полигон (замкнутую геометрию)
    qgs_points = [
        QgsPointXY(pt.x(), pt.y()) for pt in points
    ]  # Преобразуем в QgsPointXY
    polygon = QgsGeometry.fromPolygonXY(
        [qgs_points]
    )  # Обратите внимание на вложение в список
    return polygon


def add_polygon_to_layer(polygon):
    """
    Создает в памяти векторный слой с полигоном и добавляет его в проект.
    """
    polygon_layer = QgsVectorLayer("Polygon?crs=EPSG:3857", "Selected Region", "memory")
    polygon_layer_data = polygon_layer.dataProvider()

    polygon_feature = QgsFeature()
    polygon_feature.setGeometry(polygon)
    polygon_layer_data.addFeatures([polygon_feature])
    QgsProject.instance().addMapLayer(polygon_layer)
    print("Полигон добавлен в проект.", flush=True)
    return polygon_layer


def clip_dem_with_polygon(
    dem_layer, polygon_layer, masked_dem_output_path, project_folder
):
    """
    Создает буфер вокруг полигона и обрезает DEM по этому буферу.
    """
    # Параметры буфера
    buffer_distance = 100  # Задайте отступ в метрах или единицах CRS маски
    buffered_mask = os.path.join(project_folder, "buffered_mask.shp")

    # Создание буфера
    processing.run(
        "native:buffer",
        {
            "INPUT": polygon_layer,  # Исходная маска
            "DISTANCE": buffer_distance,  # Отступ
            "SEGMENTS": 5,  # Количество сегментов для сглаживания (можно увеличить для кругов)
            "END_CAP_STYLE": 0,  # Открытая граница
            "JOIN_STYLE": 0,  # Стыковка углов (например, круглая)
            "MITER_LIMIT": 2,
            "DISSOLVE": False,
            "OUTPUT": buffered_mask,
        },
    )
    print(f"Буфер сохранен в: {buffered_mask}", flush=True)

    # Используем инструмент GDAL для обрезки DEM с использованием полигона
    processing.run(
        "gdal:cliprasterbymasklayer",
        {
            "INPUT": dem_layer,
            "MASK": QgsProcessingFeatureSourceDefinition(
                buffered_mask, selectedFeaturesOnly=False
            ),
            "CROP_TO_CUTLINE": True,  # Оставляем все пиксели, пересекающие границу полигона
            "ALL_TOUCHED": True,  # Включаем все пиксели, хотя бы частично затронутые полигоном
            "KEEP_RESOLUTION": True,  # Preserve the resolution of the DEM
            "OUTPUT": masked_dem_output_path,
        },
    )
    print(f"Обрезка завершена. Выходной файл: {masked_dem_output_path}", flush=True)


def handle_points_collection(
    collector, dem_layer, masked_dem_output_path, project_folder
):
    """
    Обработчик завершения сбора точек: формирует полигон, добавляет его в проект и запускает обрезку DEM.
    """
    selected_points = collector.get_points()
    if len(selected_points) < 3:
        print("Для создания полигона необходимо выбрать хотя бы 3 точки.", flush=True)
        return

    polygon = create_polygon_from_points(selected_points)
    polygon_layer = add_polygon_to_layer(polygon)
    clip_dem_with_polygon(
        dem_layer, polygon_layer, masked_dem_output_path, project_folder
    )


def add_finish_button(collector, dem_layer, masked_dem_output_path, project_folder):
    """
    Добавляет кнопку завершения сбора точек, при нажатии которой завершается сбор и обрабатываются выбранные точки.
    """
    finish_button = QPushButton("Завершить сбор точек", iface.mainWindow())
    finish_button.setGeometry(10, 50, 200, 30)  # Позиция и размер кнопки
    finish_button.show()

    def on_button_click():
        collector.complete_collection()
        finish_button.deleteLater()  # Убираем кнопку после завершения
        handle_points_collection(
            collector, dem_layer, masked_dem_output_path, project_folder
        )
        process_complete()  # Сигнализируем об окончании процесса

    finish_button.clicked.connect(on_button_click)


def wait_for_process():
    """
    Запускает цикл ожидания завершения обработки с помощью QEventLoop.
    Функция завершится при вызове глобальной функции process_complete.
    """
    loop = QEventLoop()

    def stop_waiting():
        loop.quit()

    global process_complete
    process_complete = stop_waiting
    loop.exec_()


def process_dem_with_polygon(
    main_dem_file_path, masked_dem_output_path, project_folder
):
    """
    Основной рабочий поток обработки DEM: загружает DEM, инициирует сбор точек пользователем,
    добавляет кнопку завершения и ожидает завершения обработки.
    """
    dem_layer = QgsRasterLayer(main_dem_file_path, "SRTM DEM Layer")
    if not dem_layer.isValid():
        print("Ошибка загрузки DEM слоя.", flush=True)
        return

    QgsProject.instance().addMapLayer(dem_layer)
    print("DEM слой загружен.", flush=True)

    collector = start_point_collection()
    add_finish_button(collector, dem_layer, masked_dem_output_path, project_folder)
    wait_for_process()


def reproject_dem2(project_folder):
    # Репроекция DEM из EPSG:4326 в EPSG:3857
    output_reprojected_path = os.path.join(project_folder, "reprojected_dem.tif")
    output_path = os.path.join(project_folder, "masked_dem.tif")
    reprojected_dem = processing.run(
        "gdal:warpreproject",
        {
            "INPUT": output_path,
            "TARGET_CRS": QgsCoordinateReferenceSystem("EPSG:3857"),
            "RESAMPLING": 0,
            "NODATA": -9999,
            "TARGET_RESOLUTION": 30,
            "OPTIONS": "",
            "DATA_TYPE": 0,
            "OUTPUT": output_reprojected_path,
        },
    )["OUTPUT"]
    print(
        f"Репроекция выполнена. Файл сохранен в {output_reprojected_path}", flush=True
    )

    return reprojected_dem


def load_dem_to_numpy(project_folder):
    """
    Загружает DEM из файла в numpy-массив.
    """
    input_path = os.path.join(project_folder, "masked_dem.tif")
    dem_raster = gdal.Open(input_path)
    dem_band = dem_raster.GetRasterBand(1)
    dem_data = dem_band.ReadAsArray()
    return dem_data, dem_raster


def setting_dem_coordinates(dem_data, dem_raster):
    """
    Находит максимальные и минимальные высоты, а также вычисляет географические координаты этих точек.
    """
    max_height = np.max(dem_data)
    min_height = np.min(dem_data)

    # Координаты точек с максимальной и минимальной высотой
    max_coords = np.unravel_index(np.argmax(dem_data), dem_data.shape)
    min_coords = np.unravel_index(np.argmin(dem_data), dem_data.shape)

    # Преобразуем индексы пикселей в географические координаты (EPSG:4326)
    transform = dem_raster.GetGeoTransform()
    max_x_4326 = (
        transform[0] + max_coords[1] * transform[1] + max_coords[0] * transform[2]
    )
    max_y_4326 = (
        transform[3] + max_coords[0] * transform[4] + max_coords[1] * transform[5]
    )

    min_x_4326 = (
        transform[0] + min_coords[1] * transform[1] + min_coords[0] * transform[2]
    )
    min_y_4326 = (
        transform[3] + min_coords[0] * transform[4] + min_coords[1] * transform[5]
    )

    transformer_to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    # Преобразование координат точек с максимальной и минимальной высотой в EPSG:3857
    max_x_3857, max_y_3857 = transformer_to_3857.transform(max_x_4326, max_y_4326)
    min_x_3857, min_y_3857 = transformer_to_3857.transform(min_x_4326, min_y_4326)
    coordinates = [
        (max_x_3857, max_y_3857),
        (min_x_3857, min_y_3857),
    ]
    return coordinates, min_height, max_height


def create_temp_vector_layer():
    """
    Создает временный векторный слой для точек и добавляет его в проект.
    """
    layer = QgsVectorLayer("Point?crs=EPSG:3857", "points1", "memory")
    QgsProject.instance().addMapLayer(layer)
    return layer


def set_attribute_fields(layer):
    """
    Добавляет в слой поле 'ID' для хранения идентификатора точки.
    """
    fields = QgsFields()
    fields.append(QgsField("ID", QVariant.Int))

    layer.dataProvider().addAttributes(fields)
    layer.updateFields()
    return layer


def add_points(coordinates, layer):
    """
    Добавляет точки с заданными координатами в слой.
    """
    features = []
    for i, (x, y) in enumerate(coordinates, start=1):
        feat = QgsFeature()
        feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(x, y)))
        feat.setAttributes([i])
        features.append(feat)
    layer.dataProvider().addFeatures(features)
    layer.updateExtents()
    print("Точки успешно добавлены!", flush=True)


def create_points(project_folder):
    """
    Загружает DEM, выполняет репроекцию, вычисляет координаты точек с экстремальными значениями и
    создает временный векторный слой с этими точками.
    """
    dem_path = os.path.join(project_folder, "masked_dem.tif")
    add_dem_layer(dem_path)
    reprojected_dem_mask = reproject_dem2(project_folder)
    dem_data, dem_raster = load_dem_to_numpy(project_folder)
    coordinates, min_height, max_height = setting_dem_coordinates(dem_data, dem_raster)
    layer = create_temp_vector_layer()
    layer = set_attribute_fields(layer)
    add_points(coordinates, layer)
    return reprojected_dem_mask, min_height, max_height


def calculate(H, J, angle):
    """
    Вычисляет идеальное расстояние и шаг по высоте.
    """
    L = H * J
    hop = L * math.tan(math.radians(angle))
    print(f"Идеальное расстояние L: {L}", flush=True)
    print(f"Шаг по высоте hop: {hop}", flush=True)
    return L, hop


def construct_isolines(reprojected_dem_mask, hop, max_height, project_folder):
    """
    Создает изолинии DEM с заданным шагом.
    """
    contours_output_path = os.path.join(project_folder, "contours_output.shp")
    try:
        result = processing.run(
            "gdal:contour",
            {
                "INPUT": reprojected_dem_mask,
                "BAND": 1,
                "INTERVAL": hop,
                "FIELD_NAME": "ELEV",
                "BASE": max_height,
                "OUTPUT": "TEMPORARY_OUTPUT",
            },
        )
        print(f"Изолинии успешно созданы: {result}", flush=True)
    except Exception as e:
        print(f"Ошибка при создании изолиний: {e}", flush=True)
        raise e

    return contours_output_path, result


def add_isolines_to_a_layer(contours_output_path, result):
    """
    Загружает слой изолиний, проверяет корректность и добавляет его в проект.
    """
    contours_output_path = result["OUTPUT"]
    print(f"Временный файл изолиний создан: {contours_output_path}", flush=True)

    if os.path.getsize(contours_output_path) == 0:
        print("Ошибка: Файл изолиний пуст.", flush=True)

    # Загрузка изолиний в слой
    contours_layer = QgsVectorLayer(contours_output_path, "Contours", "ogr")

    print(f"CRS слоя: {contours_layer.crs().authid()}", flush=True)
    print(f"CRS проекта: {QgsProject.instance().crs().authid()}", flush=True)

    if not contours_layer.isValid():
        raise Exception("Не удалось загрузить слой изолиний.")
    QgsProject.instance().addMapLayer(contours_layer)
    return contours_layer


def filter_isoline(contours_layer):
    """
    Создает временный слой для фильтрации изолиний по высотному диапазону.
    """
    # Фильтрация изолиний по диапазону высот
    filtered_layer = QgsVectorLayer(
        "LineString?crs=EPSG:3857", "Filtered Contours", "memory"
    )  # Используем ту же CRS, что и DEM
    filtered_provider = filtered_layer.dataProvider()
    filtered_provider.addAttributes(
        contours_layer.fields()
    )  # Копируем атрибуты из изолиний
    filtered_layer.updateFields()
    return filtered_provider, filtered_layer


def adding_isolines_by_height(
    contours_layer, min_height, max_height, filtered_provider, filtered_layer
):
    """
    Добавляет в новый слой только те изолинии, высота которых попадает в заданный диапазон.
    """
    for feature in contours_layer.getFeatures():
        elevation = feature["ELEV"]
        if min_height <= elevation <= max_height:  # Диапазон высот
            filtered_provider.addFeatures(
                [feature]
            )  # Добавляем отфильтрованные изолинии

    filtered_layer.updateExtents()  # Обновляем границы слоя
    QgsProject.instance().addMapLayer(filtered_layer)


def create_isolines(reprojected_dem_mask, min_height, max_height, project_folder):
    """
    Объединяет шаг вычисления, создание и фильтрацию изолиний.
    """
    H = 15  # Высота лесополосы
    J = 20  # Коэффициент влияния
    angle = 3  # Угол наклона в градусах

    _, hop = calculate(H, J, angle)
    contours_output_path, result = construct_isolines(
        reprojected_dem_mask, hop, max_height, project_folder
    )
    contours_layer = add_isolines_to_a_layer(contours_output_path, result)
    filtered_provider, filtered_layer = filter_isoline(contours_layer)
    adding_isolines_by_height(
        contours_layer, min_height, max_height, filtered_provider, filtered_layer
    )
    return filtered_layer


def add_forests_layer():
    """
    Создает временный слой для лесополос и добавляет поле 'Step'.
    """
    forest_layer = QgsVectorLayer(
        "LineString?crs=EPSG:3857", "Forest Belts", "memory"
    )  # Совместимый CRS
    forest_provider = forest_layer.dataProvider()
    forest_provider.addAttributes(
        [QgsField("Step", QVariant.Int)]
    )  # Добавляем поле Step
    forest_layer.updateFields()
    return forest_layer, forest_provider


def generate_shades(base_color, steps):
    """
    Генерирует список оттенков цвета на основе базового цвета.
    """
    shades = []
    for i in range(steps):
        factor = i / (steps - 1)  # Нормализуем в диапазон [0, 1]
        # Увеличиваем интенсивность цвета для яркости
        r = int(
            base_color.red() * (1 - factor) + 255 * factor
        )  # Плавное увеличение от base к яркому
        g = int(base_color.green() * (1 - factor) + 255 * factor)
        b = int(base_color.blue() * (1 - factor) + 255 * factor)
        shades.append(QColor(r, g, b))
    return shades


def generate_color_pallete():
    """
    Генерирует палитру цветов на основе нескольких базовых цветов.
    """
    # Базовый цвет (чистый красный)
    base_color = QColor(255, 0, 0)
    # Базовый цвет (чистый зеленый)
    base_color1 = QColor(0, 255, 0)
    # Базовый цвет (чистый синий)
    base_color2 = QColor(0, 0, 255)
    grad_steps = 255
    # Генерируем оттенки
    colors = (
        generate_shades(base_color2, grad_steps)
        + generate_shades(base_color1, grad_steps)
        + generate_shades(base_color, grad_steps)
    )
    return colors


def add_forest_feature(filtered_layer, forest_provider, forest_layer, colors):
    """
    Добавляет лесополосы на основе геометрии отфильтрованных изолиний.
    """
    step_index = 1
    categories = []  # Список категорий для рендера
    # Добавление лесополос
    for feature in filtered_layer.getFeatures():
        geometry = feature.geometry()
        if not geometry.isEmpty():
            forest_feature = QgsFeature()
            forest_feature.setGeometry(geometry)
            forest_feature.setAttributes([step_index])  # Присваиваем шаг
            forest_provider.addFeatures([forest_feature])
            # Создаем символ с градиентным цветом
            color = colors[step_index % len(colors)]
            symbol = QgsSymbol.defaultSymbol(forest_layer.geometryType())
            symbol.setColor(color)
            # Добавляем категорию для рендера
            category = QgsRendererCategory(
                step_index, symbol, f"Лесополоса {step_index}"
            )
            categories.append(category)
            step_index += 1
    return categories


def config_render(forest_layer, categories):
    """
    Настраивает категориальный рендерер для слоя лесополос и добавляет слой в проект.
    """
    renderer = QgsCategorizedSymbolRenderer("Step", categories)
    forest_layer.setRenderer(renderer)
    forest_layer.updateExtents()  # Обновляем границы слоя
    QgsProject.instance().addMapLayer(forest_layer)


def check_forest_layers():
    """
    Проверяет наличие слоев "Forest Belts" и "Selected Region" в проекте.
    """
    forest_layers = QgsProject.instance().mapLayersByName("Forest Belts")
    region_layers = QgsProject.instance().mapLayersByName("Selected Region")
    if not forest_layers or not region_layers:
        print("Не найдены слои 'Forest Belts' или 'Selected Region'")
        exit()
    return forest_layers[0], region_layers[0]


def create_boundary_layer(selected_region_layer):
    """
    Создает слой границ по выбранному региону.
    """
    boundary_result = processing.run(
        "native:boundary", {"INPUT": selected_region_layer, "OUTPUT": "memory:"}
    )
    return boundary_result["OUTPUT"]


def setting_up_render(forest_layer):
    """
    Готовит слой лесополос для рендера: проверяет наличие поля 'Step' и получает уникальные шаги.
    """
    if "Step" not in [field.name() for field in forest_layer.fields()]:
        forest_layer.dataProvider().addAttributes([QgsField("Step", QVariant.Int)])
        forest_layer.updateFields()
    categories = []
    step_index = 0
    unique_steps = sorted(set(f["Step"] for f in forest_layer.getFeatures()))
    return categories, step_index, unique_steps


def process_step(
    unique_steps, forest_layer, boundary_layer, colors, categories, step_index
):
    """
    Обрабатывает каждый уникальный шаг, соединяя геометрию лесополос с ближайшей границей и обновляя рендер.
    """
    for step in unique_steps:
        print(f"Обработка шага {step}", flush=True)
        step_features = [f for f in forest_layer.getFeatures() if f["Step"] == step]
        for feature in step_features:
            geometry = feature.geometry()
            if geometry.isEmpty():
                print(f"Пропуск пустой геометрии на шаге {step}", flush=True)
                continue

            # Поиск ближайшего контура
            shortest_distance = float("inf")
            best_contour = None
            for boundary_feature in boundary_layer.getFeatures():
                boundary_geom = boundary_feature.geometry()
                distance = geometry.distance(boundary_geom)
                if distance < shortest_distance:
                    shortest_distance = distance
                    best_contour = boundary_geom

            if best_contour:
                # Создание новой замкнутой геометрии
                closed_geometry = geometry.combine(best_contour)
                new_feature = QgsFeature(forest_layer.fields())
                new_feature.setGeometry(closed_geometry)
                new_feature.setAttributes(feature.attributes())
                forest_layer.dataProvider().addFeatures([new_feature])
                # Создание символа для рендеринга
                color = colors[step_index % len(colors)]
                symbol = QgsSymbol.defaultSymbol(forest_layer.geometryType())
                symbol.setColor(color)
                category = QgsRendererCategory(step, symbol, f"Лесополоса {step}")
                categories.append(category)
            step_index += 1
    return forest_layer


def render_forest_belts(categories, forest_layer):
    """
    Применяет рендеринг к слою лесополос и добавляет его в проект.
    """
    renderer = QgsCategorizedSymbolRenderer("Step", categories)
    forest_layer.setRenderer(renderer)
    forest_layer.updateExtents()  # Обновление слоя
    QgsProject.instance().addMapLayer(forest_layer)


def generate_forest_belts_layer(
    filtered_layer,
):
    """
    Генерирует слой лесополос из отфильтрованных изолиний и настраивает рендеринг.
    """
    forest_layer, forest_provider = add_forests_layer()
    colors = generate_color_pallete()
    categories = add_forest_feature(
        filtered_layer, forest_provider, forest_layer, colors
    )
    config_render(forest_layer, categories)
    forest_layer, selected_region_layer = check_forest_layers()
    boundary_layer = create_boundary_layer(selected_region_layer)
    categories, step_index, unique_steps = setting_up_render(forest_layer)
    base_color = QColor(0, 255, 0)
    colors = generate_shades(base_color, 255)
    forest_layer = process_step(
        unique_steps, forest_layer, boundary_layer, colors, categories, step_index
    )
    render_forest_belts(categories, forest_layer)


def create_forest_belts(project_folder):
    reprojected_dem_mask, min_height, max_height = create_points(project_folder)
    filtered_layer = create_isolines(
        reprojected_dem_mask, min_height, max_height, project_folder
    )
    generate_forest_belts_layer(filtered_layer)
    print("Лесополосы успешно замкнуты и добавлены в проект.", flush=True)


def forest(project_folder):
    _, dem_path = get_main_def(project_folder)
    add_dem_layer(dem_path)

    main_dem_file_path = os.path.join(
        project_folder, "srtm_output.tif"
    )  # Путь к вашему DEM файлу
    masked_dem_output_path = os.path.join(
        project_folder, "masked_dem.tif"
    )  # Путь к выходному файлу после обрезки

    process_dem_with_polygon(main_dem_file_path, masked_dem_output_path, project_folder)
    create_forest_belts(project_folder)


class PointCollector(QgsMapToolEmitPoint):
    """
    Инструмент для сбора точек на карте. При щелчке точки добавляются в список и отображаются на карте.
    """

    collection_complete = pyqtSignal()  # Сигнал завершения сбора точек

    def __init__(self, canvas):
        super().__init__(canvas)
        self.canvas = canvas
        self.points = []
        # Создаем временный слой для отображения точек
        self.point_layer = QgsVectorLayer(
            "Point?crs=EPSG:3857", "Selected Points", "memory"
        )
        self.point_provider = self.point_layer.dataProvider()
        QgsProject.instance().addMapLayer(self.point_layer)
        # Список аннотаций для отображения координат
        self.annotations = []

    def canvasPressEvent(self, event):
        point = self.toMapCoordinates(event.pos())
        self.points.append(point)
        print(f"Точка выбрана: {point}", flush=True)
        # Добавляем точку в слой
        point_feature = QgsFeature()
        point_feature.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(point)))
        self.point_provider.addFeatures([point_feature])
        self.point_layer.updateExtents()
        self.point_layer.triggerRepaint()

    def get_points(self):
        return self.points

    def complete_collection(self):
        # Генерируем сигнал, чтобы уведомить об окончании сбора
        self.collection_complete.emit()
