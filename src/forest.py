import math
from pathlib import Path
from typing import Optional

import numpy as np
import processing
from osgeo import gdal
from pyproj import Transformer
from PyQt5.QtWidgets import QPushButton
from qgis.core import (
    QgsCategorizedSymbolRenderer,
    QgsCoordinateReferenceSystem,
    QgsFeature,
    QgsField,
    QgsFields,
    QgsGeometry,
    QgsPointXY,
    QgsProcessingFeatureSourceDefinition,
    QgsProject,
    QgsRendererCategory,
    QgsSymbol,
    QgsVectorLayer,
)
from qgis.gui import QgsMapToolEmitPoint
from qgis.PyQt.QtCore import QEventLoop, QVariant, pyqtSignal
from qgis.PyQt.QtGui import QColor
from qgis.utils import iface

from .common import add_dem_layer, get_main_def
from .progress_manager import ProgressManager


class PointCollector(QgsMapToolEmitPoint):
    """Инструмент для сбора точек на карте."""

    collection_complete = pyqtSignal()

    def __init__(self, canvas) -> None:
        super().__init__(canvas)
        self.canvas = canvas
        self.points = []
        self.point_layer = QgsVectorLayer(
            "Point?crs=EPSG:3857", "Selected Points", "memory"
        )
        self.point_provider = self.point_layer.dataProvider()
        QgsProject.instance().addMapLayer(self.point_layer)

    def canvasPressEvent(self, event) -> None:
        point = self.toMapCoordinates(event.pos())
        self.points.append(point)
        print(f"Точка выбрана: {point}", flush=True)
        point_feature = QgsFeature()
        point_feature.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(point)))
        self.point_provider.addFeatures([point_feature])
        self.point_layer.updateExtents()
        self.point_layer.triggerRepaint()

    def get_points(self):
        return self.points

    def complete_collection(self) -> None:
        self.collection_complete.emit()


def create_polygon_from_points(points):
    """Преобразует список точек в замкнутый полигон."""
    qgs_points = [QgsPointXY(pt.x(), pt.y()) for pt in points]
    return QgsGeometry.fromPolygonXY([qgs_points])


def add_polygon_to_layer(polygon):
    """Создает в памяти векторный слой с полигоном."""
    polygon_layer = QgsVectorLayer("Polygon?crs=EPSG:3857", "Selected Region", "memory")
    polygon_layer_data = polygon_layer.dataProvider()

    polygon_feature = QgsFeature()
    polygon_feature.setGeometry(polygon)
    polygon_layer_data.addFeatures([polygon_feature])
    QgsProject.instance().addMapLayer(polygon_layer)
    print("Полигон добавлен в проект.", flush=True)
    return polygon_layer


def clip_dem_with_polygon(
    dem_layer,
    polygon_layer,
    masked_dem_output_path: Path,
    project_folder: Path,
    progress,
) -> Optional[Path]:
    """Создает буфер и обрезает DEM."""
    if not progress.update(10, "Создание буфера..."):
        return None

    buffer_distance = 100
    buffered_mask = Path(project_folder) / "buffered_mask.shp"

    processing.run(
        "native:buffer",
        {
            "INPUT": polygon_layer,
            "DISTANCE": buffer_distance,
            "SEGMENTS": 5,
            "END_CAP_STYLE": 0,
            "JOIN_STYLE": 0,
            "MITER_LIMIT": 2,
            "DISSOLVE": False,
            "OUTPUT": str(buffered_mask),
        },
    )

    if not progress.update(20, "Обрезка DEM..."):
        return None

    processing.run(
        "gdal:cliprasterbymasklayer",
        {
            "INPUT": dem_layer,
            "MASK": QgsProcessingFeatureSourceDefinition(
                str(buffered_mask), selectedFeaturesOnly=False
            ),
            "CROP_TO_CUTLINE": True,
            "ALL_TOUCHED": True,
            "KEEP_RESOLUTION": True,
            "OUTPUT": str(masked_dem_output_path),
        },
    )
    return masked_dem_output_path


def reproject_dem2(project_folder: Path, progress):
    """Репроекция DEM в EPSG:3857."""
    if not progress.update(10, "Репроекция DEM..."):
        return None

    output_reprojected_path = Path(project_folder) / "reprojected_dem.tif"
    output_path = Path(project_folder) / "masked_dem.tif"

    return processing.run(
        "gdal:warpreproject",
        {
            "INPUT": str(output_path),
            "TARGET_CRS": QgsCoordinateReferenceSystem("EPSG:3857"),
            "RESAMPLING": 0,
            "NODATA": -9999,
            "TARGET_RESOLUTION": 30,
            "OPTIONS": "",
            "DATA_TYPE": 0,
            "OUTPUT": str(output_reprojected_path),
        },
    )["OUTPUT"]


def load_dem_to_numpy(project_folder: Path, progress):
    """Загружает DEM в numpy массив."""
    if not progress.update(5, "Загрузка DEM..."):
        return None, None

    input_path = Path(project_folder) / "masked_dem.tif"
    dem_raster = gdal.Open(str(input_path))
    dem_band = dem_raster.GetRasterBand(1)
    dem_data = dem_band.ReadAsArray()
    return dem_data, dem_raster


def setting_dem_coordinates(dem_data, dem_raster, progress):
    """Находит точки с экстремальными высотами."""
    if not progress.update(10, "Анализ высот..."):
        return None, None, None

    max_height = np.max(dem_data)
    min_height = np.min(dem_data)

    max_coords = np.unravel_index(np.argmax(dem_data), dem_data.shape)
    min_coords = np.unravel_index(np.argmin(dem_data), dem_data.shape)

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
    max_x_3857, max_y_3857 = transformer_to_3857.transform(max_x_4326, max_y_4326)
    min_x_3857, min_y_3857 = transformer_to_3857.transform(min_x_4326, min_y_4326)

    coordinates = [(max_x_3857, max_y_3857), (min_x_3857, min_y_3857)]
    return coordinates, min_height, max_height


def create_temp_vector_layer(progress):
    """Создает временный слой для точек."""
    if not progress.update(5, "Создание слоя..."):
        return None

    layer = QgsVectorLayer("Point?crs=EPSG:3857", "points1", "memory")
    QgsProject.instance().addMapLayer(layer)
    return layer


def set_attribute_fields(layer, progress):
    """Добавляет атрибуты к слою."""
    if not progress.update(5, "Настройка атрибутов..."):
        return None

    fields = QgsFields()
    fields.append(QgsField("ID", QVariant.Int))

    layer.dataProvider().addAttributes(fields)
    layer.updateFields()
    return layer


def add_points(coordinates, layer, progress) -> None:
    """Добавляет точки в слой."""
    if not progress.update(10, "Добавление точек..."):
        return

    features = []
    for i, (x, y) in enumerate(coordinates, start=1):
        feat = QgsFeature()
        feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(x, y)))
        feat.setAttributes([i])
        features.append(feat)

    layer.dataProvider().addFeatures(features)
    layer.updateExtents()


def calculate(h, j, angle):
    """Вычисляет параметры для изолиний."""
    length = h * j
    hop = length * math.tan(math.radians(angle))
    return length, hop


def construct_isolines(
    reprojected_dem_mask, hop, max_height, project_folder: Path, progress
):
    """Создает изолинии."""
    if not progress.update(20, "Создание изолиний..."):
        return None, None

    contours_output_path = Path(project_folder) / "contours_output.shp"
    try:
        result = processing.run(
            "gdal:contour",
            {
                "INPUT": str(reprojected_dem_mask),
                "BAND": 1,
                "INTERVAL": hop,
                "FIELD_NAME": "ELEV",
                "BASE": max_height,
                "OUTPUT": "TEMPORARY_OUTPUT",
            },
        )
    except Exception as e:
        print(f"Ошибка при создании изолиний: {e}", flush=True)
        raise

    return contours_output_path, result


def add_isolines_to_a_layer(contours_output_path: Path, result, progress):
    """Добавляет изолинии в слой."""
    if not progress.update(10, "Добавление изолиний..."):
        return None

    contours_output_path = result["OUTPUT"]
    contours_layer = QgsVectorLayer(str(contours_output_path), "Contours", "ogr")

    if not contours_layer.isValid():
        msg = "Не удалось загрузить слой изолиний."
        raise Exception(msg)

    QgsProject.instance().addMapLayer(contours_layer)
    return contours_layer


def filter_isoline(contours_layer, progress):
    """Фильтрует изолинии по высоте."""
    if not progress.update(10, "Фильтрация изолиний..."):
        return None, None

    filtered_layer = QgsVectorLayer(
        "LineString?crs=EPSG:3857", "Filtered Contours", "memory"
    )
    filtered_provider = filtered_layer.dataProvider()
    filtered_provider.addAttributes(contours_layer.fields())
    filtered_layer.updateFields()
    return filtered_provider, filtered_layer


def adding_isolines_by_height(
    contours_layer, min_height, max_height, filtered_provider, filtered_layer, progress
) -> None:
    """Добавляет изолинии в заданном диапазоне высот."""
    total_features = contours_layer.featureCount()

    for i, feature in enumerate(contours_layer.getFeatures(), start=1):
        if progress.was_canceled():
            return

        if not progress.update(
            10 + int(30 * i / total_features),
            f"Фильтрация {i}/{total_features}",
        ):
            return

        elevation = feature["ELEV"]
        if min_height <= elevation <= max_height:
            filtered_provider.addFeatures([feature])

    filtered_layer.updateExtents()
    QgsProject.instance().addMapLayer(filtered_layer)


def generate_shades(base_color, steps):
    """Генерирует оттенки цвета."""
    shades = []
    for i in range(steps):
        factor = i / (steps - 1)
        r = int(base_color.red() * (1 - factor) + 255 * factor)
        g = int(base_color.green() * (1 - factor) + 255 * factor)
        b = int(base_color.blue() * (1 - factor) + 255 * factor)
        shades.append(QColor(r, g, b))
    return shades


def generate_color_pallete():
    """Создает палитру цветов."""
    base_color = QColor(255, 0, 0)
    base_color1 = QColor(0, 255, 0)
    base_color2 = QColor(0, 0, 255)
    grad_steps = 255
    return (
        generate_shades(base_color2, grad_steps)
        + generate_shades(base_color1, grad_steps)
        + generate_shades(base_color, grad_steps)
    )


def add_forests_layer(progress):
    """Создает слой лесополос."""
    if not progress.update(5, "Создание слоя лесополос..."):
        return None, None

    forest_layer = QgsVectorLayer("LineString?crs=EPSG:3857", "Forest Belts", "memory")
    forest_provider = forest_layer.dataProvider()
    forest_provider.addAttributes([QgsField("Step", QVariant.Int)])
    forest_layer.updateFields()
    return forest_layer, forest_provider


def add_forest_feature(filtered_layer, forest_provider, forest_layer, colors, progress):
    """Добавляет лесополосы в слой."""
    total_features = filtered_layer.featureCount()
    categories = []

    for _i, feature in enumerate(filtered_layer.getFeatures(), start=1):
        if progress.was_canceled():
            return None

        if not progress.update(
            10 + int(30 * _i / total_features),
            f"Добавление {_i}/{total_features}",
        ):
            return None

        geometry = feature.geometry()
        if not geometry.isEmpty():
            forest_feature = QgsFeature()
            forest_feature.setGeometry(geometry)
            forest_feature.setAttributes([_i])
            forest_provider.addFeatures([forest_feature])

            color = colors[_i % len(colors)]
            symbol = QgsSymbol.defaultSymbol(forest_layer.geometryType())
            symbol.setColor(color)
            category = QgsRendererCategory(_i, symbol, f"Лесополоса {_i}")
            categories.append(category)

    return categories


def config_render(forest_layer, categories, progress) -> None:
    """Настраивает отображение слоя."""
    if not progress.update(10, "Настройка отображения..."):
        return

    renderer = QgsCategorizedSymbolRenderer("Step", categories)
    forest_layer.setRenderer(renderer)
    forest_layer.updateExtents()
    QgsProject.instance().addMapLayer(forest_layer)


def forest(project_folder: Path) -> None:
    """Основная функция создания лесополос."""
    # Загрузка DEM без прогресса
    _, dem_path = get_main_def(project_folder)
    dem_layer = add_dem_layer(
        dem_path
    )  # Измените функцию add_dem_layer, чтобы она возвращала слой
    if not dem_layer.isValid():
        print("Ошибка загрузки DEM слоя.", flush=True)
        return

    masked_dem_output_path = Path(project_folder) / "masked_dem.tif"

    # Сбор точек без прогресса
    canvas = iface.mapCanvas()
    collector = PointCollector(canvas)
    canvas.setMapTool(collector)

    # Ожидание завершения сбора точек
    loop = QEventLoop()
    collector.collection_complete.connect(loop.quit)

    finish_button = QPushButton("Завершить сбор точек", iface.mainWindow())
    finish_button.setGeometry(10, 50, 200, 30)
    finish_button.show()
    finish_button.clicked.connect(collector.complete_collection)

    loop.exec_()
    finish_button.deleteLater()

    # После сбора точек запускаем прогресс
    progress = ProgressManager(title="Создание лесополос", label="Начало обработки...")
    progress.init_progress(100)

    try:
        # Обработка собранных точек
        selected_points = collector.get_points()
        if len(selected_points) < 3:
            print(
                "Для создания полигона необходимо выбрать хотя бы 3 точки.", flush=True
            )
            return

        if not progress.update(10, "Создание полигона..."):
            return

        polygon = create_polygon_from_points(selected_points)
        polygon_layer = add_polygon_to_layer(polygon)

        if not progress.update(20, "Обрезка DEM..."):
            return

        masked_dem = clip_dem_with_polygon(
            dem_layer, polygon_layer, masked_dem_output_path, project_folder, progress
        )
        if not masked_dem:
            return

        if not progress.update(30, "Создание точек высот..."):
            return

        reprojected_dem_mask = reproject_dem2(project_folder, progress)
        if not reprojected_dem_mask:
            return

        dem_data, dem_raster = load_dem_to_numpy(project_folder, progress)
        if dem_data is None:
            return

        coordinates, min_height, max_height = setting_dem_coordinates(
            dem_data, dem_raster, progress
        )
        if coordinates is None:
            return

        points_layer = create_temp_vector_layer(progress)
        if points_layer is None:
            return

        points_layer = set_attribute_fields(points_layer, progress)
        if points_layer is None:
            return

        add_points(coordinates, points_layer, progress)

        if not progress.update(50, "Создание изолиний..."):
            return

        h, j, angle = 15, 20, 3
        _, hop = calculate(h, j, angle)
        contours_output_path, result = construct_isolines(
            reprojected_dem_mask,
            hop,
            max_height,
            project_folder,
            progress,
        )
        if not contours_output_path:
            return

        contours_layer = add_isolines_to_a_layer(contours_output_path, result, progress)
        if contours_layer is None:
            return

        filtered_provider, filtered_layer = filter_isoline(contours_layer, progress)
        if filtered_provider is None:
            return

        adding_isolines_by_height(
            contours_layer,
            min_height,
            max_height,
            filtered_provider,
            filtered_layer,
            progress,
        )

        if not progress.update(70, "Генерация лесополос..."):
            return

        forest_layer, forest_provider = add_forests_layer(progress)
        if forest_layer is None:
            return

        colors = generate_color_pallete()
        categories = add_forest_feature(
            filtered_layer, forest_provider, forest_layer, colors, progress
        )
        if not categories:
            return

        config_render(forest_layer, categories, progress)

        progress.update(100, "Завершено!")
        print("Лесополосы успешно созданы.", flush=True)
    finally:
        progress.finish()
