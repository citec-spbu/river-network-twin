import os
from typing import Any
import requests
import processing
from qgis.core import (
    QgsProject,
    QgsCoordinateReferenceSystem,
    QgsRasterLayer,
    QgsApplication,
)
from qgis.analysis import QgsNativeAlgorithms
from qgis.PyQt.QtWidgets import QInputDialog, QMessageBox
from pyproj import Transformer


def set_project_crs():
    """
    Устанавливает систему координат проекта на EPSG:3857 (Pseudo-Mercator).
    """
    crs = QgsCoordinateReferenceSystem("EPSG:3857")
    QgsProject.instance().setCrs(crs)


def enable_processing_algorithms():
    """
    Включает встроенные алгоритмы обработки QGIS.
    """
    QgsApplication.processingRegistry().addProvider(QgsNativeAlgorithms())


def add_opentopo_layer():
    """
    Добавляет слой OpenTopoMap в проект QGIS.
    """
    opentopo_url = (
        "type=xyz&zmin=0&zmax=21&url=https://tile.opentopomap.org/{z}/{x}/{y}.png"
    )
    opentopo_layer = QgsRasterLayer(opentopo_url, "OpenTopoMap", "wms")
    QgsProject.instance().addMapLayer(opentopo_layer)


def get_coordinates():
    """
    Координаты озера в системе координат EPSG:3857
    x_3857, y_3857 = 4316873, 7711643
    """
    x, ok_x = QInputDialog.getDouble(
        None, "Координата X", "Введите координату X:", value=4316873, decimals=6
    )
    if not ok_x:
        QMessageBox.warning(
            None, "Ошибка", "Неправильная координата X. Работа плагина прекращена."
        )
        return None, None

    y, ok_y = QInputDialog.getDouble(
        None, "Координата Y", "Введите координату Y:", value=7711643, decimals=6
    )
    if not ok_y:
        QMessageBox.warning(
            None, "Ошибка", "Неправильная координата Y. Работа плагина прекращена."
        )
        return None, None

    return x, y


def transform_coordinates(x, y):
    """
    Преобразовать координаты EPSG:3857 в широту и долготу (EPSG:4326)
    """
    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    return transformer.transform(x, y)


def download_dem(bbox, project_folder):
    """
    Загружает DEM с OpenTopography API по заданному bounding box.
    """
    api_key = "c1fcbd0b2f691c736e3bf8c43e52a54d"
    url = (
        f"https://portal.opentopography.org/API/globaldem?"
        f"demtype=SRTMGL1"
        f"&south={bbox[1]}&north={bbox[3]}"
        f"&west={bbox[0]}&east={bbox[2]}"
        f"&outputFormat=GTiff"
        f"&API_Key={api_key}"
    )
    response = requests.get(url)
    if response.status_code != 200:
        QMessageBox.critical(
            None,
            "Ошибка загрузки DEM",
        )
        raise Exception("Ошибка загрузки DEM")

    output_path = os.path.join(project_folder, "srtm_output.tif")
    with open(output_path, "wb") as f:
        f.write(response.content)

    return output_path


def reproject_dem(dem_path):
    """
    Репроецирует DEM из EPSG:4326 в EPSG:3857.
    """
    return processing.run(
        "gdal:warpreproject",
        {
            "INPUT": dem_path,
            "SOURCE_CRS": QgsCoordinateReferenceSystem("EPSG:4326"),
            "TARGET_CRS": QgsCoordinateReferenceSystem("EPSG:3857"),
            "RESAMPLING": 0,
            "NODATA": -9999,
            "TARGET_RESOLUTION": 30,
            "OPTIONS": "",
            "DATA_TYPE": 0,
            "TARGET_EXTENT": None,
            "TARGET_EXTENT_CRS": None,
            "MULTITHREADING": False,
            "EXTRA": "",
            "OUTPUT": "TEMPORARY_OUTPUT",
        },
    )["OUTPUT"]


def add_dem_layer(dem_path):
    """
    Добавляет загруженный DEM слой в проект QGIS.
    """
    dem_layer = QgsRasterLayer(dem_path, "SRTM DEM Layer")
    QgsProject.instance().addMapLayer(dem_layer)
    return dem_layer


def get_main_def(project_folder) -> Any:
    set_project_crs()
    enable_processing_algorithms()
    add_opentopo_layer()

    x, y = get_coordinates()
    if x is None or y is None:
        return

    longitude, latitude = transform_coordinates(x, y)
    bbox = [longitude - 0.5, latitude - 0.5, longitude + 0.5, latitude + 0.5]

    dem_path = download_dem(bbox, project_folder)
    return reproject_dem(dem_path), dem_path
