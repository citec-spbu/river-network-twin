import processing
from qgis.core import QgsRasterLayer


def build_basins_layer(reprojected_relief, basins_path: str) -> QgsRasterLayer:
    """
    Заполняет DEM, считает потоковые направления и строит водосборы
    через GRASS r.watershed (аналог SAGA FillSinks + Watershed).
    """
    processing.run(
        "grass7:r.watershed",
        {
            "elevation": reprojected_relief,
            "threshold": 1000,
            "basin": basins_path,
            "memory": 300,  # MB для GRASS
            "GRASS_REGION_PARAMETER": None,
            "GRASS_RASTER_FORMAT_OPT": "",
            "GRASS_RASTER_FORMAT_META": "",
        },
    )
    # Сохранить и добавить заполненные области водосбора в проект
    return QgsRasterLayer(basins_path, "basins")
