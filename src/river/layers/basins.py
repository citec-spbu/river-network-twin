import processing
from qgis.core import QgsRasterLayer
from ...river.layers.utils import load_saga_algorithms


def build_basins_layer(reprojected_relief, basins_path: str) -> QgsRasterLayer:
    load_saga_algorithms()

    # Использовать SAGA Fill Sinks для извлечения водосборов
    processing.run(
        "sagang:fillsinkswangliu",
        {
            "ELEV": reprojected_relief,
            "FILLED": "TEMPORARY_OUTPUT",
            "FDIR": "TEMPORARY_OUTPUT",
            "WSHED": basins_path,
            "MINSLOPE": 0.01,
        },
    )

    # Сохранить и добавить заполненные области водосбора в проект
    return QgsRasterLayer(basins_path, "basins")
