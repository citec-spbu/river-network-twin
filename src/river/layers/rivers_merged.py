import processing
from pathlib import Path
from qgis.core import QgsVectorLayer

from src.river.layers.utils import load_quickosm_layer


def build_merged_layer(
    extent,
    merged_path: Path,
    rivers_path: Path,
    streams_path: Path,
) -> QgsVectorLayer:
    # Загрузить реки
    rivers_layer = load_quickosm_layer(
        "rivers",
        "waterway",
        "river",
        extent,
        rivers_path,
    )
    # Загрузить ручьи
    streams_layer = load_quickosm_layer(
        "streams",
        "waterway",
        "stream",
        extent,
        streams_path,
    )

    # Объединить слои рек и ручьев
    merge_result = processing.run(
        "qgis:mergevectorlayers",
        {
            "LAYERS": [rivers_layer, streams_layer],
            "CRS": rivers_layer.crs(),
            "OUTPUT": "TEMPORARY_OUTPUT",
        },
    )["OUTPUT"]
    merge_result = processing.run(
        "native:dissolve",
        {
            "INPUT": merge_result,
            "FIELD": [],
            "SEPARATE_DISJOINT": False,
            "OUTPUT": "TEMPORARY_OUTPUT",
        },
    )["OUTPUT"]
    processing.run(
        "native:multiparttosingleparts",
        {
            "INPUT": merge_result,
            "OUTPUT": str(merged_path),
        },
    )["OUTPUT"]

    # Добавить объединенный слой в проект
    return QgsVectorLayer(str(merged_path), "rivers_merged", "ogr")
