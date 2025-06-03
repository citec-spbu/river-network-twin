import processing
from pathlib import Path
from qgis.core import QgsVectorLayer


def build_rivers_and_points_layer(
    end_y, rivers_and_points_path: Path
) -> QgsVectorLayer:
    """Строит слой рек и точек с максимальными значениями высоты.

    Args:
        end_y: Входной слой с данными о высотах
        rivers_and_points_path: Путь для сохранения результата

    Returns:
        QgsVectorLayer: Слой рек и точек
    """
    max_z = processing.run(
        "native:fieldcalculator",
        {
            "INPUT": end_y,
            "FIELD_NAME": "max_z",
            "FIELD_TYPE": 0,
            "FIELD_LENGTH": 0,
            "FIELD_PRECISION": 0,
            "FORMULA": 'if("start_z" > "end_z", "start_z", "end_z")',
            "OUTPUT": str(rivers_and_points_path),
        },
    )["OUTPUT"]

    return QgsVectorLayer(max_z, "rivers_and_points", "ogr")
