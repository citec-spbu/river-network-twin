import processing
from qgis.core import QgsVectorLayer


def build_rivers_and_points_layer(end_y, rivers_and_points_path: str) -> QgsVectorLayer:
    max_z = processing.run(
        "native:fieldcalculator",
        {
            "INPUT": end_y,
            "FIELD_NAME": "max_z",
            "FIELD_TYPE": 0,
            "FIELD_LENGTH": 0,
            "FIELD_PRECISION": 0,
            "FORMULA": 'if("start_z" > "end_z", "start_z", "end_z")',
            "OUTPUT": rivers_and_points_path,
        },
    )["OUTPUT"]

    return QgsVectorLayer(max_z, "rivers_and_points", "ogr")
