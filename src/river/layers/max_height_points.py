from pathlib import Path
from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsField,
    QgsFields,
    QgsProject,
    QgsVectorFileWriter,
    QgsVectorLayer,
    QgsWkbTypes,
)
from qgis.PyQt.QtCore import QVariant


def build_max_height_points(
    point_layer_path: Path,
    layer_name: str = "MaxHeightPoints",
) -> QgsVectorLayer:
    crs = QgsCoordinateReferenceSystem("EPSG:4326")

    # Определяем поля слоя
    fields = QgsFields()
    fields.append(QgsField("x", QVariant.Double))
    fields.append(QgsField("y", QVariant.Double))
    fields.append(QgsField("z", QVariant.Double))

    # Готовим параметры сохранения (SaveVectorOptions)
    options = QgsVectorFileWriter.SaveVectorOptions()
    options.driverName = "GPKG"
    options.fileEncoding = "UTF-8"
    options.layerName = layer_name

    # Получаем контекст преобразований из проекта
    transform_context = QgsProject.instance().transformContext()

    # Создаем GeoPackage-слой
    QgsVectorFileWriter.create(
        str(point_layer_path),
        fields,
        QgsWkbTypes.Point,
        crs,
        transform_context,
        options,
    )

    # Открываем и добавляем в проект
    uri = f"{str(point_layer_path)}|layername={layer_name}"
    return QgsVectorLayer(uri, layer_name, "ogr")
