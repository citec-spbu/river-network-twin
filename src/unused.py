from qgis.core import (
    QgsProject,
    QgsPointXY,
    QgsRasterLayer,
    QgsVectorLayer,
    QgsGeometry,
    QgsField,
    QgsRaster,
    QgsFeature,
)
from qgis.PyQt.QtCore import QVariant
import processing


# Определить максимальную высоту для каждой линии
def determine_maximum_heights(end_y, project_folder):
    max_z = processing.run(
        "native:fieldcalculator",
        {
            "INPUT": end_y,
            "FIELD_NAME": "max_z",
            "FIELD_TYPE": 0,
            "FIELD_LENGTH": 0,
            "FIELD_PRECISION": 0,
            "FORMULA": 'if("start_z" > "end_z", "start_z", "end_z")',
            "OUTPUT": f"{project_folder}rivers_with_points.gpkg",
        },
    )["OUTPUT"]

    rivers_and_points = QgsVectorLayer(max_z, "rivers_and_points")
    QgsProject.instance().addMapLayer(rivers_and_points)
    return rivers_and_points


# Создать слой точек максимальной высоты
def create_max_height_points_layer():
    point_layer = QgsVectorLayer("Point?crs=epsg:4326", "MaxHeightPoints", "memory")
    QgsProject.instance().addMapLayer(point_layer)
    layer_provider = point_layer.dataProvider()
    layer_provider.addAttributes([QgsField("x", QVariant.Double)])
    layer_provider.addAttributes([QgsField("y", QVariant.Double)])
    layer_provider.addAttributes([QgsField("z", QVariant.Double)])
    point_layer.updateFields()
    return point_layer


# Добавить новые поля для хранения высотных данных
def add_elevation_fields(layer):
    layer_provider = layer.dataProvider()
    layer_provider.addAttributes([QgsField("start_z", QVariant.Double)])
    layer_provider.addAttributes([QgsField("end_z", QVariant.Double)])
    layer.updateFields()


# Начать редактирование и заполнение значений высоты
def populate_elevation_data(layer, dem_layer):
    layer.startEditing()
    line_provider = layer.dataProvider()

    for feature in layer.getFeatures():
        geom = feature.geometry()
        polyline = (
            geom.asMultiPolyline()[0] if geom.isMultipart() else geom.asPolyline()
        )

        start_point = QgsPointXY(polyline[0])
        end_point = QgsPointXY(polyline[-1])

        start_z = dem_layer.dataProvider().identify(
            start_point, QgsRaster.IdentifyFormatValue
        )
        end_z = dem_layer.dataProvider().identify(
            end_point, QgsRaster.IdentifyFormatValue
        )

        start_z_value = start_z.results()[1] if start_z.isValid() else None
        end_z_value = end_z.results()[1] if end_z.isValid() else None

        feature["start_z"] = start_z_value
        feature["end_z"] = end_z_value
        line_provider.changeAttributeValues(
            {
                feature.id(): {
                    line_provider.fields().indexOf("start_z"): start_z_value,
                    line_provider.fields().indexOf("end_z"): end_z_value,
                }
            }
        )

    layer.commitChanges()


def calculate_coordinates(layer_path, project_folder):
    start_x = processing.run(
        "native:fieldcalculator",
        {
            "INPUT": layer_path,
            "FIELD_NAME": "start_x",
            "FIELD_TYPE": 0,
            "FIELD_LENGTH": 0,
            "FIELD_PRECISION": 0,
            "FORMULA": "x(start_point($geometry))",
            "OUTPUT": "TEMPORARY_OUTPUT",
        },
    )["OUTPUT"]

    start_y = processing.run(
        "native:fieldcalculator",
        {
            "INPUT": start_x,
            "FIELD_NAME": "start_y",
            "FIELD_TYPE": 0,
            "FIELD_LENGTH": 0,
            "FIELD_PRECISION": 0,
            "FORMULA": "y(start_point($geometry))",
            "OUTPUT": "TEMPORARY_OUTPUT",
        },
    )["OUTPUT"]

    end_x = processing.run(
        "native:fieldcalculator",
        {
            "INPUT": start_y,
            "FIELD_NAME": "end_x",
            "FIELD_TYPE": 0,
            "FIELD_LENGTH": 0,
            "FIELD_PRECISION": 0,
            "FORMULA": "x(end_point($geometry))",
            "OUTPUT": "TEMPORARY_OUTPUT",
        },
    )["OUTPUT"]

    end_y = processing.run(
        "native:fieldcalculator",
        {
            "INPUT": end_x,
            "FIELD_NAME": "end_y",
            "FIELD_TYPE": 0,
            "FIELD_LENGTH": 0,
            "FIELD_PRECISION": 0,
            "FORMULA": "y(end_point($geometry))",
            "OUTPUT": f"{project_folder}rivers_with_points.gpkg",
        },
    )["OUTPUT"]

    return end_y


# Сохранить и добавить заполненные области водосбора в проект
def add_basins_layer(project_folder):
    basins = QgsRasterLayer(f"{project_folder}basins.sdat", "basins")
    QgsProject.instance().addMapLayer(basins)


# Загрузить реки и ручьи
def quickosm_query(key, value, extent):
    return processing.run(
        "quickosm:buildqueryextent",
        {"KEY": key, "VALUE": value, "EXTENT": extent, "TIMEOUT": 25},
    )["OUTPUT_URL"]


def download_and_add_layer(url, layer_name):
    file = processing.run(
        "native:filedownloader", {"URL": url, "OUTPUT": "TEMPORARY_OUTPUT"}
    )["OUTPUT"]
    layer = QgsVectorLayer(file + f"|layername={layer_name}", layer_name, "ogr")
    QgsProject.instance().addMapLayer(layer)
    return layer


# Объединить слои рек и ручьев
def merge_and_dissolve_layers(layers, project_folder):
    merged = processing.run(
        "qgis:mergevectorlayers",
        {"LAYERS": layers, "CRS": layers[0].crs(), "OUTPUT": "TEMPORARY_OUTPUT"},
    )["OUTPUT"]
    dissolved = processing.run(
        "native:dissolve",
        {
            "INPUT": merged,
            "FIELD": [],
            "SEPARATE_DISJOINT": False,
            "OUTPUT": "TEMPORARY_OUTPUT",
        },
    )["OUTPUT"]
    return processing.run(
        "native:multiparttosingleparts",
        {"INPUT": dissolved, "OUTPUT": f"{project_folder}merge_result.gpkg"},
    )["OUTPUT"]


# Использовать SAGA Fill Sinks для извлечения водосборов
def fill_sinks(reprojected_relief, project_folder):
    return processing.run(
        "sagang:fillsinkswangliu",
        {
            "ELEV": reprojected_relief,
            "FILLED": "TEMPORARY_OUTPUT",
            "FDIR": "TEMPORARY_OUTPUT",
            "WSHED": f"{project_folder}basins.sdat",
            "MINSLOPE": 0.01,
        },
    )["WSHED"]


# Получить ссылки на слои линий и точек
def process_maximum_height_points(rivers_layer, point_layer):
    line_layer_name = "rivers_and_points"
    layers = QgsProject.instance().mapLayersByName(line_layer_name)
    layer = layers[0]

    start_points = set()
    end_points = set()

    for feature in layer.getFeatures():
        start_x = feature["start_x"]
        start_y = feature["start_y"]
        end_x = feature["end_x"]
        end_y = feature["end_y"]

        if start_x is not None and start_y is not None:
            start_points.add((start_x, start_y))

        if end_x is not None and end_y is not None:
            end_points.add((end_x, end_y))

    fields = point_layer.fields()
    point_layer.startEditing()

    for feature in layer.getFeatures():
        if feature["max_z"] is not None:
            start_x = feature["start_x"]
            start_y = feature["start_y"]
            start_z = feature["start_z"]
            end_x = feature["end_x"]
            end_y = feature["end_y"]
            end_z = feature["end_z"]

            if start_x is not None and start_y is not None and start_z is not None:
                start_point = (start_x, start_y)
                if start_z == feature["max_z"] and start_point not in end_points:
                    point = QgsPointXY(start_x, start_y)
                    new_feature = QgsFeature()
                    new_feature.setFields(fields)
                    new_feature.setGeometry(QgsGeometry.fromPointXY(point))
                    new_feature["x"] = start_x
                    new_feature["y"] = start_y
                    new_feature["z"] = start_z
                    point_layer.addFeature(new_feature)

            if end_x is not None and end_y is not None and end_z is not None:
                end_point = (end_x, end_y)
                if end_z == feature["max_z"] and end_point not in start_points:
                    point = QgsPointXY(end_x, end_y)
                    new_feature = QgsFeature()
                    new_feature.setFields(fields)
                    new_feature.setGeometry(QgsGeometry.fromPointXY(point))
                    new_feature["x"] = end_x
                    new_feature["y"] = end_y
                    new_feature["z"] = end_z
                    point_layer.addFeature(new_feature)

    point_layer.commitChanges(True)
