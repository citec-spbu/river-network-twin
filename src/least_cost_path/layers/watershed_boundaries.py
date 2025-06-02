from pathlib import Path
import networkx as nx
from qgis.core import (
    QgsFeature,
    QgsField,
    QgsFields,
    QgsGeometry,
    QgsProject,
    QgsVectorFileWriter,
    QgsVectorLayer,
    QgsWkbTypes,
)
from qgis.PyQt.QtCore import QVariant


def build_watershed_boundaries(
    lcp_layer,
    watershed_boundaries_path: Path,
    layer_name: str = "watershed_boundaries",
):
    # Собираем граф из сегментов линий
    g = nx.Graph()
    node_coords = {}

    for feat in lcp_layer.getFeatures():
        geom = feat.geometry()
        lines = geom.asMultiPolyline() if geom.isMultipart() else [geom.asPolyline()]
        for line in lines:
            for i in range(len(line) - 1):
                p1, p2 = line[i], line[i + 1]
                n1 = (round(p1.x(), 3), round(p1.y(), 3))
                n2 = (round(p2.x(), 3), round(p2.y(), 3))
                g.add_edge(n1, n2)
                node_coords[n1] = p1
                node_coords[n2] = p2

    fields = QgsFields()
    fields.append(QgsField("comp_id", QVariant.Int))

    crs = lcp_layer.crs()
    transform_context = QgsProject.instance().transformContext()

    options = QgsVectorFileWriter.SaveVectorOptions()
    options.driverName = "GPKG"
    options.fileEncoding = "UTF-8"
    options.layerName = layer_name

    QgsVectorFileWriter.create(
        str(watershed_boundaries_path),
        fields,
        QgsWkbTypes.Polygon,
        crs,
        transform_context,
        options,
    )

    uri = f"{str(watershed_boundaries_path)}|layername={layer_name}"
    final_layer = QgsVectorLayer(uri, layer_name, "ogr")

    final_layer.startEditing()
    prov = final_layer.dataProvider()

    comp_id = 0
    for comp in nx.connected_components(g):
        # Компонент с < 3 узлами не может содержать полигон
        if len(comp) < 3:
            continue

        sub_g = g.subgraph(comp)
        if not nx.cycle_basis(sub_g):
            continue

        d = sub_g.to_directed()
        cycles = list(nx.simple_cycles(d))

        max_area = 0.0
        best_geom = None

        for cycle in cycles:
            pts = [node_coords[n] for n in cycle]
            if len(pts) < 3:
                continue

            if pts[0] != pts[-1]:
                pts.append(pts[0])

            poly_geom = QgsGeometry.fromPolygonXY([pts])
            if not poly_geom.isGeosValid() or poly_geom.isEmpty():
                continue

            area = poly_geom.area()
            if area > max_area:
                max_area = area
                best_geom = poly_geom

        if best_geom is not None:
            feat = QgsFeature()
            feat.setFields(fields)
            feat.setAttribute("comp_id", comp_id)
            feat.setGeometry(best_geom)
            prov.addFeatures([feat])
            comp_id += 1

    final_layer.commitChanges()
    final_layer.updateExtents()
    final_layer.triggerRepaint()
    return final_layer
