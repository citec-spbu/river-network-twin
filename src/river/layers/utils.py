import processing
from qgis.utils import iface
from processing_saga_nextgen.saga_nextgen_plugin import SagaNextGenAlgorithmProvider
from qgis.core import QgsApplication, QgsVectorLayer, QgsField
from qgis.PyQt.QtCore import QVariant


def load_saga_algorithms():
    provider = SagaNextGenAlgorithmProvider()
    provider.loadAlgorithms()

    QgsApplication.processingRegistry().addProvider(provider=provider)


def load_quickosm_layer(
    layer_name,
    key,
    value,
    extent,
    output_path="TEMPORARY_OUTPUT",
    quickosm_layername="lines",
):
    query = processing.run(
        "quickosm:buildqueryextent",
        {
            "KEY": key,
            "VALUE": value,
            "EXTENT": extent,
            "TIMEOUT": 25,
        },
    )
    download_result = processing.run(
        "native:filedownloader",
        {
            "URL": query["OUTPUT_URL"],
            "OUTPUT": output_path,
        },
    )["OUTPUT"]
    layer = iface.addVectorLayer(
        download_result + f"|layername={quickosm_layername}", layer_name, "ogr"
    )
    return layer


def filter_rivers_by_params(
    rivers_layer, filters, layer_name="rivers_filtered"
) -> QgsVectorLayer:
    expr_parts = []
    for fld, (op, val) in filters.items():
        if isinstance(val, str):
            safe = val.replace("'", "''")
            val_repr = f"'{safe}'"
        else:
            val_repr = str(val)
        expr_parts.append(f'"{fld}" {op} {val_repr}')
    expression = " AND ".join(expr_parts) or "TRUE"

    result_path = processing.run(
        "native:extractbyexpression",
        {"INPUT": rivers_layer, "EXPRESSION": expression, "OUTPUT": "TEMPORARY_OUTPUT"},
    )["OUTPUT"]
    if isinstance(result_path, QgsVectorLayer):
        filtered = result_path
    else:
        filtered = QgsVectorLayer(result_path, layer_name, "ogr")
    filtered.setName(layer_name)
    return filtered


def compute_strahler(rivers_layer):
    node_map = {}
    edges = []
    next_node = 0

    for feat in rivers_layer.getFeatures():
        sx, sy, sz = feat["start_x"], feat["start_y"], feat["start_z"]
        ex, ey, ez = feat["end_x"], feat["end_y"], feat["end_z"]

        if sz is None or ez is None:
            continue

        if sz < ez:
            sx, sy, sz, ex, ey, ez = ex, ey, ez, sx, sy, sz

        for x, y in ((sx, sy), (ex, ey)):
            if (x, y) not in node_map:
                node_map[(x, y)] = next_node
                next_node += 1

        n1 = node_map[(sx, sy)]
        n2 = node_map[(ex, ey)]
        edges.append((n1, n2, feat.id()))

    incoming = {i: [] for i in range(next_node)}
    outgoing = {i: [] for i in range(next_node)}
    for u, v, fid in edges:
        outgoing[u].append((v, fid))
        incoming[v].append((u, fid))

    orders = {}

    def calc_edge(fid, from_node):
        if fid in orders:
            return orders[fid]
        preds = incoming[from_node]
        if not preds:
            orders[fid] = 1
            return 1
        child = [calc_edge(up_fid, up_node) for up_node, up_fid in preds]
        m = max(child)
        orders[fid] = m + 1 if child.count(m) > 1 else m
        return orders[fid]

    for node in range(next_node):
        if not outgoing[node]:
            for _, fid in incoming[node]:
                calc_edge(fid, node)

    rivers_layer.startEditing()
    if "strahler_order" not in [f.name() for f in rivers_layer.fields()]:
        rivers_layer.dataProvider().addAttributes(
            [QgsField("strahler_order", QVariant.Int)]
        )
        rivers_layer.updateFields()
    idx = rivers_layer.fields().indexOf("strahler_order")
    for _, _, fid in edges:
        rivers_layer.changeAttributeValue(fid, idx, orders.get(fid, 0))
    rivers_layer.commitChanges()

    return rivers_layer


def compute_river_length(end_y):
    length_result = processing.run(
        "native:fieldcalculator",
        {
            "INPUT": end_y,
            "FIELD_NAME": "length",
            "FIELD_TYPE": 0,
            "FIELD_LENGTH": 10,
            "FIELD_PRECISION": 3,
            "FORMULA": "$length",
            "OUTPUT": "TEMPORARY_OUTPUT",
        },
    )["OUTPUT"]
    if isinstance(length_result, QgsVectorLayer):
        return length_result
    return QgsVectorLayer(length_result, "rivers_with_length", "ogr")
