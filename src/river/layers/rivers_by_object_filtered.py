import processing
from qgis.core import QgsVectorLayer, QgsField, QgsSpatialIndex, QgsVectorLayerExporter
from qgis.PyQt.QtCore import QVariant
from .utils import compute_river_length, compute_strahler, filter_rivers_by_params


def build_rivers_by_object_filtered(
    end_y, filters, rivers_by_object_filtered_path: str
) -> QgsVectorLayer:
    rivers_merged = compute_river_length(end_y)
    rivers_merged = compute_strahler(rivers_merged)

    provider = rivers_merged.dataProvider()
    existing_fields = [f.name() for f in provider.fields()]

    rivers_merged.startEditing()
    if "group_id" not in [f.name() for f in provider.fields()]:
        provider.addAttributes([QgsField("group_id", QVariant.Int)])
    if "segment_id" not in existing_fields:
        provider.addAttributes([QgsField("segment_id", QVariant.Int)])
        rivers_merged.updateFields()
        idx_segment_id = rivers_merged.fields().indexOf("segment_id")
        for f in rivers_merged.getFeatures():
            rivers_merged.changeAttributeValue(f.id(), idx_segment_id, f.id())
    rivers_merged.commitChanges()

    idx_gid = rivers_merged.fields().indexOf("group_id")
    idx_segment_id = rivers_merged.fields().indexOf("segment_id")

    feats = {f.id(): f for f in rivers_merged.getFeatures()}
    index = QgsSpatialIndex()
    for feat in feats.values():
        index.insertFeature(feat)

    visited = set()
    current_gid = 0

    rivers_merged.startEditing()

    for fid, _ in feats.items():
        if fid in visited:
            continue
        current_gid += 1
        stack = [fid]
        visited.add(fid)
        while stack:
            cur = stack.pop()
            rivers_merged.changeAttributeValue(cur, idx_gid, current_gid)
            bbox = feats[cur].geometry().boundingBox()
            nbrs = index.intersects(bbox)
            for nb in nbrs:
                if nb in visited:
                    continue
                if feats[cur].geometry().intersects(feats[nb].geometry()):
                    visited.add(nb)
                    stack.append(nb)

    rivers_merged.commitChanges()

    result = processing.run(
        "native:dissolve",
        {
            "INPUT": rivers_merged,
            "FIELD": ["group_id"],
            "OUTPUT": rivers_by_object_filtered_path,
        },
    )
    dissolved_layer = QgsVectorLayer(result["OUTPUT"], "rivers_by_object", "ogr")

    # Добавляем поле segments для хранения списка исходных segment_id
    dissolved_layer.startEditing()
    dissolved_provider = dissolved_layer.dataProvider()
    dissolved_provider.addAttributes([QgsField("segments", QVariant.String)])
    dissolved_layer.updateFields()
    idx_segments = dissolved_layer.fields().indexOf("segments")

    group_segments = {}

    for f in rivers_merged.getFeatures():
        gid = f["group_id"]
        sid = f["segment_id"]
        if gid not in group_segments:
            group_segments[gid] = []
        group_segments[gid].append(sid)

    for f in dissolved_layer.getFeatures():
        gid = f["group_id"]
        seg_list = group_segments.get(gid, [])
        seg_list_str = ",".join(str(s) for s in seg_list)
        dissolved_layer.changeAttributeValue(f.id(), idx_segments, seg_list_str)

    dissolved_layer.commitChanges()

    QgsVectorLayerExporter.exportLayer(
        dissolved_layer,
        rivers_by_object_filtered_path,
        "GPKG",
        dissolved_layer.crs(),
        False,
    )
    final_layer = QgsVectorLayer(
        rivers_by_object_filtered_path, "rivers_by_object", "ogr"
    )
    rivers_by_object_filtered = filter_rivers_by_params(
        final_layer, filters, "rivers_by_object_filtered"
    )
    return rivers_by_object_filtered
