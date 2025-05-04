import processing
from qgis.core import QgsVectorLayer, QgsField, QgsSpatialIndex, QgsVectorLayerExporter
from qgis.PyQt.QtCore import QVariant
from .utils import compute_river_length, compute_strahler, filter_rivers_by_params


def build_rivers_by_object_filtered(
    end_y, filters, rivers_by_object_filtered_path: str
) -> QgsVectorLayer:
    segs = compute_river_length(end_y)  # поле 'length'
    segs = compute_strahler(segs)  # поле 'strahler_order'

    provider = segs.dataProvider()
    if "segment_id" not in [f.name() for f in provider.fields()]:
        segs.startEditing()
        provider.addAttributes([QgsField("segment_id", QVariant.Int)])
        segs.updateFields()
        idx_seg = segs.fields().indexOf("segment_id")
        for f in segs.getFeatures():
            segs.changeAttributeValue(f.id(), idx_seg, f.id())
        segs.commitChanges()

    segs.startEditing()
    if "group_id" not in [f.name() for f in provider.fields()]:
        segs.startEditing()
        provider.addAttributes([QgsField("group_id", QVariant.Int)])
        segs.updateFields()
        segs.commitChanges()

    idx_gid = segs.fields().indexOf("group_id")
    idx_seg = segs.fields().indexOf("segment_id")

    feats = {f.id(): f for f in segs.getFeatures()}
    index = QgsSpatialIndex()
    for feat in feats.values():
        index.insertFeature(feat)

    visited = set()
    current_gid = 0

    segs.startEditing()
    for fid in feats:
        if fid in visited:
            continue
        current_gid += 1
        stack = [fid]
        visited.add(fid)
        while stack:
            cur = stack.pop()
            segs.changeAttributeValue(cur, idx_gid, current_gid)
            bbox = feats[cur].geometry().boundingBox()
            for nb in index.intersects(bbox):
                if nb in visited:
                    continue
                if feats[cur].geometry().intersects(feats[nb].geometry()):
                    visited.add(nb)
                    stack.append(nb)
    segs.commitChanges()

    result = processing.run(
        "native:dissolve",
        {
            "INPUT": segs,
            "FIELD": ["group_id"],
            "OUTPUT": rivers_by_object_filtered_path,
        },
    )
    dissolved = QgsVectorLayer(result["OUTPUT"], "rivers_by_object", "ogr")

    # Добавить поля: segments, total_length, max_order
    dissolved.startEditing()
    dp = dissolved.dataProvider()
    dp.addAttributes(
        [
            QgsField("segments", QVariant.String),
            QgsField("total_length", QVariant.Double),
            QgsField("max_strahler_order", QVariant.Int),
        ]
    )
    dissolved.updateFields()

    idx_segs = dissolved.fields().indexOf("segments")
    idx_totlen = dissolved.fields().indexOf("total_length")
    idx_maxorder = dissolved.fields().indexOf("max_strahler_order")

    group_map = {}
    for f in segs.getFeatures():
        gid = f["group_id"]
        sid = f["segment_id"]
        group_map.setdefault(gid, []).append(sid)

    for feat in dissolved.getFeatures():
        gid = feat["group_id"]
        seg_ids = group_map.get(gid, [])
        # segments как CSV
        seg_list_str = ",".join(str(s) for s in seg_ids)
        dissolved.changeAttributeValue(feat.id(), idx_segs, seg_list_str)

        # total_length
        total_len = sum(segs.getFeature(s).attribute("length") or 0.0 for s in seg_ids)
        dissolved.changeAttributeValue(feat.id(), idx_totlen, total_len)

        # max_strahler_order
        max_ord = max(
            (segs.getFeature(s).attribute("strahler_order") or 0) for s in seg_ids
        )
        dissolved.changeAttributeValue(feat.id(), idx_maxorder, max_ord)

    dissolved.commitChanges()

    QgsVectorLayerExporter.exportLayer(
        dissolved, rivers_by_object_filtered_path, "GPKG", dissolved.crs(), False
    )
    final_layer = QgsVectorLayer(
        rivers_by_object_filtered_path, "rivers_by_object", "ogr"
    )
    rivers_by_object_filtered = filter_rivers_by_params(
        final_layer, filters, "rivers_by_object_filtered"
    )
    return rivers_by_object_filtered
