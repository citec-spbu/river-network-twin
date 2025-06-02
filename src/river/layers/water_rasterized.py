from pathlib import Path
import processing
from osgeo import gdal
from qgis.core import QgsCoordinateReferenceSystem


def build_water_rasterized(
    rivers_path: Path,
    water_path: Path,
    dem_path: Path,
    output_path,
    rivers_width=0.0003,
):
    buffered_rivers = processing.run(
        "native:buffer",
        {
            "INPUT": str(rivers_path),
            "DISTANCE": rivers_width,
            "SEGMENTS": 5,
            "END_CAP_STYLE": 0,
            "JOIN_STYLE": 0,
            "MITER_LIMIT": 2,
            "DISSOLVE": False,
            "SEPARATE_DISJOINT": False,
            "OUTPUT": "TEMPORARY_OUTPUT",
        },
    )["OUTPUT"]

    merged_rivers_water = processing.run(
        "native:union",
        {
            "INPUT": buffered_rivers,
            "OVERLAY": str(water_path),
            "OVERLAY_FIELDS_PREFIX": "",
            "OUTPUT": "TEMPORARY_OUTPUT",
            "GRID_SIZE": None,
        },
    )["OUTPUT"]

    merged_rivers_water = processing.run(
        "native:promotetomulti",
        {"INPUT": merged_rivers_water, "OUTPUT": "TEMPORARY_OUTPUT"},
    )["OUTPUT"]

    mask = processing.run(
        "native:polygonfromlayerextent",
        {"INPUT": str(dem_path), "ROUND_TO": 0, "OUTPUT": "TEMPORARY_OUTPUT"},
    )["OUTPUT"]

    merged_rivers_water = processing.run(
        "gdal:clipvectorbypolygon",
        {
            "INPUT": merged_rivers_water,
            "MASK": mask,
            "OPTIONS": "",
            "OUTPUT": "TEMPORARY_OUTPUT",
        },
    )["OUTPUT"]

    dem_dataset = gdal.Open(str(dem_path))

    if dem_dataset:
        width = dem_dataset.RasterXSize
        height = dem_dataset.RasterYSize
    else:
        return None

    rasterized = processing.run(
        "gdal:rasterize",
        {
            "INPUT": merged_rivers_water,
            "FIELD": "",
            "BURN": 1,
            "USE_Z": False,
            "UNITS": 0,
            "WIDTH": width,
            "HEIGHT": height,
            "EXTENT": str(dem_path),
            "NODATA": 0,
            "OPTIONS": None,
            "DATA_TYPE": 5,
            "INIT": None,
            "INVERT": False,
            "EXTRA": "",
            "OUTPUT": "TEMPORARY_OUTPUT",
        },
    )["OUTPUT"]

    return processing.run(
        "gdal:warpreproject",
        {
            "INPUT": rasterized,
            "SOURCE_CRS": None,
            "TARGET_CRS": QgsCoordinateReferenceSystem("EPSG:3857"),
            "RESAMPLING": 0,
            "NODATA": None,
            "TARGET_RESOLUTION": None,
            "OPTIONS": None,
            "DATA_TYPE": 0,
            "TARGET_EXTENT": None,
            "TARGET_EXTENT_CRS": None,
            "MULTITHREADING": False,
            "EXTRA": "",
            "OUTPUT": str(output_path),
        },
    )["OUTPUT"]
