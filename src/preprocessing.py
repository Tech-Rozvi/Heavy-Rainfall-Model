"""
preprocessing.py
Reproject, resample, align, clip and mask all input rasters onto a single
common analysis grid, defined by the study-area boundary and the configured
target resolution.

This is the step that makes every downstream factor layer directly comparable
pixel-for-pixel -- skipping it is the single most common source of silently
wrong susceptibility maps.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .utils import RasterLayer, get_logger, qc_check_array

logger = get_logger(__name__)

try:
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.mask import mask as rio_mask
    from rasterio.transform import Affine
    import geopandas as gpd
    _HAS_GEO = True
except ImportError:  # pragma: no cover
    _HAS_GEO = False


def build_target_grid(bounds: tuple[float, float, float, float], crs: str,
                       resolution_m: float) -> tuple["Affine", int, int]:
    """Build the transform/width/height of the common analysis grid from a
    bounding box (in the target CRS, metres) and a resolution in metres.

    Explicitly documented: the resolution should not exceed the coarsest
    genuinely informative input dataset (e.g. do not claim 100 m detail from
    a 5 km reanalysis product -- see README 'Resolution' section).
    """
    minx, miny, maxx, maxy = bounds
    width = max(1, int(np.ceil((maxx - minx) / resolution_m)))
    height = max(1, int(np.ceil((maxy - miny) / resolution_m)))
    transform = Affine(resolution_m, 0.0, minx, 0.0, -resolution_m, maxy)
    return transform, width, height


def reproject_resample_align(src_path: str | Path, dst_transform: "Affine",
                              dst_width: int, dst_height: int, dst_crs: str,
                              resampling: str = "bilinear") -> RasterLayer:
    """Reproject + resample a single raster onto the common target grid.

    resampling: use 'bilinear' for continuous fields (precip, humidity,
    temperature, elevation) and 'nearest' or 'mode' for categorical fields
    (aspect class). Choosing the wrong resampling method for categorical data
    silently invents intermediate class values, so callers must be explicit.
    """
    if not _HAS_GEO:
        raise ImportError("rasterio/geopandas required for preprocessing.")

    resampling_enum = getattr(Resampling, resampling)
    with rasterio.open(src_path) as src:
        dst_array = np.full((dst_height, dst_width), np.nan, dtype="float64")
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=resampling_enum,
        )
    qc_check_array(dst_array, name=str(src_path), allow_nan=True)
    return RasterLayer(data=dst_array, transform=dst_transform, crs=dst_crs,
                        name=Path(src_path).stem)


def clip_to_boundary(layer: RasterLayer, boundary_path: str | Path) -> RasterLayer:
    """Mask a raster to the study-area boundary polygon, setting everything
    outside to nodata. This is what allows `STUDY_AREA = "kenya.shp"` to be
    the only thing a user changes to retarget the whole pipeline.
    """
    if not _HAS_GEO:
        raise ImportError("geopandas/rasterio required for clipping.")

    gdf = gpd.read_file(boundary_path)
    if gdf.crs is None:
        raise ValueError(f"Boundary file {boundary_path} has no CRS defined.")
    gdf = gdf.to_crs(layer.crs)
    geoms = [g.__geo_interface__ for g in gdf.geometry]

    # rasterio.mask expects a rasterio dataset; write a throwaway in-memory one.
    import rasterio.io
    profile = {
        "driver": "GTiff", "height": layer.data.shape[0], "width": layer.data.shape[1],
        "count": 1, "dtype": "float64", "crs": layer.crs, "transform": layer.transform,
        "nodata": layer.nodata,
    }
    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:
            dataset.write(layer.data, 1)
        with memfile.open() as dataset:
            out_arr, out_transform = rio_mask(dataset, geoms, crop=False,
                                               nodata=layer.nodata)
    return RasterLayer(data=out_arr[0], transform=layer.transform, crs=layer.crs,
                        nodata=layer.nodata, name=layer.name)


def align_all(layers: dict[str, RasterLayer]) -> dict[str, RasterLayer]:
    """Final consistency check that all layers share one grid before they are
    handed to classification/weighting. Raises QCError on mismatch."""
    from .utils import qc_check_alignment
    qc_check_alignment({k: v.data for k, v in layers.items()})
    logger.info("All %d input layers aligned to a common grid.", len(layers))
    return layers
