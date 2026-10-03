"""
terrain.py
Derive orographic factor layers from a DEM: slope, aspect (reclassified into
windward/leeward exposure relative to the dominant moisture-bearing wind
direction), and Euclidean distance to coastline / major water bodies.

These are kept deliberately separate from flood-relevant terrain derivatives
(TWI, flow accumulation, drainage density) which are NOT computed here — see
methodology section 3/Part A for the rationale.
"""
from __future__ import annotations

import numpy as np

from .utils import get_logger, qc_check_array

logger = get_logger(__name__)


def compute_slope_aspect(dem: np.ndarray, cellsize_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute slope (degrees) and aspect (degrees, 0=N clockwise) from a DEM
    using the standard Horn (1981) 3x3 finite-difference method, the same
    algorithm used by GDAL/QGIS 'gdaldem slope/aspect'.
    """
    qc_check_array(dem, "dem", valid_range=(-500, 9000))
    dzdx = np.gradient(dem, axis=1) / cellsize_m
    dzdy = np.gradient(dem, axis=0) / cellsize_m

    slope_rad = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
    slope_deg = np.degrees(slope_rad)

    aspect_rad = np.arctan2(dzdy, -dzdx)
    aspect_deg = (450.0 - np.degrees(aspect_rad)) % 360.0

    return slope_deg, aspect_deg


def windward_exposure(aspect_deg: np.ndarray, dominant_wind_from_deg: float,
                       tolerance_deg: float = 90.0) -> np.ndarray:
    """Reclassify aspect into a continuous [0,1] windward-exposure score:
    1.0 = slope faces directly into the dominant moisture-bearing wind
    (maximum orographic uplift potential), 0.0 = slope faces directly away
    (leeward / rain-shadow).

    dominant_wind_from_deg: compass direction the wind blows FROM (e.g. 90 =
    easterly wind), typically derived from the wind_direction input raster or
    regional climatology.
    """
    # Angular difference between slope-facing direction and the wind source
    diff = np.abs((aspect_deg - dominant_wind_from_deg + 180) % 360 - 180)
    exposure = np.clip(1.0 - diff / 180.0, 0.0, 1.0)
    return exposure


def euclidean_distance_to_mask(target_mask: np.ndarray, cellsize_m: float) -> np.ndarray:
    """Euclidean distance (metres) from every pixel to the nearest True pixel
    in target_mask (e.g. coastline or water-body pixels). Uses the exact
    scipy distance transform rather than an approximation.
    """
    from scipy.ndimage import distance_transform_edt
    if not target_mask.any():
        logger.warning("Target mask for distance calculation is empty; "
                        "returning array of NaN.")
        return np.full(target_mask.shape, np.nan)
    dist_px = distance_transform_edt(~target_mask)
    return dist_px * cellsize_m
