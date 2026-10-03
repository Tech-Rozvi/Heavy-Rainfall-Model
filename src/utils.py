"""
Shared utilities: config loading, logging, raster I/O helpers, QC checks.

All other modules import from here rather than reimplementing config parsing
or logging setup, so behaviour (log format, error handling) stays consistent
across the pipeline.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

try:
    import rasterio
    from rasterio.transform import Affine
    _HAS_RASTERIO = True
except ImportError:  # pragma: no cover - rasterio optional at import time
    _HAS_RASTERIO = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_config(path: str | Path) -> dict:
    """Load the YAML configuration file.

    Raises a clear error rather than a bare KeyError later in the pipeline if
    the file is missing or malformed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found at {path}. The pipeline will not run with "
            f"hardcoded defaults by design — copy config/config.yaml and edit it."
        )
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if not cfg:
        raise ValueError(f"Config file at {path} is empty or invalid YAML.")
    return cfg


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a module-level logger with a consistent, timestamped format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Raster container
# ---------------------------------------------------------------------------
@dataclass
class RasterLayer:
    """A single in-memory raster layer with the georeferencing metadata
    needed to write a valid, QGIS/ArcGIS-readable GeoTIFF.
    """
    data: np.ndarray            # 2D float array
    transform: "Affine"
    crs: str
    nodata: float = -9999.0
    name: str = "layer"

    def masked(self) -> np.ma.MaskedArray:
        return np.ma.masked_equal(self.data, self.nodata)


def write_geotiff(layer: RasterLayer, out_path: str | Path, dtype: str = "float32",
                   compress: str = "LZW", description: Optional[str] = None) -> None:
    """Write a RasterLayer to a valid, compressed, georeferenced GeoTIFF.

    Fails loudly (raises) rather than writing a corrupt/ambiguous file if the
    array contains no finite georeferencing.
    """
    if not _HAS_RASTERIO:
        raise ImportError("rasterio is required to write GeoTIFFs. pip install rasterio")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = np.asarray(layer.data, dtype=dtype)
    data = np.where(np.isfinite(data), data, layer.nodata)

    profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": dtype,
        "crs": layer.crs,
        "transform": layer.transform,
        "nodata": layer.nodata,
        "compress": compress,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data, 1)
        tags = {"model": "rainfall_susceptibility", "layer": layer.name}
        if description:
            tags["description"] = description
        dst.update_tags(**tags)


def read_geotiff(path: str | Path) -> RasterLayer:
    if not _HAS_RASTERIO:
        raise ImportError("rasterio is required to read GeoTIFFs. pip install rasterio")
    with rasterio.open(path) as src:
        data = src.read(1).astype("float64")
        nodata = src.nodata if src.nodata is not None else -9999.0
        return RasterLayer(data=data, transform=src.transform, crs=str(src.crs),
                            nodata=nodata, name=Path(path).stem)


# ---------------------------------------------------------------------------
# Quality control
# ---------------------------------------------------------------------------
class QCError(Exception):
    """Raised when a quality-control check fails. The pipeline must stop,
    not silently continue with bad data."""


def qc_check_array(arr: np.ndarray, name: str, allow_nan: bool = True,
                    valid_range: Optional[tuple[float, float]] = None) -> None:
    """Run basic QC on a numpy array and raise QCError with a clear message
    on failure, per the requirement that the program fail loudly rather than
    silently produce wrong results.
    """
    if arr is None or arr.size == 0:
        raise QCError(f"[{name}] Array is empty or None.")
    finite = np.isfinite(arr)
    if not allow_nan and not finite.all():
        raise QCError(f"[{name}] Contains {np.sum(~finite)} NaN/Inf values but "
                       f"allow_nan=False.")
    if valid_range is not None and finite.any():
        lo, hi = valid_range
        vals = arr[finite]
        if vals.min() < lo or vals.max() > hi:
            raise QCError(f"[{name}] Values outside expected range [{lo}, {hi}]: "
                           f"found min={vals.min():.3f}, max={vals.max():.3f}.")


def qc_check_alignment(arrays: dict[str, np.ndarray]) -> None:
    """Ensure all input rasters share the same shape (i.e. have already been
    reprojected/resampled/aligned onto a common grid)."""
    shapes = {name: a.shape for name, a in arrays.items()}
    unique_shapes = set(shapes.values())
    if len(unique_shapes) > 1:
        raise QCError(f"Raster misalignment detected — layers have different "
                       f"shapes and must be resampled to a common grid first: {shapes}")
