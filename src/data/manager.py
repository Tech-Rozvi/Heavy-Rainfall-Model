"""
manager.py — Unified data acquisition (local real files | remote API stream).

Modes
-----
api    — stream CHIRPS / DEM / ERA5 (AWS Zarr) / Natural Earth into memory
local  — load client-supplied real files under data/raw/

Synthetic data is never produced here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.utils import get_logger, read_geotiff
from src import terrain as terr
from src.data.streaming.remote_client import RemoteDataClient, StreamConfig

logger = get_logger(__name__)

CHIRPS_RESOLUTION_M = 5000.0


@dataclass
class AcquisitionResult:
    daily_precip: np.ndarray
    n_years: float
    dem: np.ndarray
    slope: np.ndarray
    aspect_exposure: np.ndarray
    temperature: np.ndarray
    relative_humidity: np.ndarray
    wind_speed: np.ndarray
    distance_to_coast: np.ndarray
    transform: Any
    crs: str
    resolution_m: float
    mode: str = "api"
    sources: dict = field(default_factory=dict)
    dates: list = field(default_factory=list)


def _fill_nan_nearest(arr: np.ndarray) -> np.ndarray:
    """Fill NaN holes/margins by nearest finite neighbour (no empty borders)."""
    a = np.asarray(arr, dtype="float64")
    if np.isfinite(a).all():
        return a
    if not np.isfinite(a).any():
        return np.zeros_like(a)
    try:
        from scipy.ndimage import distance_transform_edt
        mask = ~np.isfinite(a)
        idx = distance_transform_edt(mask, return_distances=False, return_indices=True)
        return a[tuple(idx)]
    except Exception:
        fill = float(np.nanmean(a)) if np.isfinite(a).any() else 0.0
        return np.where(np.isfinite(a), a, fill)


def _fit_to_shape(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resample to target (H, W) and fill any NaN margins so layers share a full grid."""
    if arr is None:
        return np.zeros(shape, dtype="float64")
    a = np.asarray(arr, dtype="float64")
    if a.ndim > 2:
        a = np.squeeze(a)
    if a.ndim == 0:
        return np.full(shape, float(a), dtype="float64")
    if a.size == 1:
        return np.full(shape, float(a.ravel()[0]), dtype="float64")
    a = _fill_nan_nearest(a)
    if a.shape == shape:
        return a
    try:
        from scipy.ndimage import zoom
        zy = shape[0] / max(a.shape[0], 1)
        zx = shape[1] / max(a.shape[1], 1)
        out = zoom(a, (zy, zx), order=1).astype("float64")
        # zoom can introduce tiny edge NaNs
        return _fill_nan_nearest(out)
    except Exception:
        out = np.zeros(shape, dtype="float64")
        h, w = min(shape[0], a.shape[0]), min(shape[1], a.shape[1])
        out[:h, :w] = a[:h, :w]
        if h < shape[0] or w < shape[1]:
            # replicate edge values into remaining margin
            if h < shape[0] and h > 0:
                out[h:, :w] = out[h - 1, :w]
            if w < shape[1] and w > 0:
                out[:, w:] = out[:, w - 1:w]
        return _fill_nan_nearest(out)


def _target_shape_from_bounds(
    bounds: tuple[float, float, float, float],
    resolution_m: float,
) -> tuple[int, int, Any]:
    from rasterio.transform import from_bounds

    minx, miny, maxx, maxy = bounds
    res_deg = resolution_m / 111_320.0
    width = max(2, int(np.ceil((maxx - minx) / res_deg)))
    height = max(2, int(np.ceil((maxy - miny) / res_deg)))
    transform = from_bounds(minx, miny, maxx, maxy, width, height)
    return height, width, transform


def bounds_from_aoi_path(path: str) -> tuple[float, float, float, float]:
    import geopandas as gpd

    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")
    minx, miny, maxx, maxy = gdf.total_bounds
    return float(minx), float(miny), float(maxx), float(maxy)


def resolve_gaul_bounds(
    country: Optional[str] = None,
    adm1: Optional[str] = None,
    adm2: Optional[str] = None,
    adm3: Optional[str] = None,
    gaul_data_dir: str = "data/boundaries/gaul",
    gaul_layer: Optional[str] = None,
) -> tuple[tuple[float, float, float, float], dict]:
    """
    Resolve administrative selection to WGS84 bounds.

    1. If GAUL vectors exist under gaul_data_dir, filter by name hierarchy.
    2. Else query Nominatim/OSM over HTTP (no local vectors required).
    """
    meta: dict[str, Any] = {
        "country": country,
        "adm1": adm1,
        "adm2": adm2,
        "adm3": adm3,
        "source": None,
    }
    if not country:
        raise ValueError("country is required for GAUL / geocoding resolution")

    # --- try local GAUL ---
    data_dir = Path(gaul_data_dir)
    layer_path = Path(gaul_layer) if gaul_layer else None
    if layer_path is None and data_dir.exists():
        for pat in ("*.gpkg", "*.shp", "*.geojson", "**/*.gpkg", "**/*.shp", "**/*.geojson"):
            hits = list(data_dir.glob(pat))
            if hits:
                layer_path = hits[0]
                break

    if layer_path is not None and layer_path.exists():
        try:
            import geopandas as gpd

            gdf = gpd.read_file(layer_path)
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")
            else:
                gdf = gdf.to_crs("EPSG:4326")
            cols = {c.lower(): c for c in gdf.columns}

            def _col(*names):
                for n in names:
                    if n.lower() in cols:
                        return cols[n.lower()]
                return None

            c0 = _col("ADM0_NAME", "gaul0_name", "NAME_0", "COUNTRY", "country")
            c1 = _col("ADM1_NAME", "gaul1_name", "NAME_1", "adm1")
            c2 = _col("ADM2_NAME", "gaul2_name", "NAME_2", "adm2")
            c3 = _col("ADM3_NAME", "gaul3_name", "NAME_3", "adm3")

            mask = np.ones(len(gdf), dtype=bool)
            if c0 and country:
                mask &= gdf[c0].astype(str).str.casefold() == country.casefold()
            if c1 and adm1:
                mask &= gdf[c1].astype(str).str.casefold() == adm1.casefold()
            if c2 and adm2:
                mask &= gdf[c2].astype(str).str.casefold() == adm2.casefold()
            if c3 and adm3:
                mask &= gdf[c3].astype(str).str.casefold() == adm3.casefold()
            sub = gdf[mask]
            if sub.empty:
                raise ValueError(f"No GAUL features matched {meta}")
            minx, miny, maxx, maxy = sub.total_bounds
            meta["source"] = f"local_gaul:{layer_path}"
            meta["n_features"] = int(len(sub))
            return (float(minx), float(miny), float(maxx), float(maxy)), meta
        except Exception as exc:
            logger.warning("Local GAUL resolution failed (%s); trying Nominatim", exc)

    # --- Nominatim / OSM fallback ---
    parts = [p for p in (adm3, adm2, adm1, country) if p]
    query = ", ".join(parts)
    import requests

    url = "https://nominatim.openstreetmap.org/search"
    headers = {"User-Agent": "HeavyRainfallModel/2.1 (tech-rozvi; research)"}
    params = {"q": query, "format": "json", "limit": 1}
    r = requests.get(url, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    results = r.json()
    if not results:
        raise RuntimeError(
            f"Nominatim found no place for '{query}'. "
            "Provide --aoi GeoJSON or place GAUL files under data/boundaries/gaul/."
        )
    hit = results[0]
    if "boundingbox" in hit:
        # Nominatim: [south, north, west, east]
        south, north, west, east = map(float, hit["boundingbox"])
        bounds = (west, south, east, north)
    else:
        lon, lat = float(hit["lon"]), float(hit["lat"])
        pad = 0.25
        bounds = (lon - pad, lat - pad, lon + pad, lat + pad)
    meta["source"] = "nominatim_osm"
    meta["display_name"] = hit.get("display_name")
    logger.info("Resolved AOI via Nominatim: %s → %s", query, bounds)
    return bounds, meta


class DataAcquisitionManager:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.dacfg = cfg.get("data_acquisition", {})
        raw = cfg.get("paths", {}).get("raw_dir", "data/raw")
        self.raw_dir = Path(raw)

    def _make_client(self) -> RemoteDataClient:
        """Instantiate RemoteDataClient with config-driven ERA5 endpoint."""
        atm = self.dacfg.get("atmospheric", {})
        endpoint = atm.get("endpoint", "s3://era5-pds/zarr/")
        requires_auth = bool(atm.get("requires_auth", False))
        cfg = StreamConfig(
            allow_ephemeral_temp=True,
            era5_endpoint=endpoint,
            era5_requires_auth=requires_auth,
        )
        # Must match RemoteDataClient.__init__(self, config=None, **kwargs)
        return RemoteDataClient(config=cfg)

    def acquire(
        self,
        mode: str = "api",
        aoi_bounds: Optional[tuple[float, float, float, float]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        resolution_m: Optional[float] = None,
    ) -> AcquisitionResult:
        mode = (mode or self.dacfg.get("mode", "api")).lower()
        if mode in ("api", "stream", "remote", "auto"):
            if aoi_bounds is None:
                raise ValueError(
                    "API mode requires AOI bounds (--aoi or --country / GAUL selection). "
                    "Synthetic fallback is disabled."
                )
            start_date = start_date or self.cfg.get("time", {}).get("start_date", "2015-01-01")
            end_date = end_date or self.cfg.get("time", {}).get("end_date", "2020-12-31")
            resolution_m = float(
                resolution_m
                or self.cfg.get("study_area", {}).get("target_resolution_m", CHIRPS_RESOLUTION_M)
            )
            resolution_m = max(resolution_m, CHIRPS_RESOLUTION_M)
            return self.acquire_remote(aoi_bounds, start_date, end_date, resolution_m)
        return self.acquire_local()

    def acquire_local(self) -> AcquisitionResult:
        raw = self.raw_dir
        precip_path = raw / "daily_precip.npy"
        dem_path = raw / "dem.tif"
        if not precip_path.exists():
            raise FileNotFoundError(
                f"Missing {precip_path}. Place real data or use --mode api "
                "with --aoi / --country. Synthetic auto-generation is disabled."
            )
        if not dem_path.exists():
            raise FileNotFoundError(f"Missing {dem_path}.")

        import rasterio

        daily = np.load(precip_path)
        with rasterio.open(dem_path) as src:
            dem = src.read(1).astype("float64")
            transform = src.transform
            crs = str(src.crs) if src.crs else "EPSG:4326"
            resolution_m = abs(src.transform.a)

        def _tif(name: str) -> np.ndarray:
            p = raw / f"{name}.tif"
            if not p.exists():
                raise FileNotFoundError(
                    f"Missing required raster {p}. Synthetic fill is disabled."
                )
            return read_geotiff(p).data.astype("float64")

        temperature = _tif("temperature")
        rh = _tif("relative_humidity")
        wind = _tif("wind_speed")
        dist = _tif("distance_to_coast")

        slope, aspect_deg = terr.compute_slope_aspect(dem, cellsize_m=float(resolution_m))
        wind_from = float(
            self.cfg.get("factors", {})
            .get("terrain_factors", {})
            .get("aspect_exposure", {})
            .get("dominant_wind_from_deg", 90.0)
        )
        aspect = terr.windward_exposure(aspect_deg, dominant_wind_from_deg=wind_from)
        n_years = float(
            self.cfg.get("time", {}).get(
                "n_years_hint", max(1.0, daily.shape[0] / 365.25)
            )
        )

        return AcquisitionResult(
            daily_precip=daily,
            n_years=n_years,
            dem=dem,
            slope=slope,
            aspect_exposure=aspect,
            temperature=temperature,
            relative_humidity=rh,
            wind_speed=wind,
            distance_to_coast=dist,
            transform=transform,
            crs=crs,
            resolution_m=float(resolution_m),
            mode="local",
            sources={"mode": "local", "raw_dir": str(raw)},
        )

    def acquire_remote(
        self,
        bounds: tuple[float, float, float, float],
        start_date: str,
        end_date: str,
        resolution_m: float = CHIRPS_RESOLUTION_M,
    ) -> AcquisitionResult:
        client = self._make_client()
        sources: dict[str, Any] = {}

        logger.info("REMOTE — CHIRPS %s → %s", start_date, end_date)
        try:
            precip = client.fetch_chirps_daily(bounds, start_date, end_date)
        except Exception as exc:
            raise RuntimeError(
                f"CHIRPS remote stream failed: {exc}. Synthetic precipitation is disabled."
            ) from exc
        sources["precipitation"] = precip.get("source", {})

        logger.info("REMOTE — Copernicus DEM")
        try:
            dem_r = client.fetch_dem(bounds)
        except Exception as exc:
            raise RuntimeError(
                f"DEM remote stream failed: {exc}. Synthetic DEM is disabled."
            ) from exc
        sources["dem"] = dem_r.get("source", {})

        logger.info("REMOTE — ERA5 AWS Zarr (no CDS key)")
        try:
            atm = client.fetch_era5_land_climatology(bounds, start_date, end_date)
        except Exception as exc:
            raise RuntimeError(
                f"ERA5 AWS Zarr stream failed: {exc}. Synthetic atmosphere is disabled."
            ) from exc
        sources["atmospheric"] = atm.get("source", {})

        logger.info("REMOTE — Natural Earth coast distance")
        try:
            coast = client.fetch_distance_to_coast(bounds, resolution_m)
        except Exception as exc:
            raise RuntimeError(
                f"Coastline stream failed: {exc}. Synthetic coast is disabled."
            ) from exc
        sources["coast"] = coast.get("source", {})

        daily = np.asarray(precip["precipitation"], dtype="float64")
        n_years = max(1.0, daily.shape[0] / 365.25)
        dates = list(precip.get("dates") or [])

        th, tw, target_transform = _target_shape_from_bounds(bounds, resolution_m)
        target = (th, tw)

        if daily.ndim == 3:
            if daily.shape[1:] != target and not (daily.shape[1] == 1 and daily.shape[2] == 1):
                stacked = [_fit_to_shape(daily[i], target) for i in range(daily.shape[0])]
                daily = np.stack(stacked, axis=0)
            spatial = daily.shape[1:] if daily.shape[1] > 1 else target
        else:
            spatial = target

        dem = _fill_nan_nearest(_fit_to_shape(dem_r["elevation"], spatial))
        temperature = _fill_nan_nearest(_fit_to_shape(np.squeeze(atm["temperature"]), spatial))
        rh = _fill_nan_nearest(_fit_to_shape(np.squeeze(atm["relative_humidity"]), spatial))
        wind = _fill_nan_nearest(_fit_to_shape(np.squeeze(atm["wind_speed"]), spatial))
        dist = _fill_nan_nearest(_fit_to_shape(coast["distance_to_coast"], spatial))

        slope, aspect_deg = terr.compute_slope_aspect(dem, cellsize_m=float(resolution_m))
        wind_from = float(
            self.cfg.get("factors", {})
            .get("terrain_factors", {})
            .get("aspect_exposure", {})
            .get("dominant_wind_from_deg", 90.0)
        )
        aspect = terr.windward_exposure(aspect_deg, dominant_wind_from_deg=wind_from)

        transform = (
            precip.get("transform")
            or dem_r.get("transform")
            or coast.get("transform")
            or target_transform
        )

        return AcquisitionResult(
            daily_precip=daily,
            n_years=n_years,
            dem=dem,
            slope=slope,
            aspect_exposure=aspect,
            temperature=temperature,
            relative_humidity=rh,
            wind_speed=wind,
            distance_to_coast=dist,
            transform=transform,
            crs="EPSG:4326",
            resolution_m=float(resolution_m),
            mode="api",
            sources=sources,
            dates=dates,
        )
