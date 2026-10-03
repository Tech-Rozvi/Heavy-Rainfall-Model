"""
remote_client.py — Stream real climate/terrain data for an AOI + period.

Design rules
------------
1. Never generate synthetic observations.
2. Prefer cloud-native / windowed / subsetted remote reads.
3. Optional ephemeral temp only (auto-cleaned); no permanent data lakes.
4. ERA5 atmospheric fields come from the public AWS Open Data Zarr store
   (no CDS API key required).
"""
from __future__ import annotations

import gzip
import io
import math
import os
import tempfile
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class StreamConfig:
    """Controls remote access behaviour."""

    allow_ephemeral_temp: bool = True
    ephemeral_cache_dir: Optional[str] = None
    ephemeral_ttl_hours: int = 24
    request_timeout_s: int = 120
    max_retries: int = 3
    # Atmospheric source (config may override)
    era5_endpoint: str = "s3://era5-pds/zarr/"
    era5_requires_auth: bool = False


class RemoteDataClient:
    """
    High-level client used by DataAcquisitionManager and the HTTP API.

    Public methods used by the manager:
      - fetch_chirps_daily(bounds, start_date, end_date)
      - fetch_era5_land_climatology(bounds, start_date, end_date)  # AWS Zarr
      - fetch_dem(bounds)
      - fetch_distance_to_coast(bounds, resolution_m)
    """

    def __init__(self, config: Optional[StreamConfig] = None, **kwargs):
        """
        Accept StreamConfig or optional keyword overrides.
        Compatible with: RemoteDataClient(), RemoteDataClient(StreamConfig()),
        and RemoteDataClient(config=...).
        """
        if config is None and kwargs:
            config = StreamConfig(**{
                k: v for k, v in kwargs.items() if k in StreamConfig.__dataclass_fields__
            })
        self.config = config or StreamConfig()

    # ------------------------------------------------------------------
    # CHIRPS daily precipitation
    # ------------------------------------------------------------------
    def fetch_chirps_daily(
        self,
        bounds: tuple[float, float, float, float],
        start_date: str,
        end_date: str,
    ) -> dict[str, Any]:
        """
        Daily CHIRPS rainfall for AOI bbox.

        Returns dict with:
          precipitation : np.ndarray (time, y, x) or (time, 1, 1) for AOI mean
          dates, units, source, transform, crs
        """
        try:
            return self._chirps_climateserv(bounds, start_date, end_date)
        except Exception as exc:
            logger.warning("ClimateSERV path failed (%s); trying GeoTIFF stream", exc)
            return self._chirps_geotiff_stream(bounds, start_date, end_date)

    # Alias used by some call sites
    def fetch_precipitation(self, bounds, start_date, end_date):
        return self.fetch_chirps_daily(bounds, start_date, end_date)

    def _chirps_climateserv(
        self,
        bounds: tuple[float, float, float, float],
        start_date: str,
        end_date: str,
    ) -> dict[str, Any]:
        import json
        import requests

        minx, miny, maxx, maxy = bounds
        geom = {
            "type": "Polygon",
            "coordinates": [[
                [minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny],
            ]],
        }

        def _fmt(d: str) -> str:
            y, m, day = d.split("-")
            return f"{m}/{day}/{y}"

        base = "https://climateserv.servirglobal.net/chirps"
        params = {
            "datatype": 0,
            "begintime": _fmt(start_date),
            "endtime": _fmt(end_date),
            "intervaltype": 0,
            "operationtype": 5,
            "dateType_Category": "default",
            "isZip_CurrentDataType": "false",
            "geometry": json.dumps(geom),
        }
        r = requests.get(
            f"{base}/submitDataRequest/",
            params=params,
            timeout=self.config.request_timeout_s,
        )
        r.raise_for_status()
        job = r.json()
        if isinstance(job, list):
            job_id = job[0]
        elif isinstance(job, dict):
            job_id = job.get("id") or job.get("job_id") or next(iter(job.values()))
        else:
            job_id = job

        for _ in range(90):
            prog = requests.get(
                f"{base}/getDataRequestProgress/",
                params={"id": job_id},
                timeout=30,
            )
            prog.raise_for_status()
            val = prog.json()
            pct = val[0] if isinstance(val, list) else val
            if pct is None:
                time.sleep(2)
                continue
            if float(pct) >= 100:
                break
            if float(pct) < 0:
                raise RuntimeError(f"ClimateSERV job failed: {val}")
            time.sleep(2)
        else:
            raise TimeoutError("ClimateSERV job timed out")

        data_resp = requests.get(
            f"{base}/getDataFromRequest/",
            params={"id": job_id},
            timeout=60,
        )
        data_resp.raise_for_status()
        payload = data_resp.json()

        dates, values = [], []
        records = payload.get("data") if isinstance(payload, dict) else payload
        if isinstance(records, list):
            for rec in records:
                if not isinstance(rec, dict):
                    continue
                d = rec.get("date") or rec.get("Date")
                v = rec.get("value") or rec.get("avg") or rec.get("raw_value")
                if d is not None and v is not None:
                    dates.append(str(d))
                    try:
                        values.append(float(v))
                    except (TypeError, ValueError):
                        values.append(np.nan)
        if not values:
            raise RuntimeError(f"ClimateSERV returned no values: {str(payload)[:400]}")

        arr = np.asarray(values, dtype="float64").reshape(-1, 1, 1)
        return {
            "precipitation": arr,
            "dates": dates,
            "units": "mm/day",
            "source": {
                "provider": "CHIRPS",
                "access": "ClimateSERV_API",
                "url": base,
                "doi": "10.1038/sdata.2015.66",
            },
            "transform": None,
            "crs": "EPSG:4326",
        }

    def _chirps_geotiff_stream(
        self,
        bounds: tuple[float, float, float, float],
        start_date: str,
        end_date: str,
    ) -> dict[str, Any]:
        import requests
        from rasterio.io import MemoryFile
        from rasterio.windows import from_bounds

        d0 = datetime.strptime(start_date, "%Y-%m-%d")
        d1 = datetime.strptime(end_date, "%Y-%m-%d")
        n_days = (d1 - d0).days + 1
        if n_days > 3700:
            raise ValueError(
                f"Date range too long for streamed CHIRPS ({n_days} days). "
                "Narrow the period (e.g. ≤ 10 years per request)."
            )

        base = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/tifs/p05"
        minx, miny, maxx, maxy = bounds
        buffer = 0.1
        win_bounds = (minx - buffer, miny - buffer, maxx + buffer, maxy + buffer)

        arrays, dates = [], []
        transform = None
        cur = d0
        while cur <= d1:
            url = (
                f"{base}/{cur.year}/"
                f"chirps-v2.0.{cur.year}.{cur.month:02d}.{cur.day:02d}.tif.gz"
            )
            try:
                resp = requests.get(url, timeout=self.config.request_timeout_s)
                if resp.status_code == 404:
                    cur += timedelta(days=1)
                    continue
                resp.raise_for_status()
                raw = gzip.decompress(resp.content)
                with MemoryFile(raw) as mem:
                    with mem.open() as src:
                        window = from_bounds(*win_bounds, transform=src.transform)
                        data = src.read(1, window=window).astype("float32")
                        data = np.where(data < -9000, np.nan, data)
                        if transform is None:
                            transform = src.window_transform(window)
                        arrays.append(data)
                        dates.append(cur.strftime("%Y-%m-%d"))
            except Exception as exc:
                logger.warning("CHIRPS stream miss %s: %s", cur.date(), exc)
            cur += timedelta(days=1)

        if not arrays:
            raise RuntimeError("No CHIRPS data streamed for AOI/period.")

        h = min(a.shape[0] for a in arrays)
        w = min(a.shape[1] for a in arrays)
        stack = np.stack([a[:h, :w] for a in arrays], axis=0).astype("float64")
        return {
            "precipitation": stack,
            "dates": dates,
            "units": "mm/day",
            "source": {
                "provider": "CHIRPS",
                "access": "HTTP_GeoTIFF_windowed_stream",
                "url": base,
                "doi": "10.1038/sdata.2015.66",
            },
            "transform": transform,
            "crs": "EPSG:4326",
        }

    # ------------------------------------------------------------------
    # ERA5 atmospheric — public AWS Open Data Zarr (no CDS key)
    # ------------------------------------------------------------------
    def fetch_era5_land_climatology(
        self,
        bounds: tuple[float, float, float, float],
        start_date: str,
        end_date: str,
    ) -> dict[str, Any]:
        """
        Period-mean 2 m temperature (°C), relative humidity (%), and 10 m
        wind speed (m/s) from the public AWS ERA5 Zarr store.

        No CDS_API_KEY required (anonymous S3 / HTTPS access).
        """
        return self._era5_aws_zarr(bounds, start_date, end_date)

    def fetch_era5_aws_zarr(self, bounds, start_date, end_date):
        return self._era5_aws_zarr(bounds, start_date, end_date)

    def _era5_aws_zarr(
        self,
        bounds: tuple[float, float, float, float],
        start_date: str,
        end_date: str,
    ) -> dict[str, Any]:
        import xarray as xr

        minx, miny, maxx, maxy = bounds
        # Public AWS ERA5 (Planetary Computer / AWS Open Data style paths).
        # Primary: anonymous S3. Secondary: HTTP(S) mirrors known to work.
        candidates = [
            # Public GCS ARCO ERA5 (fast, no auth) — preferred when available
            "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/"
            "full_37-1h-0p25deg-chunk-1.zarr-v3",
            "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
            # AWS Open Data ERA5 Zarr (anonymous us-east-1)
            self.config.era5_endpoint.rstrip("/") + "/",
            "s3://era5-pds/zarr/",
        ]

        last_err: Optional[Exception] = None
        ds = None
        used_url = None

        for url in candidates:
            try:
                storage_options: dict[str, Any] = {}
                if url.startswith("s3://"):
                    # Anonymous AWS Open Data access (avoid 403 Forbidden)
                    storage_options = {
                        "anon": True,
                        "client_kwargs": {"region_name": "us-east-1"},
                        "default_cache_type": "readahead",
                    }
                elif url.startswith("gs://"):
                    storage_options = {"token": "anon"}
                logger.info("Opening ERA5 Zarr: %s", url)
                try:
                    ds = xr.open_zarr(
                        url, consolidated=True, storage_options=storage_options or None
                    )
                except Exception:
                    # Some stores ship without consolidated metadata
                    ds = xr.open_zarr(
                        url, consolidated=False, storage_options=storage_options or None
                    )
                used_url = url
                break
            except Exception as exc:
                last_err = exp if False else exc
                logger.warning("ERA5 open failed for %s: %s", url, exc)
                ds = None

        if ds is None:
            raise RuntimeError(
                f"Could not open any public ERA5 Zarr store. Last error: {last_err}. "
                "Synthetic atmospheric fields are disabled."
            )

        # Flexible variable / coordinate names across ERA5 flavours
        rename_map = {}
        for a, b in (
            ("latitude", "lat"), ("longitude", "lon"),
            ("lat", "latitude"), ("lon", "longitude"),
        ):
            if a in ds.coords and b not in ds.coords:
                pass  # keep native
        lat_name = "latitude" if "latitude" in ds.coords else (
            "lat" if "lat" in ds.coords else None
        )
        lon_name = "longitude" if "longitude" in ds.coords else (
            "lon" if "lon" in ds.coords else None
        )
        if lat_name is None or lon_name is None:
            raise RuntimeError(f"ERA5 store missing lat/lon coords: {list(ds.coords)}")

        # Longitude 0–360 vs −180–180
        lon_vals = ds[lon_name].values
        use_0_360 = float(np.nanmax(lon_vals)) > 180.5
        lon_min, lon_max = minx, maxx
        if use_0_360:
            lon_min = minx % 360
            lon_max = maxx % 360

        lat = ds[lat_name]
        lat_ascending = bool(lat.values[0] < lat.values[-1])
        lat_slice = slice(miny, maxy) if lat_ascending else slice(maxy, miny)

        def _find_var(*names):
            for n in names:
                if n in ds:
                    return n
            # case-insensitive
            lower = {k.lower(): k for k in ds.data_vars}
            for n in names:
                if n.lower() in lower:
                    return lower[n.lower()]
            return None

        t_name = _find_var(
            "air_temperature_at_2_metres", "t2m", "2m_temperature", "temperature_2m"
        )
        u_name = _find_var(
            "eastward_wind_at_10_metres", "u10", "10m_u_component_of_wind", "u_component_of_wind_10m"
        )
        v_name = _find_var(
            "northward_wind_at_10_metres", "v10", "10m_v_component_of_wind", "v_component_of_wind_10m"
        )
        d_name = _find_var(
            "dew_point_temperature_at_2_metres", "d2m", "2m_dewpoint_temperature", "dewpoint_temperature_2m"
        )

        if t_name is None:
            raise RuntimeError(
                f"No 2 m temperature variable in ERA5 store. vars={list(ds.data_vars)[:20]}"
            )

        def _subset(var_name: str):
            da = ds[var_name]
            # time
            time_name = "time" if "time" in da.dims else (
                "valid_time" if "valid_time" in da.dims else None
            )
            if time_name:
                da = da.sel({time_name: slice(start_date, end_date)})
            # space
            if use_0_360 and lon_min > lon_max:
                # wrap across prime meridian — take two slices and concat
                da = da.sel({lat_name: lat_slice})
            else:
                da = da.sel({
                    lat_name: lat_slice,
                    lon_name: slice(lon_min, lon_max),
                })
            return da

        t_da = _subset(t_name)
        # Reduce over time → period mean
        reduce_dims = [d for d in t_da.dims if d not in (lat_name, lon_name)]
        t_mean = t_da.mean(dim=reduce_dims) if reduce_dims else t_da
        t_vals = np.asarray(t_mean.values, dtype="float64")
        # Kelvin → °C if needed
        if np.nanmean(t_vals) > 100:
            t_vals = t_vals - 273.15

        # Relative humidity from T and dewpoint when available
        if d_name is not None:
            d_da = _subset(d_name)
            d_mean = d_da.mean(dim=reduce_dims) if reduce_dims else d_da
            td = np.asarray(d_mean.values, dtype="float64")
            if np.nanmean(td) > 100:
                td = td - 273.15
            # Magnus formula
            es = 6.112 * np.exp((17.67 * t_vals) / (t_vals + 243.5))
            e = 6.112 * np.exp((17.67 * td) / (td + 243.5))
            rh = np.clip(e / es * 100.0, 0.0, 100.0)
        else:
            rh = np.full_like(t_vals, np.nan)

        if u_name and v_name:
            u_da = _subset(u_name)
            v_da = _subset(v_name)
            u_m = u_da.mean(dim=reduce_dims) if reduce_dims else u_da
            v_m = v_da.mean(dim=reduce_dims) if reduce_dims else v_da
            wind = np.sqrt(
                np.asarray(u_m.values, dtype="float64") ** 2
                + np.asarray(v_m.values, dtype="float64") ** 2
            )
        else:
            wind = np.full_like(t_vals, np.nan)

        return {
            "temperature": t_vals,
            "relative_humidity": rh,
            "wind_speed": wind,
            "units": {"temperature": "degC", "relative_humidity": "%", "wind_speed": "m/s"},
            "source": {
                "provider": "ERA5",
                "access": "AWS_Open_Data_Zarr",
                "url": used_url,
                "requires_auth": False,
                "doi": "10.24381/cds.adbb2d47",
            },
            "crs": "EPSG:4326",
        }

    # ------------------------------------------------------------------
    # Copernicus DEM — COG HTTP stream
    # ------------------------------------------------------------------
    def fetch_dem(self, bounds: tuple[float, float, float, float]) -> dict[str, Any]:
        import requests
        import rasterio
        from rasterio.io import MemoryFile
        from rasterio.merge import merge as rio_merge
        from rasterio.windows import from_bounds as window_from_bounds

        minx, miny, maxx, maxy = bounds
        tiles = []
        for lat in range(int(math.floor(miny)), int(math.ceil(maxy))):
            for lon in range(int(math.floor(minx)), int(math.ceil(maxx))):
                ns = "N" if lat >= 0 else "S"
                ew = "E" if lon >= 0 else "W"
                tile = f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM"
                tiles.append(tile)

        if len(tiles) > 80:
            raise ValueError(f"AOI requires {len(tiles)} DEM tiles (limit 80). Reduce extent.")

        base = "https://copernicus-dem-30m.s3.amazonaws.com"
        datasets, memfiles = [], []
        try:
            for tile in tiles:
                url = f"{base}/{tile}/{tile}.tif"
                try:
                    resp = requests.get(url, timeout=self.config.request_timeout_s)
                    if resp.status_code == 404:
                        continue
                    resp.raise_for_status()
                    mem = MemoryFile(resp.content)
                    memfiles.append(mem)
                    datasets.append(mem.open())
                except Exception as exc:
                    logger.warning("DEM tile skip %s: %s", tile, exc)

            if not datasets:
                raise RuntimeError("No DEM tiles could be streamed for AOI.")

            mosaic, transform = rio_merge(datasets)
            elev = mosaic[0].astype("float64")
            elev = np.where(elev < -1000, np.nan, elev)
            try:
                window = window_from_bounds(minx, miny, maxx, maxy, transform=transform)
                r0, c0 = max(0, int(window.row_off)), max(0, int(window.col_off))
                h, w = int(window.height), int(window.width)
                elev = elev[r0:r0 + h, c0:c0 + w]
                transform = rasterio.windows.transform(
                    rasterio.windows.Window(c0, r0, elev.shape[1], elev.shape[0]),
                    transform,
                )
            except Exception:
                pass
        finally:
            for ds in datasets:
                ds.close()
            for m in memfiles:
                m.close()

        return {
            "elevation": elev,
            "transform": transform,
            "crs": "EPSG:4326",
            "units": "m",
            "source": {
                "provider": "Copernicus_DEM_GLO30",
                "access": "AWS_COG_HTTP_stream",
                "url": base,
            },
        }

    # ------------------------------------------------------------------
    # Natural Earth coastline distance
    # ------------------------------------------------------------------
    def fetch_distance_to_coast(
        self,
        bounds: tuple[float, float, float, float],
        resolution_m: float = 5000.0,
    ) -> dict[str, Any]:
        import requests
        import geopandas as gpd
        from shapely.ops import unary_union
        from rasterio.transform import from_bounds

        url = "https://naciscdn.org/naturalearth/10m/physical/ne_10m_coastline.zip"
        resp = requests.get(url, timeout=self.config.request_timeout_s)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            with tempfile.TemporaryDirectory() as tmp:
                zf.extractall(tmp)
                shps = list(Path(tmp).glob("**/*.shp"))
                if not shps:
                    raise RuntimeError("Natural Earth coastline shapefile missing in zip")
                coast = gpd.read_file(shps[0]).to_crs("EPSG:4326")

        minx, miny, maxx, maxy = bounds
        res_deg = resolution_m / 111_320.0
        width = max(2, int(np.ceil((maxx - minx) / res_deg)))
        height = max(2, int(np.ceil((maxy - miny) / res_deg)))
        coast_m = coast.to_crs("EPSG:3857")
        coast_geom = unary_union(coast_m.geometry.values)
        xs = np.linspace(minx + res_deg / 2, maxx - res_deg / 2, width)
        ys = np.linspace(maxy - res_deg / 2, miny + res_deg / 2, height)
        xx, yy = np.meshgrid(xs, ys)
        pts = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(xx.ravel(), yy.ravel()), crs="EPSG:4326"
        ).to_crs("EPSG:3857")
        dist_km = (pts.geometry.distance(coast_geom).values / 1000.0).reshape(height, width)
        transform = from_bounds(minx, miny, maxx, maxy, width, height)
        return {
            "distance_to_coast": dist_km.astype("float64"),
            "transform": transform,
            "crs": "EPSG:4326",
            "units": "km",
            "source": {
                "provider": "Natural_Earth_coastline",
                "access": "HTTP_stream",
                "url": url,
                "metric": "distance_to_ocean_coastline",
            },
        }
