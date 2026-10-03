"""
generate_synthetic_data.py
Creates a small, physically-plausible synthetic example dataset (per project
requirement G) so the full pipeline can be exercised end-to-end without
downloading CHIRPS/ERA5/GPM/DEM data. NOT real observational data -- every
output file and log message says so explicitly.

Grid: 60 x 60 pixels at 5000 m resolution (~300 km x 300 km), loosely
mimicking a coastal-to-inland gradient so the factors behave realistically:
rainfall/humidity decay inland, elevation rises inland, a synthetic
mountain range creates an orographic (windward/leeward) rainfall contrast.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .utils import RasterLayer, write_geotiff, get_logger

logger = get_logger(__name__)


def generate(out_dir: str | Path, seed: int = 42, size: int = 60,
             resolution_m: float = 5000.0, n_years: int = 3) -> dict:
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    (out_dir / "validation").mkdir(parents=True, exist_ok=True)

    from rasterio.transform import Affine
    origin_x, origin_y = 36.0, 1.0  # arbitrary lon/lat-like origin for realism
    transform = Affine(resolution_m, 0, origin_x, 0, -resolution_m, origin_y)
    crs = "EPSG:32637"  # placeholder projected CRS (UTM 37N) treated as metric grid

    yy, xx = np.mgrid[0:size, 0:size]

    # --- Coastline along the western edge (x=0), moisture source ---
    distance_to_coast = xx.astype(float) * resolution_m
    distance_to_coast += rng.normal(0, resolution_m * 0.5, size=(size, size))
    distance_to_coast = np.clip(distance_to_coast, 0, None)

    # --- Elevation: rises inland, with a synthetic mountain ridge around x=35 ---
    base_elev = (xx / size) * 1800.0
    ridge = 900.0 * np.exp(-((xx - 38) ** 2) / (2 * 6 ** 2))
    dem = base_elev + ridge + rng.normal(0, 40, size=(size, size))
    dem = np.clip(dem, 0, None)

    # --- Slope/aspect derived from DEM ---
    from .terrain import compute_slope_aspect, windward_exposure
    slope_deg, aspect_deg = compute_slope_aspect(dem, cellsize_m=resolution_m)
    dominant_wind_from_deg = 270.0  # westerly (onshore) wind, from the coast
    exposure = windward_exposure(aspect_deg, dominant_wind_from_deg)

    # --- Humidity: decays inland, boosted slightly on windward slopes ---
    rh = 85 - (xx / size) * 35 + 5 * exposure + rng.normal(0, 3, size=(size, size))
    rh = np.clip(rh, 10, 100)

    # --- Temperature: rises inland/lower elevation (lapse rate ~6.5 C/km) ---
    temperature = 28 - (dem / 1000.0) * 6.5 + rng.normal(0, 1.0, size=(size, size))

    # --- Wind speed: stronger near coast ---
    wind_speed = 8 - (xx / size) * 4 + rng.normal(0, 0.8, size=(size, size))
    wind_speed = np.clip(wind_speed, 0.5, None)

    # --- Daily precipitation stack (time, y, x): base climatology modulated
    #     by humidity, orographic exposure, and coastal proximity, with
    #     realistic gamma-distributed daily rainfall and a wet season pulse ---
    n_days = 365 * n_years
    day_of_year = np.arange(n_days) % 365
    wet_season = np.isin(day_of_year, np.arange(60, 150))  # synthetic wet season

    orographic_boost = 1.0 + 1.2 * exposure  # up to +120% on windward slopes
    coastal_factor = 1.0 + 1.5 * np.exp(-distance_to_coast / (150_000))
    humidity_factor = rh / 100.0

    mean_intensity = 3.0 * orographic_boost * coastal_factor * humidity_factor
    daily_precip = np.zeros((n_days, size, size), dtype="float32")
    for t in range(n_days):
        season_mult = 2.5 if wet_season[t] else 0.6
        shape_param = 0.6  # gamma shape -> skewed, realistic daily rainfall
        scale = (mean_intensity * season_mult) / shape_param
        rain = rng.gamma(shape_param, scale)
        # ~55% dry-day probability outside wet season, ~15% within
        dry_prob = 0.15 if wet_season[t] else 0.55
        dry_mask = rng.random((size, size)) < dry_prob
        rain[dry_mask] = 0.0
        daily_precip[t] = rain

    # --- Write factor GeoTIFFs ---
    def _write(name, arr):
        layer = RasterLayer(data=arr.astype("float64"), transform=transform, crs=crs,
                             name=name)
        write_geotiff(layer, out_dir / "raw" / f"{name}.tif",
                       description=f"SYNTHETIC EXAMPLE DATA — {name} — not observational")
        return layer

    _write("dem", dem)
    _write("slope", slope_deg)
    _write("aspect_exposure", exposure)
    _write("relative_humidity", rh)
    _write("temperature", temperature)
    _write("wind_speed", wind_speed)
    _write("distance_to_coast", distance_to_coast)

    np.save(out_dir / "raw" / "daily_precip.npy", daily_precip)

    # --- Study area boundary (bounding box of the grid) ---
    minx, maxy = transform * (0, 0)
    maxx, miny = transform * (size, size)
    boundary = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature", "properties": {"name": "synthetic_study_area"},
            "geometry": {"type": "Polygon", "coordinates": [[
                [minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]
            ]]}
        }]
    }
    with open(out_dir / "raw" / "study_area.geojson", "w") as f:
        json.dump(boundary, f)

    # --- Synthetic validation "stations": sample real pixel values + noise,
    #     representing where a station happened to be, with an independent
    #     "observed" heavy-rain-day count that correlates with but is not
    #     identical to the model inputs (simulates real-world imperfection) ---
    n_stations = 25
    station_rows = rng.integers(2, size - 2, n_stations)
    station_cols = rng.integers(2, size - 2, n_stations)
    records = []
    annual_total = daily_precip.sum(axis=0) / n_years
    heavy_days = (daily_precip >= np.nanpercentile(daily_precip, 95)).sum(axis=0)
    for i in range(n_stations):
        r, c = station_rows[i], station_cols[i]
        x, y = transform * (c + 0.5, r + 0.5)
        obs_value = float(annual_total[r, c] * rng.normal(1.0, 0.08))
        obs_event = int(heavy_days[r, c] > np.percentile(heavy_days, 70))
        source = "station" if i % 3 == 0 else ("CHIRPS" if i % 3 == 1 else "ERA5-Land")
        records.append({
            "station_id": f"STN_{i:03d}", "x": x, "y": y, "source": source,
            "observed_value": round(obs_value, 2), "observed_event": obs_event,
        })
    import pandas as pd
    pd.DataFrame(records).to_csv(out_dir / "validation" / "stations.csv", index=False)

    logger.info("Synthetic example dataset generated at %s (THIS IS NOT REAL "
                "OBSERVATIONAL DATA — for pipeline testing only).", out_dir)

    return {
        "transform": transform, "crs": crs, "size": size, "resolution_m": resolution_m,
        "n_years": n_years, "daily_precip_path": out_dir / "raw" / "daily_precip.npy",
    }


if __name__ == "__main__":
    generate(out_dir="data")
