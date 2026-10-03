#!/usr/bin/env python3
"""
main.py — Heavy Rainfall Susceptibility Model (API-first).

Default mode is **api** (remote stream). Synthetic data is off unless
--allow-synthetic is passed.

Examples
--------
  python main.py --country Zimbabwe --adm1 Harare \\
      --start-date 2018-01-01 --end-date 2019-12-31

  python main.py --aoi path/to/aoi.geojson \\
      --start-date 2018-01-01 --end-date 2019-12-31

  python main.py --mode local   # real files under data/raw/ only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.utils import load_config, get_logger, RasterLayer, write_geotiff
from src import rainfall_indices as ri
from src import classification as clf
from src import correlation as corr
from src import weighting as wt
from src import model as mdl
from src import uncertainty as unc
from src import validation as val
from src import visualization as viz
from src.data.manager import (
    DataAcquisitionManager,
    AcquisitionResult,
    bounds_from_aoi_path,
    resolve_gaul_bounds,
)

logger = get_logger("main")


def run_pipeline(
    cfg: dict,
    data: AcquisitionResult,
    *,
    write_outputs: bool = True,
) -> dict[str, Any]:
    """Shared science path for CLI and API (no data acquisition here)."""
    paths = cfg["paths"]
    if write_outputs:
        for p in paths.values():
            Path(p).mkdir(parents=True, exist_ok=True)

    seed = cfg["project"]["random_seed"]
    np.random.seed(seed)

    daily_precip = data.daily_precip
    n_years = data.n_years
    dem = data.dem
    slope = data.slope
    aspect_exposure = data.aspect_exposure
    relative_humidity = data.relative_humidity
    temperature = data.temperature
    wind_speed = data.wind_speed
    distance_to_coast = data.distance_to_coast
    transform = data.transform
    crs = data.crs

    logger.info("STEP 1 — Rainfall indices (mode=%s)", data.mode)
    indices = ri.compute_all_indices(
        daily_precip,
        n_years=n_years,
        extreme_percentile=cfg["heavy_rain_definition"]["percentile"],
    )
    spatial = dem.shape
    for k, v in list(indices.items()):
        if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape != spatial:
            if v.shape == (1, 1):
                indices[k] = np.full(spatial, float(v[0, 0]))
            else:
                from src.data.manager import _fit_to_shape
                indices[k] = _fit_to_shape(v, spatial)

    logger.info("STEP 2 — Factor assembly")
    raw_factors = {
        "mean_annual_precip": indices["mean_annual_precip"],
        "rx1day": indices["rx1day"],
        "rx5day": indices["rx5day"],
        "r95p": indices["r95p"],
        "relative_humidity": relative_humidity,
        "temperature": temperature,
        "wind_speed": wind_speed,
        "distance_to_coast": distance_to_coast,
        "elevation": dem,
        "slope": slope,
        "aspect_exposure": aspect_exposure,
    }

    fcfg = cfg["factors"]
    all_factor_cfg = {**fcfg["atmospheric_factors"], **fcfg["terrain_factors"]}
    enabled_factors = {k: v for k, v in all_factor_cfg.items() if v.get("enabled")}
    for name in fcfg.get("excluded_flood_factors", []):
        if name in enabled_factors:
            raise RuntimeError(f"Flood factor '{name}' must not be scored here.")

    logger.info("STEP 3 — Standardization")
    standardized = {}
    for name, fconf in enabled_factors.items():
        if name not in raw_factors:
            logger.warning("Factor %s missing; skip", name)
            continue
        arr = raw_factors[name]
        if fconf["standardization"] == "fuzzy_decay":
            standardized[name] = clf.standardize(
                arr, "fuzzy_decay", decay_km=fconf.get("decay_km", 100)
            )
        else:
            standardized[name] = clf.standardize(arr, fconf["standardization"])

    logger.info("STEP 4 — Correlation / VIF")
    diag = corr.run_full_diagnostics(
        standardized,
        pearson_thresh=cfg["correlation"]["pearson_flag_threshold"],
        vif_thresh=cfg["correlation"]["vif_flag_threshold"],
    )
    if write_outputs and diag.get("pearson") is not None:
        out_stat = Path(paths["output_stat_dir"])
        out_stat.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(diag["pearson"]).to_csv(out_stat / "pearson_correlation.csv")
        if diag.get("vif") is not None:
            pd.DataFrame(diag["vif"]).to_csv(out_stat / "vif.csv", index=False)

    logger.info("STEP 5 — Weighting")
    order = [f for f in cfg["weighting"]["ahp_factor_order"] if f in standardized]
    pairwise = wt.build_default_pairwise_matrix(order)
    ahp_result = wt.ahp_weights(
        pairwise, order, cfg["weighting"]["ahp_max_acceptable_cr"]
    )
    rf_result = None
    method = cfg["weighting"].get("method", "ahp")
    if method in ("random_forest", "average_of_both", "rf"):
        y_proxy = clf.percentile_rank_standardize(indices["mean_annual_precip"]).ravel()
        X_df = pd.DataFrame({f: standardized[f].ravel() for f in order})
        mask = np.isfinite(y_proxy) & np.all(np.isfinite(X_df.values), axis=1)
        if mask.sum() > 50:
            rf_result = wt.random_forest_weights(
                X_df.loc[mask], y_proxy[mask], random_seed=seed
            )
    final_weights = wt.resolve_final_weights(
        ahp_result, rf_result, method=method if method != "rf" else "random_forest"
    )

    logger.info("STEP 6 — Susceptibility score")
    model = mdl.HeavyRainSusceptibilityModel(weights=final_weights)
    score = model.compute({k: standardized[k] for k in final_weights})
    classes, legend = model.classify_result(
        score,
        cfg["classification"]["n_classes"],
        cfg["classification"]["class_labels"],
    )

    logger.info("STEP 7 — Uncertainty")
    mc = unc.monte_carlo_uncertainty(
        final_weights,
        {k: standardized[k] for k in final_weights},
        n_iterations=cfg["uncertainty"]["monte_carlo_iterations"],
        noise_sigma=float(cfg["uncertainty"].get("input_noise_sigma", 0.05)),
        random_seed=seed,
    )

    report: dict[str, Any] = {}
    logger.info("STEP 8 — Validation")
    station_path = cfg.get("validation", {}).get("station_data_path")
    if station_path and Path(station_path).exists():
        stations = pd.read_csv(station_path)
        try:
            report = val.validate_against_stations(score, transform, stations)
        except Exception as exc:
            logger.warning("Validation skipped: %s", exc)
            report = {"error": str(exc)}
        if write_outputs and report:
            Path(paths["output_report_dir"]).mkdir(parents=True, exist_ok=True)
            with open(Path(paths["output_report_dir"]) / "validation_report.json", "w") as f:
                json.dump(report, f, indent=2, default=str)

    if write_outputs:
        logger.info("STEP 9 — GeoTIFFs")
        out_raster_dir = Path(paths["output_raster_dir"])
        out_raster_dir.mkdir(parents=True, exist_ok=True)
        nodata = cfg.get("nodata_value", -9999.0)

        def _write(name, arr, desc):
            layer = RasterLayer(
                data=arr, transform=transform, crs=crs, nodata=nodata, name=name
            )
            write_geotiff(layer, out_raster_dir / f"{name}.tif", description=desc)

        _write(
            "rainfall_susceptibility_score",
            score,
            "Heavy Rainfall SUSCEPTIBILITY (relative 0-1). NOT a forecast.",
        )
        _write("rainfall_susceptibility_classified", classes, f"Classes: {legend}")
        _write("mean_annual_precip_baseline", indices["mean_annual_precip"], "MAP mm/yr")
        _write("rx1day", indices["rx1day"], "ETCCDI Rx1day mm")
        _write("r95p", indices["r95p"], "ETCCDI R95p mm")
        _write("uncertainty_std", mc["std_score"], "MC std")

        fig_dir = Path(paths["output_fig_dir"])
        fig_dir.mkdir(parents=True, exist_ok=True)
        try:
            viz.plot_raster_map(
                score,
                "Heavy Rainfall Susceptibility Score",
                fig_dir / "susceptibility_map.png",
                cmap="YlGnBu",
                label="Susceptibility (0-1)",
                vmin=0,
                vmax=1,
            )
        except Exception as exc:
            logger.warning("Figure generation failed: %s", exp if False else exc)

        report_dir = Path(paths.get("output_report_dir", "outputs/reports"))
        report_dir.mkdir(parents=True, exist_ok=True)
        with open(report_dir / "data_sources.json", "w") as f:
            json.dump(
                {
                    "mode": data.mode,
                    "sources": data.sources,
                    "resolution_m": data.resolution_m,
                    "n_years": data.n_years,
                    "synthetic": False,
                },
                f,
                indent=2,
                default=str,
            )

    logger.info("Pipeline complete (mode=%s, synthetic=false)", data.mode)
    return {
        "score": score,
        "classes": classes,
        "legend": legend,
        "final_weights": final_weights,
        "ahp_result": ahp_result,
        "rf_result": rf_result,
        "diagnostics": diag,
        "monte_carlo": mc,
        "validation": report,
        "mode": data.mode,
        "sources": data.sources,
    }


def run(
    config_path: str,
    *,
    mode: Optional[str] = None,
    aoi: Optional[str] = None,
    country: Optional[str] = None,
    adm1: Optional[str] = None,
    adm2: Optional[str] = None,
    adm3: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    allow_synthetic: bool = False,
    generate_synthetic_data: bool = False,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    paths = cfg["paths"]
    for p in paths.values():
        Path(p).mkdir(parents=True, exist_ok=True)

    mode = (mode or cfg.get("data_acquisition", {}).get("mode", "api")).lower()
    sci = cfg.get("scientific_options", {})
    allow_synth = bool(
        allow_synthetic
        or generate_synthetic_data
        or sci.get("allow_synthetic_data", False)
    )

    # Synthetic only when explicitly allowed
    if generate_synthetic_data:
        if not allow_synth:
            raise RuntimeError(
                "Refusing synthetic data. Pass --allow-synthetic together with "
                "--generate-synthetic-data if you really need a demo grid."
            )
        logger.warning("SYNTHETIC DATA — explicit override only")
        from src.generate_synthetic_data import generate as generate_synthetic

        generate_synthetic(
            out_dir=Path(paths["raw_dir"]).parent,
            seed=cfg["project"]["random_seed"],
        )
        mode = "local"

    manager = DataAcquisitionManager(cfg)
    bounds = None
    aoi_meta: dict[str, Any] = {}

    if aoi:
        bounds = bounds_from_aoi_path(aoi)
        aoi_meta = {"source": "aoi_file", "path": aoi}
        logger.info("AOI from file %s → %s", aoi, bounds)
    elif country:
        gaul_cfg = cfg.get("data_acquisition", {}).get("gaul", {})
        bounds, aoi_meta = resolve_gaul_bounds(
            country=country,
            adm1=adm1,
            adm2=adm2,
            adm3=adm3,
            gaul_data_dir=gaul_cfg.get("data_dir", "data/boundaries/gaul"),
            gaul_layer=gaul_cfg.get("layer"),
        )
        logger.info("AOI from admin hierarchy %s → %s", aoi_meta, bounds)

    if mode in ("api", "auto", "stream", "remote"):
        if bounds is None:
            raise ValueError(
                "API mode requires --aoi <geojson> or --country [--adm1 --adm2 --adm3]. "
                "Synthetic fallback is disabled."
            )
        data = manager.acquire(
            mode="api",
            aoi_bounds=bounds,
            start_date=start_date,
            end_date=end_date,
        )
        data.sources["aoi"] = aoi_meta
    else:
        # local: require real files; do not invent
        precip = Path(paths["raw_dir"]) / "daily_precip.npy"
        if not precip.exists() and not allow_synth:
            raise FileNotFoundError(
                f"Missing {precip}. Use --mode api with --country/--aoi, "
                "or supply real local rasters. Synthetic is disabled by default."
            )
        data = manager.acquire(mode="local")

    return run_pipeline(cfg, data, write_outputs=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Heavy Rainfall Susceptibility Model (API streaming by default)"
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--mode",
        choices=["api", "local", "auto"],
        default=None,
        help="Default from config (api). local = real files under data/raw/",
    )
    parser.add_argument("--aoi", default=None, help="AOI GeoJSON / Shapefile")
    parser.add_argument("--country", default=None, help="GAUL / geocoder country (ADM0)")
    parser.add_argument("--adm1", default=None, help="County / province (ADM1)")
    parser.add_argument("--adm2", default=None, help="Sub-county / district (ADM2)")
    parser.add_argument("--adm3", default=None, help="Ward (ADM3) when available")
    parser.add_argument("--start-date", default=None, help="YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD")
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Permit synthetic demo data (OFF by default)",
    )
    parser.add_argument(
        "--generate-synthetic-data",
        action="store_true",
        help="Write synthetic demo grids (requires --allow-synthetic)",
    )
    args = parser.parse_args()

    run(
        args.config,
        mode=args.mode,
        aoi=args.aoi,
        country=args.country,
        adm1=args.adm1,
        adm2=args.adm2,
        adm3=args.adm3,
        start_date=args.start_date,
        end_date=args.end_date,
        allow_synthetic=args.allow_synthetic,
        generate_synthetic_data=args.generate_synthetic_data,
    )
