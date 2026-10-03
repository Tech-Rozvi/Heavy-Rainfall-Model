"""
Rainfall Model — Remote Data API

Streams real CHIRPS / ERA5-Land / Copernicus DEM / Natural Earth data.
No synthetic data. CLI and API share run_pipeline().

Start:
    uvicorn api.app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import os
import traceback
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="Rainfall Susceptibility — Remote Data API",
    description=(
        "On-demand CHIRPS, ERA5-Land, Copernicus DEM, Natural Earth. "
        "No synthetic data. No permanent local climate archives required."
    ),
    version="2.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class BBox(BaseModel):
    minx: float
    miny: float
    maxx: float
    maxy: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.minx, self.miny, self.maxx, self.maxy)


class Period(BaseModel):
    start_date: str = Field(..., examples=["2015-01-01"])
    end_date: str = Field(..., examples=["2020-12-31"])


class DataRequestBody(BaseModel):
    bbox: BBox
    period: Period
    target_resolution_m: float = 5000.0


class ModelRunBody(DataRequestBody):
    config_path: str = "config/config.yaml"
    write_outputs: bool = True


def _client():
    from src.data.streaming.remote_client import RemoteDataClient, StreamConfig
    return RemoteDataClient(StreamConfig(allow_ephemeral_temp=True))


def _arr_summary(arr: np.ndarray) -> dict:
    finite = arr[np.isfinite(arr)]
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(finite.min()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "mean": float(finite.mean()) if finite.size else None,
        "missing_frac": float(1.0 - finite.size / arr.size) if arr.size else 1.0,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "synthetic_data": False,
        "access_mode": "remote_stream_api",
        "cds_key_configured": bool(
            os.environ.get("CDS_API_KEY") or os.environ.get("CDSAPI_KEY")
        ),
    }


@app.get("/sources")
def sources():
    return {
        "precipitation": {
            "primary": "CHIRPS v2",
            "access": ["ClimateSERV_API", "HTTP_GeoTIFF_windowed_stream"],
            "auth": False,
            "doi": "10.1038/sdata.2015.66",
        },
        "atmospheric": {
            "primary": "ERA5-Land ARCO Zarr",
            "access": ["ARCO_Zarr_remote"],
            "auth": "CDS_API_KEY recommended",
            "doi": "10.24381/cds.e2161bac",
        },
        "dem": {
            "primary": "Copernicus DEM GLO-30",
            "access": ["AWS_COG_HTTP_stream"],
            "auth": False,
        },
        "coastline": {
            "primary": "Natural Earth 10m coastline",
            "access": ["HTTP_stream"],
            "auth": False,
        },
        "policy": {
            "synthetic_data": "disabled",
            "permanent_local_download_required": False,
        },
    }


@app.post("/v1/data/precipitation")
def data_precipitation(body: DataRequestBody):
    try:
        result = _client().fetch_chirps_daily(
            body.bbox.as_tuple(), body.period.start_date, body.period.end_date
        )
        return {
            "variable": "precipitation",
            "units": result["units"],
            "source": result["source"],
            "bounds": list(body.bbox.as_tuple()),
            "period": body.period.model_dump(),
            "summary": _arr_summary(result["precipitation"]),
            "n_days": int(result["precipitation"].shape[0]),
            "synthetic": False,
        }
    except Exception as exc:
        raise HTTPException(502, f"precipitation stream failed: {exc}") from exc


@app.post("/v1/data/atmospheric")
def data_atmospheric(body: DataRequestBody):
    try:
        result = _client().fetch_era5_land_climatology(
            body.bbox.as_tuple(), body.period.start_date, body.period.end_date
        )
        return {
            "variables": ["temperature", "relative_humidity", "wind_speed"],
            "units": result["units"],
            "source": result["source"],
            "summaries": {
                k: _arr_summary(result[k])
                for k in ("temperature", "relative_humidity", "wind_speed")
            },
            "synthetic": False,
        }
    except Exception as exc:
        raise HTTPException(502, f"atmospheric stream failed: {exc}") from exc


@app.post("/v1/data/dem")
def data_dem(body: DataRequestBody):
    try:
        result = _client().fetch_dem(body.bbox.as_tuple())
        return {
            "variable": "elevation",
            "units": result["units"],
            "source": result["source"],
            "summary": _arr_summary(result["elevation"]),
            "synthetic": False,
        }
    except Exception as exc:
        raise HTTPException(502, f"DEM stream failed: {exc}") from exc


@app.post("/v1/data/coast")
def data_coast(body: DataRequestBody):
    try:
        result = _client().fetch_distance_to_coast(
            body.bbox.as_tuple(), body.target_resolution_m
        )
        return {
            "variable": "distance_to_coast",
            "units": result["units"],
            "source": result["source"],
            "summary": _arr_summary(result["distance_to_coast"]),
            "synthetic": False,
        }
    except Exception as exc:
        raise HTTPException(502, f"coast stream failed: {exc}") from exc


@app.post("/v1/data/bundle")
def data_bundle(body: DataRequestBody):
    try:
        from src.utils import load_config
        from src.data.manager import DataAcquisitionManager

        cfg = load_config("config/config.yaml")
        mgr = DataAcquisitionManager(cfg)
        acq = mgr.acquire_remote(
            body.bbox.as_tuple(),
            body.period.start_date,
            body.period.end_date,
            resolution_m=max(body.target_resolution_m, 5000.0),
        )
        return {
            "synthetic_data": False,
            "access_mode": "remote_stream",
            "bounds": list(body.bbox.as_tuple()),
            "period": body.period.model_dump(),
            "resolution_m": acq.resolution_m,
            "sources": acq.sources,
            "summaries": {
                "precipitation": _arr_summary(acq.daily_precip),
                "dem": _arr_summary(acq.dem),
                "temperature": _arr_summary(acq.temperature),
                "relative_humidity": _arr_summary(acq.relative_humidity),
                "wind_speed": _arr_summary(acq.wind_speed),
                "distance_to_coast": _arr_summary(acq.distance_to_coast),
            },
        }
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(502, f"bundle failed: {exc}") from exc


@app.post("/v1/model/run")
def model_run(body: ModelRunBody):
    """Stream remote data and run the SAME pipeline as main.py."""
    try:
        from src.utils import load_config
        from src.data.manager import DataAcquisitionManager
        from main import run_pipeline

        cfg = load_config(body.config_path)
        mgr = DataAcquisitionManager(cfg)
        acq = mgr.acquire_remote(
            body.bbox.as_tuple(),
            body.period.start_date,
            body.period.end_date,
            resolution_m=max(body.target_resolution_m, 5000.0),
        )
        result = run_pipeline(cfg, acq, write_outputs=body.write_outputs)

        def _safe_weights(w):
            if isinstance(w, dict):
                return {k: float(v) for k, v in w.items()}
            return w

        return {
            "synthetic_data": False,
            "access_mode": "remote_stream",
            "bounds": list(body.bbox.as_tuple()),
            "period": body.period.model_dump(),
            "mode": result.get("mode"),
            "sources": result.get("sources"),
            "weights": _safe_weights(result.get("final_weights")),
            "score_summary": _arr_summary(result["score"]),
            "class_legend": result.get("legend"),
            "uncertainty_std_summary": _arr_summary(result["monte_carlo"]["std_score"]),
            "validation": result.get("validation"),
        }
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(500, f"model run failed: {exc}") from exc
