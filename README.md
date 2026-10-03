# Heavy Rainfall Susceptibility Model

A reproducible, configuration-driven Python pipeline for estimating **relative
heavy-rainfall susceptibility** from atmospheric and orographic factors, with
transparent weighting (AHP + data-driven), collinearity screening, validation
against observed rainfall, and formal uncertainty quantification.

> **This is a susceptibility/relative-risk model, not a rainfall forecast.**
> See [Susceptibility vs. Prediction](#susceptibility-vs-prediction) and
> [Rainfall vs. Flood Risk](#rainfall-susceptibility-vs-flood-hazard) below —
> these distinctions are enforced in the code, not just in this document.

---

## Table of contents
1. [Installation](#installation)
2. [Data requirements](#data-requirements)
3. [Folder structure](#folder-structure)
4. [Configuration](#configuration)
5. [Running the model](#running-the-model)
6. [Outputs](#outputs)
7. [Validation](#validation)
8. [Susceptibility vs. Prediction](#susceptibility-vs-prediction)
9. [Rainfall susceptibility vs. flood hazard](#rainfall-susceptibility-vs-flood-hazard)
10. [Handling collinearity](#handling-collinearity)
11. [Resolution guidance](#resolution-guidance)
12. [Performance notes](#performance-notes)
13. [Troubleshooting](#troubleshooting)

---

## Installation

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`shap` and `dask` are optional (listed in requirements.txt with a comment) —
the pipeline degrades gracefully without them (SHAP importance is skipped
with a warning; Dask/chunked processing is only needed for national-scale
NetCDF stacks that don't fit in memory).

## Data requirements

The pipeline needs, at minimum, rasters aligned to a common grid (the
`preprocessing.py` module handles reprojection/resampling for you if your
sources differ):

| Layer | Example source | Notes |
|---|---|---|
| Daily precipitation stack (time, y, x) | CHIRPS, ERA5-Land, GPM IMERG, TAMSAT, station-gridded product | Used to derive the historical baseline + ETCCDI extreme indices |
| DEM | Copernicus DEM, SRTM | Used for elevation/slope/aspect |
| Relative humidity, temperature, wind speed | ERA5-Land, WorldClim (climatology) | |
| Distance to coast/water bodies | Derived from a coastline/waterbody vector (e.g. Natural Earth) | |
| Study-area boundary | A shapefile/GeoJSON polygon | The ONLY file you change to retarget the model to a new region |
| Station/validation observations | National met service, GHCN | CSV: `station_id,x,y,source,observed_value,observed_event` |

No real data? Run with `--generate-synthetic-data` — see
[Running the model](#running-the-model). The synthetic generator
(`src/generate_synthetic_data.py`) creates a small, physically-plausible
example (a coastal-to-inland gradient with a synthetic mountain range) so you
can exercise the entire pipeline immediately. **It is not real observational
data**, and every file it produces is tagged as such in its metadata.

## Folder structure

```
rainfall_model/
├── config/config.yaml        # ALL parameters — no hardcoding in src/
├── data/{raw,processed,validation}/
├── src/
│   ├── utils.py               # config/logging/raster I/O/QC
│   ├── preprocessing.py       # reproject/resample/align/clip
│   ├── rainfall_indices.py    # Rx1day, Rx5day, R95p, R99p, R10mm, R20mm, CDD, CWD
│   ├── terrain.py             # slope/aspect/windward exposure/distance
│   ├── classification.py      # standardization + classification methods
│   ├── correlation.py         # Pearson/Spearman/VIF diagnostics
│   ├── weighting.py           # AHP + Random Forest weighting
│   ├── model.py                # weighted-overlay susceptibility + ML prediction model
│   ├── validation.py          # continuous + classification metrics
│   ├── uncertainty.py         # weight sensitivity + Monte Carlo
│   ├── visualization.py       # maps and charts
│   └── generate_synthetic_data.py
├── outputs/{rasters,statistics,figures,reports}/
├── tests/                      # pytest unit tests
├── requirements.txt
├── main.py
└── README.md
```

## Configuration

Everything is driven by `config/config.yaml`: study area path, CRS, target
resolution, date range, heavy-rain threshold definition, which factors are
enabled and how each is standardized/classified, the AHP pairwise matrix
order, correlation/VIF thresholds, uncertainty settings, and all I/O paths.
**No dates, weights, thresholds, or paths are hardcoded in `src/*.py`.**

To retarget the model to a new country/region, you only need to:
1. Point `study_area.boundary_path` at the new boundary file.
2. Point the raw data paths at datasets covering that region.
3. Review the AHP pairwise matrix (`weighting.ahp_factor_order` + the default
   matrix built by `weighting.build_default_pairwise_matrix`) with a local
   domain expert — do not assume the default judgments transfer to a
   different climate regime without review.

## Running the model

```bash
# First run (generates a synthetic example dataset automatically):
python main.py --config config/config.yaml --generate-synthetic-data

# Subsequent runs against your own data in data/raw/:
python main.py --config config/config.yaml
```

Run the tests:
```bash
pytest tests/ -v
```

## Outputs

`outputs/rasters/` (GeoTIFF, CRS/transform/nodata set, QGIS/ArcGIS-ready):
- `rainfall_susceptibility_score.tif` — continuous 0–1 relative susceptibility
- `rainfall_susceptibility_classified.tif` — 5-class (Very Low → Very High)
- `mean_annual_precip_baseline.tif`, `rx1day.tif`, `r95p.tif` — historical baseline/extremes
- `uncertainty_std.tif`, `uncertainty_cv.tif` — Monte Carlo uncertainty layers

`outputs/statistics/`: Pearson correlation matrix, VIF table, AHP-vs-RF
weight comparison, weight-sensitivity table.

`outputs/figures/`: susceptibility map, classified map, uncertainty map,
correlation heatmap, AHP/RF weight bar charts, observed-vs-predicted
scatterplots per validation source, sensitivity bar chart.

`outputs/reports/validation_report.json`: full per-source validation metrics.

## Validation

Station/CHIRPS/ERA5-Land observations are read from
`config.validation.station_data_path` and are **never pooled together** —
metrics are reported per source, because ground stations, satellite, and
reanalysis products have different, non-interchangeable error
characteristics. Continuous metrics (Pearson, Spearman, RMSE, MAE, R²,
bias) apply to the continuous susceptibility score / rainfall amount.
Classification metrics (ROC-AUC, precision, recall, F1) apply only when a
binary observed heavy-rain-event label is available, using the threshold
definition set in `config.heavy_rain_definition`.

## Susceptibility vs. Prediction

Two distinct model classes exist in `src/model.py`, and their outputs are
never mixed in filenames or reports:

- **`HeavyRainSusceptibilityModel`** — the transparent weighted-overlay
  `Σ(Wi·Xi)` combination. It answers "where, relative to the rest of the
  study area, is heavy rainfall more likely to matter" — it does NOT output
  mm of rain.
- **`HeavyRainPredictionModel`** — an optional supervised ML model (Random
  Forest / Gradient Boosting / regularized or logistic regression) that
  genuinely predicts a rainfall quantity or event probability, trained
  against real observed targets with **spatial cross-validation**
  (`model.spatial_cv_folds`) to avoid leakage from spatially autocorrelated
  neighbouring pixels (see the docstring in `model.py` for why naive random
  pixel splitting is invalid here).

Only use the word "prediction" for output that came from the second class.

## Rainfall susceptibility vs. flood hazard

The model intentionally **excludes** TWI, land use/land cover, soil
moisture, NDVI, and watershed/drainage metrics from the rainfall
susceptibility score (`config.factors.excluded_flood_factors`, enforced in
`main.py`). These variables describe what happens to water *after* it falls
(runoff, infiltration, drainage) — including them would silently turn a
rainfall model into a flood-hazard model. If you need a flood/runoff impact
layer, build it as a **separate module** that consumes this model's
susceptibility raster as one input among others (elevation, TWI, land
cover) — do not fold it back into this score.

Elevation, slope, and aspect ARE kept, but only as **orographic modifiers of
rainfall generation** (forced ascent on windward slopes), which is a
legitimate atmospheric mechanism, not a runoff mechanism.

## Handling collinearity

`correlation.run_full_diagnostics` runs Pearson, Spearman, and VIF on the
standardized factor layers before weighting. Flagged pairs (|r| ≥ 0.70 by
default) or high-VIF factors (>5 by default) are **not auto-dropped** — the
choice of which correlated variable to keep (or whether to combine them via
PCA) is a scientific judgment that depends on which variable the domain
expert trusts more for the study region, so the pipeline logs a warning and
writes `outputs/statistics/pearson_correlation.csv` / `vif.csv` for review.
In the bundled synthetic example, the atmospheric factors are deliberately
correlated by construction (coastal proximity drives humidity, temperature,
and rainfall together) — this is realistic and is exactly the situation the
diagnostic is meant to surface for review, not silently combine.

## Resolution guidance

`study_area.target_resolution_m` should not exceed the coarsest scientifically
meaningful input. If your rainfall product is ERA5-Land (~9 km) but your DEM
is 30 m, running the whole model at 30 m does not create 30 m rainfall
information — it just resamples the coarse product, implying false
precision. Choose the target resolution to match your **coarsest genuinely
informative** driver (usually the rainfall/atmospheric layer), and document
that choice in your output metadata.

## Performance notes

- Large national NetCDF precipitation stacks should be opened with
  `xarray`/`dask` in chunked mode rather than loaded fully into RAM;
  `rainfall_indices.py`'s vectorised rolling-sum and consecutive-run
  functions are written to work equally well over dask-backed arrays.
- Raster reprojection uses `rasterio`'s windowed reprojection rather than
  reading whole files where the source supports it.
- Monte Carlo uncertainty iterations (`uncertainty.monte_carlo_uncertainty`)
  are the most expensive step at country scale — reduce
  `uncertainty.monte_carlo_iterations` for a quick test run, increase it for
  a final production run.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `QCError: Raster misalignment detected` | Inputs weren't run through `preprocessing.reproject_resample_align` first — every input layer must share one grid before classification/weighting. |
| AHP `is_consistent: False` (CR > 0.10) | Your pairwise comparison matrix has contradictory judgments (e.g. A>B, B>C, but C>A). Revise `weighting.build_default_pairwise_matrix` or supply your own matrix. |
| `ValueError: Weights must sum to 1.0` | `weighting.resolve_final_weights` output wasn't renormalized after a manual edit — always renormalize after changing any individual weight. |
| Validation raises `Only N valid paired observations` | Fewer than 3 stations overlap the raster's valid extent for that source — check station coordinates are in the same CRS as the raster. |
| `ImportError: shap not installed` (warning, not a crash) | SHAP is optional; `pip install shap` if you want it, otherwise ignore. |

---

### Citing methodology sources
Extreme-precipitation index definitions follow the ETCCDI standard set
(Rx1day, Rx5day, R95p, R99p, R10mm, R20mm, CDD, CWD). AHP methodology
follows Saaty (1980), including the standard Random Index table used for the
Consistency Ratio calculation.
