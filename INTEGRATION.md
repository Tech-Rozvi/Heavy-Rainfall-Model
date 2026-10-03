# API integration (post-refactor)

## Layout

```
api/app.py
src/data/manager.py
src/data/streaming/remote_client.py
main.py          # run_pipeline shared with API
```

`rainfall_api_fresh/` is obsolete; use root `api/` instead.

## Commands

```bash
export PYTHONPATH=.
export CDS_API_KEY=uid:key   # for ERA5-Land

# Local real files
python main.py --mode local

# Remote stream
python main.py --mode api --aoi aoi.geojson \
  --start-date 2018-01-01 --end-date 2019-12-31

# API server
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Synthetic data requires **both** `--allow-synthetic` and `--generate-synthetic-data`.
