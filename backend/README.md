# Backend Quickstart

## 1) Start PostgreSQL

```bash
docker compose -f backend/docker-compose.yml up -d
```

## 2) Configure environment

Copy `.env.example` to `.env` and set `OPENROUTER_API_KEY`.

## 3) Load CSV into PostgreSQL

```bash
python load_taxi_to_postgres.py
```

Optional import behavior:

```bash
# fail_if_exists (default, safest) | truncate | append
$env:IMPORT_MODE="truncate"
python load_taxi_to_postgres.py
```

Optional CSV override:

```bash
$env:TAXI_CSV_PATH="dataset/taxi_trip_data.csv"
python load_taxi_to_postgres.py
```

## 4) Run the multi-agent dashboard

```bash
python main.py
```

Smoke test:

```bash
python scripts/smoke_run.py
# or shortcut
python smoke_run.py
```
