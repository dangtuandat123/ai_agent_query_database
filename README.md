# AI Agent Database Dashboard (Taxi Data)

Simple MVP using:
- PostgreSQL (data source)
- LangGraph (multi-agent flow)
- OpenRouter API via `langchain-openai`
- LangChain Retriever (hybrid: embeddings + BM25) for schema RAG

## What it does
- Receive natural language question (Vietnamese or English)
- Read schema dynamically from PostgreSQL
- Retrieve relevant tables/columns for the current question (schema retrieval)
- Route question
- Generate SQL using retrieved schema context
- Execute SQL on PostgreSQL
- Return concise answer

## Project structure
- `main.py`: entrypoint, optional `--question`, prints workflow + result
- `taxi_agent/config.py`: environment config
- `taxi_agent/types.py`: typed graph state/result
- `taxi_agent/sql_guard.py`: read-only SQL validation
- `taxi_agent/retrieval.py`: LangChain hybrid schema retriever (vector + BM25, prefers EnsembleRetriever when available)
- `taxi_agent/services/schema_service.py`: schema context builder + cache TTL
- `taxi_agent/services/metadata_service.py`: metadata hint builder (table/column/value hints)
- `taxi_agent/services/sql_service.py`: SQL generate/repair/execute orchestration
- `taxi_agent/services/language_service.py`: VN/EN fallback messages
- `taxi_agent/db.py`: PostgreSQL client + LangChain `SQLDatabase` schema context adapter (+ backward-compatible SQL guard exports)
- `taxi_agent/prompts.py`: prompts for router/sql/answer agents
- `taxi_agent/graph.py`: LangGraph facade (`TaxiDashboardAgent`)
- `load_taxi_to_postgres.py`: load CSV into PostgreSQL table
- `scripts/smoke_run.py`: smoke run with VN/EN questions
- `backend/docker-compose.yml`: local PostgreSQL service
- `tests/`: unit tests

## Setup
1) Install dependencies:

```bash
pip install -r requirements.txt
```

2) Set environment:
- Copy `.env.example` -> `.env`
- Fill `OPENROUTER_API_KEY`

3) Start PostgreSQL:

```bash
docker compose -f backend/docker-compose.yml up -d
```

4) Prepare CSV data:
- Large dataset is intentionally not tracked in Git.
- Put CSV at default path `dataset/taxi_trip_data.csv` or set `TAXI_CSV_PATH`.

5) Load CSV data:

```bash
python load_taxi_to_postgres.py
```

6) Run app:

```bash
python main.py
```

Optional question override:

```bash
python main.py --question "For March 2018, top payment type by revenue?"
```

Optional thread id (for follow-up context continuity within same process):

```bash
python main.py --thread-id finance-team
```

CLI help:

```bash
python main.py --help
```

## Tests
Unit tests:

```bash
pytest -q
```

Smoke run (requires live DB + model key):

```bash
python scripts/smoke_run.py
```

Shortcut alias:

```bash
python smoke_run.py
```

## Notes
- SQL guard only allows one read-only query (`SELECT` / `WITH ... SELECT`).
- SQL guard also blocks `SELECT ... INTO` and locking clauses (`FOR UPDATE`, `FOR SHARE`, ...).
- Agent retries SQL generation when query fails (`MAX_SQL_RETRIES`).
- Query timeout is configurable via `QUERY_TIMEOUT_MS` (default: 30000ms).
- DB connect timeout is configurable via `DB_CONNECT_TIMEOUT_SECONDS` (default: 10s).
  Used by both the agent runtime and data loader.
- SQL repair auto-expands from retrieved schema to full schema when needed.
- DB execution pins `search_path` to `DB_SCHEMA` for safer schema isolation.
- Graph now includes an intent step (`sql_query` / `sql_followup`) and a security preflight before query execution.
- Follow-up memory is scoped by `thread_id` (default: `default`).
- Follow-up memory is in-process and bounded (LRU-style by thread) to avoid unbounded growth.
- Data loader supports safe modes with `IMPORT_MODE`: `fail_if_exists`, `truncate`, `append`.
- Optional CSV path override for loader/preview via `TAXI_CSV_PATH`
  (relative paths are resolved from project root).
- Schema introspection uses in-memory cache (`SCHEMA_CACHE_TTL_SECONDS`, default 300s).
  - Set `SCHEMA_CACHE_TTL_SECONDS=0` to disable schema cache.
- For large schemas, tune:
  - `SCHEMA_TOP_K_TABLES`
  - `SCHEMA_MAX_COLUMNS_PER_TABLE`
  - `SCHEMA_CONTEXT_MAX_CHARS`
  - `SCHEMA_FULL_CONTEXT_MAX_CHARS`
  - `ENABLE_SCHEMA_EMBEDDINGS`
  - `OPENROUTER_EMBEDDING_MODEL` (default: `google/gemini-embedding-001`)
  - `SCHEMA_RETRIEVER_SEARCH_TYPE` (`mmr` or `similarity`)
  - `SCHEMA_RETRIEVER_FETCH_K` (used when search type is `mmr`)
  - `LOG_LEVEL`
- If `rank_bm25` is missing in your environment, retriever auto-falls back to vector-only and logs a warning.
- `.env`, dataset files, caches, and generated workflow files are ignored via `.gitignore`.
