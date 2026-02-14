from dataclasses import dataclass
import os
from typing import Optional


@dataclass(frozen=True)
class Settings:
    postgres_dsn: str
    openrouter_api_key: str
    openrouter_model: str
    openrouter_base_url: str
    openrouter_site_url: Optional[str]
    openrouter_app_name: Optional[str]
    query_row_limit: int
    query_timeout_ms: int
    max_sql_retries: int
    db_schema: str
    schema_top_k_tables: int
    schema_max_columns_per_table: int
    schema_context_max_chars: int
    schema_full_context_max_chars: int
    schema_cache_ttl_seconds: int
    enable_schema_embeddings: bool
    openrouter_embedding_model: str
    schema_retriever_search_type: str
    schema_retriever_fetch_k: int
    log_level: str
    db_connect_timeout_seconds: int = 10
    memory_max_threads: int = 200


def _get_int_env(name: str, default: int, min_value: int = 1) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got: {raw!r}") from exc

    if value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got: {value}")
    return value


def _get_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name, str(default)).strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(
        f"{name} must be a boolean value "
        "(accepted: true/false, 1/0, yes/no, on/off), got: "
        f"{os.getenv(name)!r}"
    )


def _get_log_level_env(name: str, default: str = "INFO") -> str:
    allowed_levels = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}
    raw = (os.getenv(name, default) or default).strip().upper()
    if raw in allowed_levels:
        return raw
    return default


def load_settings() -> Settings:
    openrouter_api_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    if not openrouter_api_key:
        raise ValueError(
            "Missing OPENROUTER_API_KEY. Set it in your environment or .env file."
        )

    postgres_dsn = (
        os.getenv(
            "POSTGRES_DSN",
            "postgresql://postgres:postgres@localhost:5432/taxi_db",
        ).strip()
        or "postgresql://postgres:postgres@localhost:5432/taxi_db"
    )
    openrouter_model = (
        os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash").strip()
        or "google/gemini-2.5-flash"
    )
    openrouter_base_url = (
        os.getenv(
            "OPENROUTER_BASE_URL",
            "https://openrouter.ai/api/v1",
        ).strip()
        or "https://openrouter.ai/api/v1"
    )
    openrouter_site_url_raw = (os.getenv("OPENROUTER_SITE_URL") or "").strip()
    openrouter_site_url = openrouter_site_url_raw or None
    openrouter_app_name_raw = (
        os.getenv("OPENROUTER_APP_NAME", "Taxi Agent Dashboard").strip()
    )
    openrouter_app_name = openrouter_app_name_raw or None

    query_row_limit = _get_int_env("QUERY_ROW_LIMIT", 100)
    query_timeout_ms = _get_int_env("QUERY_TIMEOUT_MS", 30000)
    max_sql_retries = _get_int_env("MAX_SQL_RETRIES", 1, min_value=0)
    db_schema = os.getenv("DB_SCHEMA", "public").strip() or "public"
    schema_top_k_tables = _get_int_env("SCHEMA_TOP_K_TABLES", 5)
    schema_max_columns_per_table = _get_int_env("SCHEMA_MAX_COLUMNS_PER_TABLE", 40)
    schema_context_max_chars = _get_int_env("SCHEMA_CONTEXT_MAX_CHARS", 12000)
    schema_full_context_max_chars = _get_int_env(
        "SCHEMA_FULL_CONTEXT_MAX_CHARS", 30000
    )
    schema_cache_ttl_seconds = _get_int_env(
        "SCHEMA_CACHE_TTL_SECONDS",
        300,
        min_value=0,
    )
    enable_schema_embeddings = _get_bool_env("ENABLE_SCHEMA_EMBEDDINGS", True)
    openrouter_embedding_model = (
        os.getenv(
            "OPENROUTER_EMBEDDING_MODEL",
            "google/gemini-embedding-001",
        ).strip()
        or "google/gemini-embedding-001"
    )
    schema_retriever_search_type = (
        os.getenv("SCHEMA_RETRIEVER_SEARCH_TYPE", "mmr").strip().lower()
    )
    if schema_retriever_search_type not in {"similarity", "mmr"}:
        schema_retriever_search_type = "mmr"
    schema_retriever_fetch_k = _get_int_env("SCHEMA_RETRIEVER_FETCH_K", 20)
    log_level = _get_log_level_env("LOG_LEVEL", "INFO")
    db_connect_timeout_seconds = _get_int_env("DB_CONNECT_TIMEOUT_SECONDS", 10)
    memory_max_threads = _get_int_env("MEMORY_MAX_THREADS", 200)

    return Settings(
        postgres_dsn=postgres_dsn,
        openrouter_api_key=openrouter_api_key,
        openrouter_model=openrouter_model,
        openrouter_base_url=openrouter_base_url,
        openrouter_site_url=openrouter_site_url,
        openrouter_app_name=openrouter_app_name,
        query_row_limit=query_row_limit,
        query_timeout_ms=query_timeout_ms,
        max_sql_retries=max_sql_retries,
        db_schema=db_schema,
        schema_top_k_tables=schema_top_k_tables,
        schema_max_columns_per_table=schema_max_columns_per_table,
        schema_context_max_chars=schema_context_max_chars,
        schema_full_context_max_chars=schema_full_context_max_chars,
        schema_cache_ttl_seconds=schema_cache_ttl_seconds,
        enable_schema_embeddings=enable_schema_embeddings,
        openrouter_embedding_model=openrouter_embedding_model,
        schema_retriever_search_type=schema_retriever_search_type,
        schema_retriever_fetch_k=schema_retriever_fetch_k,
        log_level=log_level,
        db_connect_timeout_seconds=db_connect_timeout_seconds,
        memory_max_threads=memory_max_threads,
    )
