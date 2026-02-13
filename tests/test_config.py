import pytest

from taxi_agent.config import load_settings


def _set_base_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("POSTGRES_DSN", "postgresql://postgres:postgres@localhost:5432/taxi_db")
    monkeypatch.setenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENROUTER_SITE_URL", "http://localhost")
    monkeypatch.setenv("OPENROUTER_APP_NAME", "Taxi-Agent-Dashboard")
    monkeypatch.setenv("QUERY_ROW_LIMIT", "100")
    monkeypatch.setenv("QUERY_TIMEOUT_MS", "30000")
    monkeypatch.setenv("MAX_SQL_RETRIES", "1")
    monkeypatch.setenv("DB_SCHEMA", "public")
    monkeypatch.setenv("SCHEMA_TOP_K_TABLES", "5")
    monkeypatch.setenv("SCHEMA_MAX_COLUMNS_PER_TABLE", "40")
    monkeypatch.setenv("SCHEMA_CONTEXT_MAX_CHARS", "12000")
    monkeypatch.setenv("SCHEMA_FULL_CONTEXT_MAX_CHARS", "30000")
    monkeypatch.setenv("SCHEMA_CACHE_TTL_SECONDS", "300")
    monkeypatch.setenv("ENABLE_SCHEMA_EMBEDDINGS", "true")
    monkeypatch.setenv("OPENROUTER_EMBEDDING_MODEL", "google/gemini-embedding-001")
    monkeypatch.setenv("SCHEMA_RETRIEVER_SEARCH_TYPE", "mmr")
    monkeypatch.setenv("SCHEMA_RETRIEVER_FETCH_K", "20")
    monkeypatch.setenv("LOG_LEVEL", "INFO")


def test_load_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_base_env(monkeypatch)
    settings = load_settings()

    assert settings.query_timeout_ms == 30000
    assert settings.schema_cache_ttl_seconds == 300
    assert settings.schema_full_context_max_chars == 30000
    assert settings.log_level == "INFO"


def test_load_settings_invalid_int(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_base_env(monkeypatch)
    monkeypatch.setenv("QUERY_ROW_LIMIT", "abc")
    with pytest.raises(ValueError):
        load_settings()


def test_load_settings_invalid_search_type_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_base_env(monkeypatch)
    monkeypatch.setenv("SCHEMA_RETRIEVER_SEARCH_TYPE", "invalid")
    settings = load_settings()
    assert settings.schema_retriever_search_type == "mmr"


def test_load_settings_search_type_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_base_env(monkeypatch)
    monkeypatch.setenv("SCHEMA_RETRIEVER_SEARCH_TYPE", "SIMILARITY")
    settings = load_settings()
    assert settings.schema_retriever_search_type == "similarity"


def test_load_settings_boolean_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_base_env(monkeypatch)
    monkeypatch.setenv("ENABLE_SCHEMA_EMBEDDINGS", "0")
    settings = load_settings()
    assert settings.enable_schema_embeddings is False

    monkeypatch.setenv("ENABLE_SCHEMA_EMBEDDINGS", "yes")
    settings = load_settings()
    assert settings.enable_schema_embeddings is True


def test_load_settings_invalid_boolean(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_base_env(monkeypatch)
    monkeypatch.setenv("ENABLE_SCHEMA_EMBEDDINGS", "maybe")
    with pytest.raises(ValueError):
        load_settings()


def test_schema_cache_ttl_zero_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_base_env(monkeypatch)
    monkeypatch.setenv("SCHEMA_CACHE_TTL_SECONDS", "0")
    settings = load_settings()
    assert settings.schema_cache_ttl_seconds == 0


def test_openrouter_api_key_whitespace_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_base_env(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "   ")
    with pytest.raises(ValueError):
        load_settings()


def test_log_level_invalid_fallback_to_info(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_base_env(monkeypatch)
    monkeypatch.setenv("LOG_LEVEL", "verbose")
    settings = load_settings()
    assert settings.log_level == "INFO"


def test_trimmed_optional_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_base_env(monkeypatch)
    monkeypatch.setenv("OPENROUTER_SITE_URL", "   ")
    monkeypatch.setenv("OPENROUTER_APP_NAME", "   ")
    settings = load_settings()
    assert settings.openrouter_site_url is None
    assert settings.openrouter_app_name is None


def test_blank_core_strings_fallback_to_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_base_env(monkeypatch)
    monkeypatch.setenv("POSTGRES_DSN", "   ")
    monkeypatch.setenv("OPENROUTER_MODEL", "   ")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "   ")
    settings = load_settings()
    assert settings.postgres_dsn == "postgresql://postgres:postgres@localhost:5432/taxi_db"
    assert settings.openrouter_model == "google/gemini-2.5-flash"
    assert settings.openrouter_base_url == "https://openrouter.ai/api/v1"


def test_blank_embedding_model_fallback_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_base_env(monkeypatch)
    monkeypatch.setenv("OPENROUTER_EMBEDDING_MODEL", "   ")
    settings = load_settings()
    assert settings.openrouter_embedding_model == "google/gemini-embedding-001"
