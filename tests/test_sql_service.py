import logging
from types import SimpleNamespace
from typing import Any, List

from taxi_agent.services.sql_service import (
    SQLService,
    classify_sql_error,
    redact_sensitive_text,
)


class FakeDB:
    def __init__(self, rows: List[dict[str, Any]] | None = None, should_fail: bool = False):
        self.rows = rows or []
        self.should_fail = should_fail

    def run_query(self, sql: str) -> List[dict[str, Any]]:
        _ = sql
        if self.should_fail:
            raise RuntimeError("statement timeout")
        return self.rows


class FakeSQLLLM:
    def __init__(self, sql: str = "SELECT 1", should_fail: bool = False):
        self.sql = sql
        self.should_fail = should_fail
        self.last_messages: Any = None

    def invoke(self, messages: Any) -> Any:
        self.last_messages = messages
        _ = messages
        if self.should_fail:
            raise RuntimeError("llm failed")
        return SimpleNamespace(sql=self.sql, reasoning="ok")


class ProviderFailSQLLLM(FakeSQLLLM):
    def invoke(self, messages: Any) -> Any:
        self.last_messages = messages
        _ = messages
        raise RuntimeError("401 Unauthorized")


class ConnectionFailSQLLLM(FakeSQLLLM):
    def invoke(self, messages: Any) -> Any:
        self.last_messages = messages
        _ = messages
        raise RuntimeError("could not connect to server: Connection refused")


class SensitiveFailSQLLLM(FakeSQLLLM):
    def invoke(self, messages: Any) -> Any:
        self.last_messages = messages
        _ = messages
        raise RuntimeError(
            "could not connect using postgresql://postgres:supersecret@localhost:5432/taxi_db "
            "password=supersecret api_key=sk-secret-token"
        )


class RawSQLLLM:
    def __init__(self, content: str):
        self.content = content

    def invoke(self, messages: Any) -> Any:
        _ = messages
        return SimpleNamespace(content=self.content)


def test_classify_sql_error() -> None:
    assert classify_sql_error("outside allowed schema context") == "allowlist"
    assert classify_sql_error("Only SELECT queries are allowed.") == "guard"
    assert classify_sql_error("statement timeout") == "timeout"
    assert classify_sql_error("Error code: 401 - User not found.") == "provider"
    assert classify_sql_error("connection refused") == "connection"
    assert classify_sql_error("could not translate host name \"localhostx\"") == "connection"
    assert classify_sql_error("relation does not exist") == "db"
    assert classify_sql_error('relation "only_table" does not exist') == "db"


def test_generate_sql_requires_schema_context() -> None:
    service = SQLService(
        sql_llm=FakeSQLLLM(),
        db=FakeDB(),  # type: ignore[arg-type]
        logger=logging.getLogger("test.sql"),
    )
    result = service.generate_sql(question="q", schema_context="", allowed_tables=[])
    assert result["sql_error_type"] == "schema_context"
    assert result["sql_query"] == ""


def test_execute_sql_guard_rejection() -> None:
    service = SQLService(
        sql_llm=FakeSQLLLM(),
        db=FakeDB(),  # type: ignore[arg-type]
        logger=logging.getLogger("test.sql"),
    )
    result = service.execute_sql(
        sql_query="SELECT * FROM public.other_table",
        allowed_tables=["public.taxi_trip_data", "taxi_trip_data"],
    )
    assert result["sql_error_type"] == "allowlist"


def test_execute_sql_rejects_empty_allowlist() -> None:
    service = SQLService(
        sql_llm=FakeSQLLLM(),
        db=FakeDB(),  # type: ignore[arg-type]
        logger=logging.getLogger("test.sql"),
    )
    result = service.execute_sql(
        sql_query="SELECT * FROM public.taxi_trip_data LIMIT 1",
        allowed_tables=[],
    )
    assert result["sql_error_type"] == "allowlist"
    assert "refusing to execute" in result["sql_error"].lower()


def test_execute_sql_db_timeout() -> None:
    service = SQLService(
        sql_llm=FakeSQLLLM(),
        db=FakeDB(should_fail=True),  # type: ignore[arg-type]
        logger=logging.getLogger("test.sql"),
    )
    result = service.execute_sql(
        sql_query="SELECT * FROM public.taxi_trip_data LIMIT 1",
        allowed_tables=["public.taxi_trip_data", "taxi_trip_data"],
    )
    assert result["sql_error_type"] == "timeout"


def test_execute_sql_guard_non_allowlist() -> None:
    service = SQLService(
        sql_llm=FakeSQLLLM(),
        db=FakeDB(),  # type: ignore[arg-type]
        logger=logging.getLogger("test.sql"),
    )
    result = service.execute_sql(
        sql_query="DELETE FROM public.taxi_trip_data",
        allowed_tables=["public.taxi_trip_data", "taxi_trip_data"],
    )
    assert result["sql_error_type"] == "guard"


def test_generate_sql_empty_output_is_generation_error() -> None:
    service = SQLService(
        sql_llm=FakeSQLLLM(sql="   "),
        db=FakeDB(),  # type: ignore[arg-type]
        logger=logging.getLogger("test.sql"),
    )
    result = service.generate_sql(
        question="q",
        schema_context="Table: public.taxi_trip_data",
        allowed_tables=["public.taxi_trip_data", "taxi_trip_data"],
    )
    assert result["sql_error_type"] == "generation"
    assert "empty sql" in result["sql_error"].lower()


def test_generate_sql_provider_error_classification() -> None:
    service = SQLService(
        sql_llm=ProviderFailSQLLLM(),
        db=FakeDB(),  # type: ignore[arg-type]
        logger=logging.getLogger("test.sql"),
    )
    result = service.generate_sql(
        question="q",
        schema_context="Table: public.taxi_trip_data",
        allowed_tables=["public.taxi_trip_data", "taxi_trip_data"],
    )
    assert result["sql_error_type"] == "provider"
    assert "401" in result["sql_error"]


def test_generate_sql_connection_error_classification() -> None:
    service = SQLService(
        sql_llm=ConnectionFailSQLLLM(),
        db=FakeDB(),  # type: ignore[arg-type]
        logger=logging.getLogger("test.sql"),
    )
    result = service.generate_sql(
        question="q",
        schema_context="Table: public.taxi_trip_data",
        allowed_tables=["public.taxi_trip_data", "taxi_trip_data"],
    )
    assert result["sql_error_type"] == "connection"
    assert "connect" in result["sql_error"].lower()


def test_repair_sql_connection_error_classification() -> None:
    service = SQLService(
        sql_llm=ConnectionFailSQLLLM(),
        db=FakeDB(),  # type: ignore[arg-type]
        logger=logging.getLogger("test.sql"),
    )
    result = service.repair_sql(
        question="q",
        failed_sql="SELECT 1",
        sql_error="err",
        schema_context="Table: public.taxi_trip_data",
        allowed_tables=["public.taxi_trip_data", "taxi_trip_data"],
        attempts=1,
    )
    assert result["sql_error_type"] == "connection"


def test_redact_sensitive_text_masks_dsn_and_api_key() -> None:
    raw = (
        "failed with postgresql://postgres:supersecret@localhost:5432/taxi_db "
        "password=supersecret api_key=sk-live-secret "
        "Authorization: Bearer token-value"
    )
    redacted = redact_sensitive_text(raw)
    assert "supersecret" not in redacted
    assert "sk-live-secret" not in redacted
    assert "token-value" not in redacted
    assert "***" in redacted


def test_generate_sql_redacts_sensitive_error_message() -> None:
    service = SQLService(
        sql_llm=SensitiveFailSQLLLM(),
        db=FakeDB(),  # type: ignore[arg-type]
        logger=logging.getLogger("test.sql"),
    )
    result = service.generate_sql(
        question="q",
        schema_context="Table: public.taxi_trip_data",
        allowed_tables=["public.taxi_trip_data", "taxi_trip_data"],
    )
    assert "supersecret" not in result["sql_error"]
    assert "sk-secret-token" not in result["sql_error"]


def test_generate_sql_fallback_parses_raw_sql_when_structured_fails() -> None:
    service = SQLService(
        sql_llm=FakeSQLLLM(should_fail=True),
        db=FakeDB(),  # type: ignore[arg-type]
        logger=logging.getLogger("test.sql"),
        raw_llm=RawSQLLLM("```sql\nSELECT * FROM public.taxi_trip_data LIMIT 1;\n```"),
    )
    result = service.generate_sql(
        question="q",
        schema_context="Table: public.taxi_trip_data",
        allowed_tables=["public.taxi_trip_data", "taxi_trip_data"],
    )
    assert result["sql_error"] == ""
    assert result["sql_query"] == "SELECT * FROM public.taxi_trip_data LIMIT 1"


def test_repair_sql_empty_output_is_repair_error() -> None:
    service = SQLService(
        sql_llm=FakeSQLLLM(sql="   "),
        db=FakeDB(),  # type: ignore[arg-type]
        logger=logging.getLogger("test.sql"),
    )
    result = service.repair_sql(
        question="q",
        failed_sql="SELECT 1",
        sql_error="err",
        schema_context="Table: public.taxi_trip_data",
        allowed_tables=["public.taxi_trip_data", "taxi_trip_data"],
        attempts=1,
    )
    assert result["sql_error_type"] == "repair"
    assert "empty sql" in result["sql_error"].lower()
    assert result["sql_query"] == ""
    assert result["last_failed_sql"] == "SELECT 1"


def test_repair_sql_requires_schema_context() -> None:
    service = SQLService(
        sql_llm=FakeSQLLLM(),
        db=FakeDB(),  # type: ignore[arg-type]
        logger=logging.getLogger("test.sql"),
    )
    result = service.repair_sql(
        question="q",
        failed_sql="SELECT 1",
        sql_error="err",
        schema_context="",
        allowed_tables=[],
        attempts=1,
    )
    assert result["sql_error_type"] == "schema_context"
    assert result["sql_query"] == ""
    assert result["last_failed_sql"] == "SELECT 1"


def test_repair_sql_provider_error_classification() -> None:
    service = SQLService(
        sql_llm=FakeSQLLLM(should_fail=True),
        db=FakeDB(),  # type: ignore[arg-type]
        logger=logging.getLogger("test.sql"),
    )
    result = service.repair_sql(
        question="q",
        failed_sql="SELECT 1",
        sql_error="err",
        schema_context="Table: public.taxi_trip_data",
        allowed_tables=["public.taxi_trip_data", "taxi_trip_data"],
        attempts=1,
    )
    # FakeSQLLLM throws generic RuntimeError so still repair.
    assert result["sql_error_type"] == "repair"
    assert result["last_failed_sql"] == "SELECT 1"


def test_generate_sql_prompt_uses_configured_row_limit() -> None:
    llm = FakeSQLLLM(sql="SELECT 1")
    service = SQLService(
        sql_llm=llm,
        db=FakeDB(),  # type: ignore[arg-type]
        logger=logging.getLogger("test.sql"),
        row_limit=250,
    )
    _ = service.generate_sql(
        question="q",
        schema_context="Table: public.taxi_trip_data",
        allowed_tables=["public.taxi_trip_data", "taxi_trip_data"],
    )
    system_prompt = llm.last_messages[0].content
    assert "LIMIT 250" in system_prompt
