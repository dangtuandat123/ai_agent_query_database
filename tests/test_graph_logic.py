from dataclasses import replace
from types import SimpleNamespace
from typing import Any, List

import pytest

from taxi_agent.config import Settings
from taxi_agent.graph import IntentDecision, RouteDecision, SQLDraft, TaxiDashboardAgent
from taxi_agent.schema import ColumnSchema, TableSchema


class FakeRetriever:
    def __init__(self, selected_tables: List[TableSchema]):
        self.selected_tables = selected_tables

    def refresh(self, tables: List[TableSchema]) -> None:
        _ = tables

    def retrieve_tables(self, question: str) -> List[TableSchema]:
        _ = question
        return self.selected_tables


class FakeDB:
    def __init__(self, tables: List[TableSchema], rows: List[dict[str, Any]]):
        self.tables = tables
        self.rows = rows
        self.queries: List[str] = []
        self.schema_calls = 0

    def get_table_schemas(self, table_schema: str = "public") -> List[TableSchema]:
        _ = table_schema
        self.schema_calls += 1
        return self.tables

    def run_query(self, sql: str) -> List[dict[str, Any]]:
        self.queries.append(sql)
        return self.rows


class FakeLLM:
    def __init__(
        self,
        *,
        route: str = "sql",
        route_reason: str = "ok",
        intent: str = "sql_query",
        intent_reason: str = "standalone",
        sql_first: str = "SELECT 1",
        sql_second: str = "SELECT 1",
        answer_text: str = "answer",
        fail_answer: bool = False,
        sql_fail_on_calls: set[int] | None = None,
    ):
        self.route = route
        self.route_reason = route_reason
        self.intent = intent
        self.intent_reason = intent_reason
        self.sql_first = sql_first
        self.sql_second = sql_second
        self.answer_text = answer_text
        self.fail_answer = fail_answer
        self.sql_fail_on_calls = sql_fail_on_calls or set()
        self.sql_calls = 0

    def with_structured_output(self, schema: Any) -> Any:
        if schema is RouteDecision:
            return SimpleNamespace(
                invoke=lambda messages: SimpleNamespace(
                    route=self.route,
                    reason=self.route_reason,
                )
            )
        if schema is IntentDecision:
            return SimpleNamespace(
                invoke=lambda messages: SimpleNamespace(
                    intent=self.intent,
                    reason=self.intent_reason,
                )
            )
        if schema is SQLDraft:
            return SimpleNamespace(invoke=self._invoke_sql)
        raise AssertionError("Unexpected schema type")

    def _invoke_sql(self, messages: Any) -> Any:
        _ = messages
        self.sql_calls += 1
        if self.sql_calls in self.sql_fail_on_calls:
            raise RuntimeError("llm sql failure")
        if self.sql_calls == 1:
            return SimpleNamespace(sql=self.sql_first, reasoning="first")
        return SimpleNamespace(sql=self.sql_second, reasoning="second")

    def invoke(self, messages: Any) -> Any:
        _ = messages
        if self.fail_answer:
            raise RuntimeError("LLM answer failure")
        return SimpleNamespace(content=self.answer_text)


def _settings() -> Settings:
    return Settings(
        postgres_dsn="postgresql://postgres:postgres@localhost:5432/taxi_db",
        openrouter_api_key="test-key",
        openrouter_model="google/gemini-2.5-flash",
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_site_url="http://localhost",
        openrouter_app_name="Taxi-Agent-Dashboard",
        query_row_limit=100,
        query_timeout_ms=30000,
        max_sql_retries=1,
        db_schema="public",
        schema_top_k_tables=1,
        schema_max_columns_per_table=40,
        schema_context_max_chars=12000,
        schema_full_context_max_chars=30000,
        schema_cache_ttl_seconds=300,
        enable_schema_embeddings=False,
        openrouter_embedding_model="google/gemini-embedding-001",
        schema_retriever_search_type="mmr",
        schema_retriever_fetch_k=20,
        log_level="INFO",
    )


def _tables() -> List[TableSchema]:
    table_a = TableSchema(
        table_schema="public",
        table_name="table_a",
        columns=[ColumnSchema("id", "integer", 1)],
    )
    table_b = TableSchema(
        table_schema="public",
        table_name="table_b",
        columns=[ColumnSchema("id", "integer", 1)],
    )
    return [table_a, table_b]


def test_graph_repairs_with_expanded_allowlist() -> None:
    tables = _tables()
    fake_db = FakeDB(tables=tables, rows=[{"id": 1}])
    fake_llm = FakeLLM(
        route="sql",
        sql_first="SELECT * FROM public.table_b LIMIT 1",
        sql_second="SELECT * FROM public.table_b LIMIT 1",
        answer_text="done",
    )
    fake_retriever = FakeRetriever(selected_tables=[tables[0]])

    agent = TaxiDashboardAgent(
        _settings(),
        db_client=fake_db,  # type: ignore[arg-type]
        llm=fake_llm,  # type: ignore[arg-type]
        schema_retriever=fake_retriever,  # type: ignore[arg-type]
    )
    result = agent.ask("Use table_b")

    assert result["route"] == "sql"
    assert result["intent"] == "sql_query"
    assert result["sql_error"] == ""
    assert result["attempts"] == 1
    assert "table_b" in result["sql_query"].lower()
    assert result["final_answer"] == "done"
    assert any("table_b" in item for item in result.get("allowed_tables", []))


def test_graph_unsupported_route() -> None:
    tables = _tables()
    fake_db = FakeDB(tables=tables, rows=[])
    fake_llm = FakeLLM(route="unsupported", route_reason="external data required")
    fake_retriever = FakeRetriever(selected_tables=[tables[0]])

    agent = TaxiDashboardAgent(
        _settings(),
        db_client=fake_db,  # type: ignore[arg-type]
        llm=fake_llm,  # type: ignore[arg-type]
        schema_retriever=fake_retriever,  # type: ignore[arg-type]
    )
    result = agent.ask("What is weather in Hanoi?")

    assert result["route"] == "unsupported"
    assert "cannot answer" in result["final_answer"].lower()


def test_graph_answer_fallback_when_llm_fails() -> None:
    tables = _tables()
    fake_db = FakeDB(tables=tables, rows=[{"id": 1}, {"id": 2}])
    fake_llm = FakeLLM(
        route="sql",
        sql_first="SELECT * FROM public.table_a LIMIT 2",
        sql_second="SELECT * FROM public.table_a LIMIT 2",
        fail_answer=True,
    )
    fake_retriever = FakeRetriever(selected_tables=[tables[0]])

    s = replace(_settings(), max_sql_retries=0)
    agent = TaxiDashboardAgent(
        s,
        db_client=fake_db,  # type: ignore[arg-type]
        llm=fake_llm,  # type: ignore[arg-type]
        schema_retriever=fake_retriever,  # type: ignore[arg-type]
    )
    result = agent.ask("How many rows?")

    assert result["sql_error"] == ""
    assert "Query succeeded" in result["final_answer"]


def test_graph_empty_question_short_circuit() -> None:
    tables = _tables()
    fake_db = FakeDB(tables=tables, rows=[{"id": 1}])
    fake_llm = FakeLLM()
    fake_retriever = FakeRetriever(selected_tables=[tables[0]])

    agent = TaxiDashboardAgent(
        _settings(),
        db_client=fake_db,  # type: ignore[arg-type]
        llm=fake_llm,  # type: ignore[arg-type]
        schema_retriever=fake_retriever,  # type: ignore[arg-type]
    )
    result = agent.ask("   ")

    assert result["route"] == "unsupported"
    assert result["route_reason"] == "Empty question."
    assert "The question is empty" in result["final_answer"]
    assert fake_db.schema_calls == 0
    assert fake_db.queries == []


def test_graph_generation_error_preserved_without_execute() -> None:
    tables = _tables()
    fake_db = FakeDB(tables=tables, rows=[{"id": 1}])
    fake_llm = FakeLLM(route="sql", sql_fail_on_calls={1})
    fake_retriever = FakeRetriever(selected_tables=[tables[0]])

    s = replace(_settings(), max_sql_retries=0)
    agent = TaxiDashboardAgent(
        s,
        db_client=fake_db,  # type: ignore[arg-type]
        llm=fake_llm,  # type: ignore[arg-type]
        schema_retriever=fake_retriever,  # type: ignore[arg-type]
    )
    result = agent.ask("Do a SQL query")

    assert result["sql_error_type"] == "generation"
    assert "SQL generation failed" in result["sql_error"]
    assert fake_db.queries == []


def test_graph_generation_error_then_regenerate_success() -> None:
    tables = _tables()
    fake_db = FakeDB(tables=tables, rows=[{"id": 1}])
    fake_llm = FakeLLM(
        route="sql",
        sql_first="SELECT * FROM public.table_a LIMIT 1",
        sql_second="SELECT * FROM public.table_a LIMIT 1",
        answer_text="done",
        sql_fail_on_calls={1},
    )
    fake_retriever = FakeRetriever(selected_tables=[tables[0]])

    s = replace(_settings(), max_sql_retries=1)
    agent = TaxiDashboardAgent(
        s,
        db_client=fake_db,  # type: ignore[arg-type]
        llm=fake_llm,  # type: ignore[arg-type]
        schema_retriever=fake_retriever,  # type: ignore[arg-type]
    )
    result = agent.ask("Show one row")

    assert result["sql_error"] == ""
    assert result["attempts"] == 1
    assert result["final_answer"] == "done"
    assert len(fake_db.queries) == 1
    assert fake_llm.sql_calls == 2


def test_graph_followup_intent_uses_previous_turn_context() -> None:
    tables = _tables()
    fake_db = FakeDB(tables=tables, rows=[{"id": 1}])
    fake_llm = FakeLLM(
        route="sql",
        intent="sql_followup",
        sql_first="SELECT * FROM public.table_a LIMIT 1",
        sql_second="SELECT * FROM public.table_b LIMIT 1",
        answer_text="done",
    )
    fake_retriever = FakeRetriever(selected_tables=tables)
    agent = TaxiDashboardAgent(
        _settings(),
        db_client=fake_db,  # type: ignore[arg-type]
        llm=fake_llm,  # type: ignore[arg-type]
        schema_retriever=fake_retriever,  # type: ignore[arg-type]
    )

    first = agent.ask("Show one row from table_a")
    second = agent.ask("còn table_b thì sao?")

    assert first["sql_error"] == ""
    assert second["intent"] == "sql_followup"
    assert second["previous_question"] == "Show one row from table_a"
    assert "table_b" in second["sql_query"].lower()


def test_graph_embedding_init_failure_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    tables = _tables()
    fake_db = FakeDB(tables=tables, rows=[{"id": 1}])
    fake_llm = FakeLLM(route="unsupported", route_reason="not sql")

    class BrokenEmbeddings:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _ = (args, kwargs)
            raise RuntimeError("bad embedding config")

    monkeypatch.setattr("taxi_agent.graph.OpenAIEmbeddings", BrokenEmbeddings)

    agent = TaxiDashboardAgent(
        _settings(),
        db_client=fake_db,  # type: ignore[arg-type]
        llm=fake_llm,  # type: ignore[arg-type]
    )
    assert agent.embedding_model is None


def test_graph_internal_failure_returns_safe_response() -> None:
    tables = _tables()
    fake_db = FakeDB(tables=tables, rows=[{"id": 1}])
    fake_llm = FakeLLM()
    fake_retriever = FakeRetriever(selected_tables=[tables[0]])

    agent = TaxiDashboardAgent(
        _settings(),
        db_client=fake_db,  # type: ignore[arg-type]
        llm=fake_llm,  # type: ignore[arg-type]
        schema_retriever=fake_retriever,  # type: ignore[arg-type]
    )

    class BrokenGraph:
        def invoke(self, state: Any) -> Any:
            _ = state
            raise RuntimeError("graph crashed")

    agent.graph = BrokenGraph()  # type: ignore[assignment]
    result = agent.ask("How many trips?")

    assert result["route"] == "unsupported"
    assert result["sql_error_type"] == "internal"
    assert "graph crashed" in result["sql_error"]
    assert "internal error" in result["final_answer"].lower()
