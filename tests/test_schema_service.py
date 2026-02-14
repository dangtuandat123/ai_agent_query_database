import logging
from typing import List

from taxi_agent.schema import ColumnSchema, TableSchema
from taxi_agent.services.schema_service import SchemaService


class FakeDB:
    def __init__(self, tables: List[TableSchema]):
        self.tables = tables
        self.calls = 0

    def get_table_schemas(self, table_schema: str = "public") -> List[TableSchema]:
        _ = table_schema
        self.calls += 1
        return self.tables


class FakeLangChainDB(FakeDB):
    def __init__(self, tables: List[TableSchema], table_info: str):
        super().__init__(tables)
        self.table_info = table_info
        self.info_calls = 0

    def get_table_info(self, table_names: List[str], table_schema: str = "public") -> str:
        _ = (table_names, table_schema)
        self.info_calls += 1
        return self.table_info


class FakeRetriever:
    def __init__(self, selected_tables: List[TableSchema], should_raise: bool = False):
        self.selected_tables = selected_tables
        self.should_raise = should_raise

    def refresh(self, tables: List[TableSchema]) -> None:
        _ = tables

    def retrieve_tables(self, question: str) -> List[TableSchema]:
        _ = question
        if self.should_raise:
            raise RuntimeError("retriever failure")
        return self.selected_tables


def _tables() -> List[TableSchema]:
    table_a = TableSchema(
        table_schema="public",
        table_name="taxi_trip_data",
        columns=[
            ColumnSchema("payment_type", "integer", 1),
            ColumnSchema("total_amount", "double precision", 2),
        ],
    )
    table_b = TableSchema(
        table_schema="public",
        table_name="zones",
        columns=[ColumnSchema("zone_id", "integer", 1)],
    )
    return [table_a, table_b]


def test_schema_service_uses_cache() -> None:
    tables = _tables()
    db = FakeDB(tables)
    retriever = FakeRetriever([tables[0]])
    service = SchemaService(
        db=db,  # type: ignore[arg-type]
        schema_retriever=retriever,  # type: ignore[arg-type]
        db_schema="public",
        max_columns_per_table=40,
        context_max_chars=1000,
        full_context_max_chars=3000,
        top_k_tables=1,
        cache_ttl_seconds=300,
        logger=logging.getLogger("test.schema"),
    )

    first = service.build_for_question("q1")
    second = service.build_for_question("q2")

    assert db.calls == 1
    assert first.schema_error == ""
    assert second.schema_error == ""
    assert "public.taxi_trip_data" in first.schema_context


def test_schema_service_cache_can_be_disabled() -> None:
    tables = _tables()
    db = FakeDB(tables)
    retriever = FakeRetriever([tables[0]])
    service = SchemaService(
        db=db,  # type: ignore[arg-type]
        schema_retriever=retriever,  # type: ignore[arg-type]
        db_schema="public",
        max_columns_per_table=40,
        context_max_chars=1000,
        full_context_max_chars=3000,
        top_k_tables=1,
        cache_ttl_seconds=0,
        logger=logging.getLogger("test.schema"),
    )

    _ = service.build_for_question("q1")
    _ = service.build_for_question("q2")
    assert db.calls == 2


def test_schema_service_invalidate_cache_forces_reload() -> None:
    tables = _tables()
    db = FakeDB(tables)
    retriever = FakeRetriever([tables[0]])
    service = SchemaService(
        db=db,  # type: ignore[arg-type]
        schema_retriever=retriever,  # type: ignore[arg-type]
        db_schema="public",
        max_columns_per_table=40,
        context_max_chars=1000,
        full_context_max_chars=3000,
        top_k_tables=1,
        cache_ttl_seconds=300,
        logger=logging.getLogger("test.schema"),
    )

    _ = service.build_for_question("q1")
    service.invalidate_cache()
    _ = service.build_for_question("q2")
    assert db.calls == 2


def test_schema_service_fallback_when_retriever_fails() -> None:
    tables = _tables()
    db = FakeDB(tables)
    retriever = FakeRetriever([tables[0]], should_raise=True)
    service = SchemaService(
        db=db,  # type: ignore[arg-type]
        schema_retriever=retriever,  # type: ignore[arg-type]
        db_schema="public",
        max_columns_per_table=40,
        context_max_chars=1000,
        full_context_max_chars=3000,
        top_k_tables=1,
        cache_ttl_seconds=300,
        logger=logging.getLogger("test.schema"),
    )

    result = service.build_for_question("q")
    assert result.schema_error == ""
    # fallback should pick top_k_tables=1 from all tables.
    assert result.allowed_tables == ["public.taxi_trip_data", "taxi_trip_data"]


def test_schema_service_fallback_when_retriever_returns_empty() -> None:
    tables = _tables()
    db = FakeDB(tables)
    retriever = FakeRetriever([])
    service = SchemaService(
        db=db,  # type: ignore[arg-type]
        schema_retriever=retriever,  # type: ignore[arg-type]
        db_schema="public",
        max_columns_per_table=40,
        context_max_chars=1000,
        full_context_max_chars=3000,
        top_k_tables=1,
        cache_ttl_seconds=300,
        logger=logging.getLogger("test.schema"),
    )

    result = service.build_for_question("q")
    assert result.schema_error == ""
    assert result.allowed_tables == ["public.taxi_trip_data", "taxi_trip_data"]


def test_schema_service_caches_empty_schema() -> None:
    db = FakeDB([])
    retriever = FakeRetriever([])
    service = SchemaService(
        db=db,  # type: ignore[arg-type]
        schema_retriever=retriever,  # type: ignore[arg-type]
        db_schema="public",
        max_columns_per_table=40,
        context_max_chars=1000,
        full_context_max_chars=3000,
        top_k_tables=1,
        cache_ttl_seconds=300,
        logger=logging.getLogger("test.schema"),
    )

    first = service.build_for_question("q1")
    second = service.build_for_question("q2")

    assert db.calls == 1
    assert "No tables found" in first.schema_error
    assert "No tables found" in second.schema_error


def test_schema_service_prefers_langchain_table_info() -> None:
    tables = _tables()
    db = FakeLangChainDB(
        tables=tables,
        table_info="CREATE TABLE public.taxi_trip_data (payment_type INTEGER);",
    )
    retriever = FakeRetriever([tables[0]])
    service = SchemaService(
        db=db,  # type: ignore[arg-type]
        schema_retriever=retriever,  # type: ignore[arg-type]
        db_schema="public",
        max_columns_per_table=40,
        context_max_chars=1000,
        full_context_max_chars=3000,
        top_k_tables=1,
        cache_ttl_seconds=300,
        logger=logging.getLogger("test.schema"),
    )

    result = service.build_for_question("q")
    assert "CREATE TABLE public.taxi_trip_data" in result.schema_context
    assert db.info_calls >= 1


def test_schema_service_langchain_table_info_truncates() -> None:
    tables = _tables()
    db = FakeLangChainDB(
        tables=tables,
        table_info="CREATE TABLE public.taxi_trip_data (" + ("x " * 200) + ");",
    )
    retriever = FakeRetriever([tables[0]])
    service = SchemaService(
        db=db,  # type: ignore[arg-type]
        schema_retriever=retriever,  # type: ignore[arg-type]
        db_schema="public",
        max_columns_per_table=40,
        context_max_chars=80,
        full_context_max_chars=3000,
        top_k_tables=1,
        cache_ttl_seconds=300,
        logger=logging.getLogger("test.schema"),
    )

    result = service.build_for_question("q")
    assert len(result.schema_context) <= 80


def test_schema_service_reuses_cached_full_context() -> None:
    tables = _tables()
    db = FakeLangChainDB(
        tables=tables,
        table_info="CREATE TABLE public.taxi_trip_data (payment_type INTEGER);",
    )
    retriever = FakeRetriever([tables[0]])
    service = SchemaService(
        db=db,  # type: ignore[arg-type]
        schema_retriever=retriever,  # type: ignore[arg-type]
        db_schema="public",
        max_columns_per_table=40,
        context_max_chars=1000,
        full_context_max_chars=3000,
        top_k_tables=1,
        cache_ttl_seconds=300,
        logger=logging.getLogger("test.schema"),
    )

    _ = service.build_for_question("q1")
    _ = service.build_for_question("q2")
    # First run: schema_context + full_context. Second run: schema_context only.
    assert db.info_calls == 3
