import threading
import time
from typing import Any

from taxi_agent.db import PostgresClient


class FakeCursor:
    def __init__(self) -> None:
        self.executed: list[tuple[str, Any]] = []

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False

    def execute(self, sql: str, params: Any = None) -> None:
        self.executed.append((sql, params))

    def fetchmany(self, size: int) -> list[dict[str, Any]]:
        _ = size
        return [{"ok": 1}]


class FakeConnection:
    def __init__(self, cursor: FakeCursor) -> None:
        self._cursor = cursor

    def __enter__(self) -> "FakeConnection":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False

    def cursor(self, row_factory: Any = None) -> FakeCursor:
        _ = row_factory
        return self._cursor


def test_run_query_sets_read_only_and_timeout(monkeypatch: Any) -> None:
    fake_cursor = FakeCursor()
    fake_conn = FakeConnection(fake_cursor)
    seen: dict[str, Any] = {}

    def fake_connect(dsn: str, connect_timeout: int | None = None) -> FakeConnection:
        seen["dsn"] = dsn
        seen["connect_timeout"] = connect_timeout
        return fake_conn

    monkeypatch.setattr("taxi_agent.db.psycopg.connect", fake_connect)

    client = PostgresClient(
        dsn="postgresql://postgres:postgres@localhost:5432/taxi_db",
        query_timeout_ms=1234,
        connect_timeout_seconds=7,
    )
    rows = client.run_query("SELECT 1")

    assert rows == [{"ok": 1}]
    assert seen["connect_timeout"] == 7
    assert fake_cursor.executed[0][0] == "SET TRANSACTION READ ONLY"
    assert fake_cursor.executed[1][0] == "SELECT set_config('search_path', %s, true)"
    assert fake_cursor.executed[1][1] == ("public",)
    assert fake_cursor.executed[2][0] == "SELECT set_config('statement_timeout', %s, true)"
    assert fake_cursor.executed[2][1] == ("1234",)
    assert fake_cursor.executed[3][0] == "SELECT 1"


def test_get_table_info_uses_langchain_sqldatabase_cache(monkeypatch: Any) -> None:
    created = {"count": 0}
    captured_kwargs: dict[str, Any] = {}

    class FakeSQLDatabase:
        def get_table_info(self, table_names: list[str]) -> str:
            return f"tables={','.join(table_names)}"

    def fake_from_uri(*args: Any, **kwargs: Any) -> FakeSQLDatabase:
        _ = args
        created["count"] += 1
        captured_kwargs.update(kwargs)
        return FakeSQLDatabase()

    monkeypatch.setattr("taxi_agent.db.SQLDatabase.from_uri", fake_from_uri)

    client = PostgresClient(
        dsn="postgresql://postgres:postgres@localhost:5432/taxi_db",
    )
    out1 = client.get_table_info(["taxi_trip_data"], table_schema="public")
    out2 = client.get_table_info(["taxi_trip_data"], table_schema="public")

    assert out1 == "tables=taxi_trip_data"
    assert out2 == "tables=taxi_trip_data"
    assert created["count"] == 1
    assert captured_kwargs["schema"] == "public"
    assert captured_kwargs["sample_rows_in_table_info"] == 0
    assert captured_kwargs["engine_args"]["connect_args"]["connect_timeout"] == 10


def test_get_table_info_cache_thread_safe(monkeypatch: Any) -> None:
    created = {"count": 0}

    class FakeSQLDatabase:
        def get_table_info(self, table_names: list[str]) -> str:
            return f"tables={','.join(table_names)}"

    def fake_from_uri(*args: Any, **kwargs: Any) -> FakeSQLDatabase:
        _ = (args, kwargs)
        time.sleep(0.05)
        created["count"] += 1
        return FakeSQLDatabase()

    monkeypatch.setattr("taxi_agent.db.SQLDatabase.from_uri", fake_from_uri)

    client = PostgresClient(
        dsn="postgresql://postgres:postgres@localhost:5432/taxi_db",
    )
    results: list[str] = []

    def worker() -> None:
        results.append(client.get_table_info(["taxi_trip_data"], table_schema="public"))

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(results) == 5
    assert all(item == "tables=taxi_trip_data" for item in results)
    assert created["count"] == 1


def test_check_connection_redacts_sensitive_error(monkeypatch: Any) -> None:
    def fake_connect(dsn: str, connect_timeout: int | None = None) -> Any:
        _ = (dsn, connect_timeout)
        raise RuntimeError(
            "failed postgresql://postgres:supersecret@localhost:5432/taxi_db"
        )

    monkeypatch.setattr("taxi_agent.db.psycopg.connect", fake_connect)

    client = PostgresClient(
        dsn="postgresql://postgres:postgres@localhost:5432/taxi_db",
    )
    ok, message = client.check_connection()
    assert ok is False
    assert "supersecret" not in message
    assert "***" in message
