import argparse
from dataclasses import replace

from main import main as run_main
from taxi_agent.config import Settings


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


def test_main_workflow_render_failure_still_runs(monkeypatch, capsys) -> None:
    class FakeAgent:
        def __init__(self, settings: Settings) -> None:
            _ = settings

        def get_workflow_mermaid(self) -> str:
            raise RuntimeError("mermaid failed")

        def save_workflow_mermaid(self, file_path: str = "agent_workflow.mmd") -> str:
            _ = file_path
            return "agent_workflow.mmd"

        def ask(self, question: str):
            return {
                "route": "sql",
                "sql_query": "SELECT 1",
                "final_answer": f"ok: {question}",
                "sql_rows": [{"x": 1}],
                "sql_error": "",
            }

    monkeypatch.setattr("main.parse_args", lambda: argparse.Namespace(question="abc"))
    monkeypatch.setattr("main.load_dotenv", lambda: None)
    monkeypatch.setattr("main.load_settings", _settings)
    monkeypatch.setattr("main.TaxiDashboardAgent", FakeAgent)

    run_main()
    out = capsys.readouterr().out

    assert "Workflow render/save skipped due to error" in out
    assert "Question: abc" in out
    assert "Route: sql" in out
    assert "ok: abc" in out


def test_main_passes_thread_id_when_supported(monkeypatch, capsys) -> None:
    captured = {"thread_id": ""}

    class FakeAgent:
        def __init__(self, settings: Settings) -> None:
            _ = settings

        def get_workflow_mermaid(self) -> str:
            return "graph TD;"

        def save_workflow_mermaid(self, file_path: str = "agent_workflow.mmd") -> str:
            _ = file_path
            return "agent_workflow.mmd"

        def ask(self, question: str, thread_id: str = "default"):
            captured["thread_id"] = thread_id
            return {
                "route": "sql",
                "intent": "sql_query",
                "sql_query": "SELECT 1",
                "final_answer": f"ok: {question}",
                "sql_rows": [],
                "sql_error": "",
            }

    monkeypatch.setattr(
        "main.parse_args",
        lambda: argparse.Namespace(question="abc", thread_id="team-finance"),
    )
    monkeypatch.setattr("main.load_dotenv", lambda: None)
    monkeypatch.setattr("main.load_settings", _settings)
    monkeypatch.setattr("main.TaxiDashboardAgent", FakeAgent)

    run_main()
    out = capsys.readouterr().out

    assert captured["thread_id"] == "team-finance"
    assert "Thread ID: team-finance" in out
