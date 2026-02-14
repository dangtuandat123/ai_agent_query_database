import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taxi_agent.config import load_settings
from taxi_agent.graph import TaxiDashboardAgent


QUESTIONS = [
    "Trong tháng 3/2018, payment_type nào có tổng doanh thu cao nhất?",
    "For March 2018, show top 3 payment types by total revenue.",
]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    load_dotenv()
    try:
        settings = load_settings()
    except Exception as exc:
        print(f"Cannot run smoke test: {exc}")
        return 1

    agent = TaxiDashboardAgent(settings)
    try:
        output_file = agent.save_workflow_mermaid("agent_workflow.mmd")
        print(f"Workflow written to: {Path(output_file)}")
    except Exception as exc:
        print(f"Workflow render/save skipped due to error: {exc}")

    for index, question in enumerate(QUESTIONS, start=1):
        print(f"\n--- Smoke case {index} ---")
        print(f"Q: {question}")
        result = agent.ask(question)
        print(f"Route: {result.get('route', 'n/a')}")
        print(f"Attempts: {result.get('attempts', 0)}")
        display_sql = result.get("sql_query") or result.get("last_failed_sql") or "n/a"
        print(f"SQL: {display_sql}")
        answer = result.get("final_answer", "")
        print(f"Answer: {answer[:500]}")
        if result.get("sql_error"):
            print(f"SQL error: {result['sql_error']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
