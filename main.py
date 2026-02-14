import argparse
import inspect
import logging
import sys

from dotenv import load_dotenv

from taxi_agent.config import load_settings
from taxi_agent.graph import TaxiDashboardAgent


DEFAULT_TEST_QUESTION = (
    "Trong quý 2/2018, với từng cặp (PULocationID, DOLocationID) và payment_type, "
    "hãy tính số chuyến, tổng doanh thu total_amount, median trip_distance, thời lượng "
    "trung bình chuyến (phút, từ pickup đến dropoff), tỷ lệ chuyến có tip_amount > 0, "
    "và doanh thu trên mỗi dặm. Chỉ giữ các nhóm có ít nhất 5.000 chuyến, loại bỏ các "
    "chuyến có trip_distance <= 0 hoặc total_amount <= 0, sau đó trả về top 10 nhóm theo "
    "doanh thu giảm dần và thêm cột xếp hạng dense_rank."
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Taxi Agent Database Dashboard")
    parser.add_argument(
        "--question",
        type=str,
        default="",
        help="Optional input question (VN/EN).",
    )
    parser.add_argument(
        "--thread-id",
        type=str,
        default="default",
        help="Optional conversation thread id for follow-up context.",
    )
    return parser.parse_args(argv)


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def configure_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    # Keep third-party transport noise out of normal runs.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def main() -> None:
    configure_stdout()
    args = parse_args()
    load_dotenv()
    settings = load_settings()
    configure_logging(settings.log_level)

    question = args.question.strip() or DEFAULT_TEST_QUESTION
    thread_id = (getattr(args, "thread_id", "default") or "default").strip() or "default"

    agent = TaxiDashboardAgent(settings)
    try:
        workflow_mermaid = agent.get_workflow_mermaid()
        workflow_file = agent.save_workflow_mermaid("agent_workflow.mmd")
        print("=== LangGraph Agent Workflow (Mermaid) ===")
        print(workflow_mermaid)
        print(f"\nWorkflow file saved at: {workflow_file}\n")
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "Workflow render/save failed: %s",
            exc,
        )
        print(f"Workflow render/save skipped due to error: {exc}\n")

    ask_signature = inspect.signature(agent.ask)
    if "thread_id" in ask_signature.parameters:
        result = agent.ask(question, thread_id=thread_id)
    else:
        # Backward compatibility for older agent signatures.
        result = agent.ask(question)

    print("=== Taxi Agent Database Dashboard ===")
    print(f"Question: {question}")
    print(f"Thread ID: {thread_id}")
    print(f"Route: {result.get('route', 'n/a')}")
    if result.get("intent"):
        print(f"Intent: {result.get('intent', 'n/a')}")
    print(f"SQL: {result.get('sql_query', 'n/a')}")
    print("\nAnswer:")
    print(result.get("final_answer", "No answer"))

    sql_rows = result.get("sql_rows", [])
    if sql_rows:
        print("\nRows preview (first 5):")
        for row in sql_rows[:5]:
            print(row)

    sql_error = result.get("sql_error")
    if sql_error:
        print(f"\nSQL error: {sql_error}")


if __name__ == "__main__":
    main()
