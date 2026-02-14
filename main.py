import argparse
import inspect
import logging
import sys

from dotenv import load_dotenv

from taxi_agent.config import load_settings
from taxi_agent.graph import TaxiDashboardAgent


DEFAULT_TEST_QUESTION = (
    "Bài test tự sửa SQL cực khó: Trong tháng 3/2018, hãy dùng QUALIFY để lấy top 10 cặp điểm đón/điểm trả theo từng "
    "hình thức thanh toán dựa trên tổng doanh thu (nếu schema thực tế dùng tên cột khác như PULocationID/DOLocationID thì tự ánh xạ). "
    "Đồng thời trả thêm: tổng số chuyến, tổng doanh thu, trung vị quãng đường, P90 thời lượng chuyến, tỷ lệ chuyến có tip > 0, "
    "doanh thu trên mỗi dặm, và xếp hạng trong từng hình thức thanh toán. "
    "Loại các chuyến bất thường (quãng đường <= 0, tổng tiền <= 0, thời lượng <= 0 hoặc > 180 phút)."
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
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass


def configure_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )
    # Keep third-party transport noise out of normal runs.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("openai._base_client").setLevel(logging.WARNING)


def main() -> int:
    configure_stdout()
    args = parse_args()
    load_dotenv()
    try:
        settings = load_settings()
    except Exception as exc:
        print(f"Configuration error: {exc}")
        return 1
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

    try:
        ask_signature = inspect.signature(agent.ask)
        supports_thread_id = "thread_id" in ask_signature.parameters
    except (TypeError, ValueError):
        supports_thread_id = False

    try:
        if supports_thread_id:
            result = agent.ask(question, thread_id=thread_id)
            effective_thread_id = thread_id
        else:
            # Backward compatibility for older/opaque agent signatures.
            result = agent.ask(question)
            effective_thread_id = "default"
            if thread_id != "default":
                logging.getLogger(__name__).warning(
                    "Agent.ask() signature does not expose thread_id; using default thread.",
                )
    except Exception as exc:
        logging.getLogger(__name__).exception("Agent execution failed: %s", exc)
        print(f"Runtime error: {exc}")
        return 1

    print("=== Taxi Agent Database Dashboard ===")
    print(f"Question: {question}")
    print(f"Thread ID: {effective_thread_id}")
    print(f"Route: {result.get('route', 'n/a')}")
    if result.get("intent"):
        print(f"Intent: {result.get('intent', 'n/a')}")
    print(f"Attempts: {result.get('attempts', 0)}")
    display_sql = result.get("sql_query") or result.get("last_failed_sql") or "n/a"
    print(f"SQL: {display_sql}")
    print("\nAnswer:")
    print(result.get("final_answer", "No answer"))

    sql_rows = result.get("sql_rows", [])
    if sql_rows:
        print("\nRows preview (first 5):")
        for row in sql_rows[:5]:
            print(row)

    sql_error = result.get("sql_error")
    if sql_error:
        if result.get("sql_error_type"):
            print(f"SQL error type: {result.get('sql_error_type')}")
        print(f"\nSQL error: {sql_error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

