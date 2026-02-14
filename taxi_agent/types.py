from typing import Any, Dict, List, TypedDict


class DashboardState(TypedDict, total=False):
    question: str
    thread_id: str
    route: str
    route_reason: str
    intent: str
    intent_reason: str
    schema_overview: str
    schema_context: str
    schema_context_full: str
    metadata_context: str
    schema_error: str
    allowed_tables: List[str]
    all_allowed_tables: List[str]
    previous_question: str
    previous_sql_query: str
    previous_final_answer: str
    sql_query: str
    last_failed_sql: str
    sql_reasoning: str
    sql_rows: List[Dict[str, Any]]
    sql_error: str
    sql_error_type: str
    sql_error_message: str
    attempts: int
    final_answer: str


class AgentResult(DashboardState, total=False):
    """Public ask() result shape (backward-compatible superset of state)."""
