import logging
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from ..db import PostgresClient
from ..prompts import SQL_GENERATOR_SYSTEM_PROMPT, SQL_REPAIR_SYSTEM_PROMPT
from ..sql_guard import normalize_sql, validate_readonly_sql


def classify_sql_error(sql_error: str) -> str:
    lowered = sql_error.lower()
    guard_markers = (
        "only one sql statement is allowed",
        "only select queries are allowed",
        "select into is not allowed",
        "locking clauses are not allowed",
        "write or destructive sql is not allowed",
        "query must reference at least one table",
        "sql is empty",
    )
    if "outside allowed schema context" in lowered:
        return "allowlist"
    if any(marker in lowered for marker in guard_markers):
        return "guard"
    if "timeout" in lowered or "statement timeout" in lowered:
        return "timeout"
    if (
        "401" in lowered
        or "403" in lowered
        or "unauthorized" in lowered
        or "forbidden" in lowered
        or "invalid api key" in lowered
        or "user not found" in lowered
    ):
        return "provider"
    if (
        "connection refused" in lowered
        or "could not connect" in lowered
        or "connection timeout" in lowered
        or "connection is lost" in lowered
        or "server closed the connection unexpectedly" in lowered
    ):
        return "connection"
    return "db"


class SQLService:
    def __init__(
        self,
        sql_llm: Any,
        db: PostgresClient,
        logger: logging.Logger,
        row_limit: int = 100,
    ):
        self.sql_llm = sql_llm
        self.db = db
        self.logger = logger
        self.row_limit = row_limit

    def generate_sql(
        self,
        *,
        question: str,
        schema_context: str,
        allowed_tables: List[str],
        metadata_context: str = "",
        conversation_context: str = "",
    ) -> Dict[str, Any]:
        if not schema_context or not allowed_tables:
            msg = "Schema context is empty, cannot generate SQL."
            return {
                "sql_query": "",
                "sql_reasoning": "",
                "sql_error": msg,
                "sql_error_type": "schema_context",
                "sql_error_message": msg,
            }

        allowed_tables_text = ", ".join(allowed_tables)
        try:
            draft = self.sql_llm.invoke(
                [
                    SystemMessage(
                        content=SQL_GENERATOR_SYSTEM_PROMPT.format(
                            schema_text=schema_context,
                            allowed_tables=allowed_tables_text,
                            metadata_context=(
                                metadata_context.strip() or "No metadata hints."
                            ),
                            conversation_context=(
                                conversation_context.strip()
                                or "No prior conversation context."
                            ),
                            row_limit=self.row_limit,
                        )
                    ),
                    HumanMessage(content=question),
                ]
            )
            sql_query = normalize_sql(draft.sql)
            if not sql_query:
                msg = "SQL generation failed: model returned empty SQL."
                self.logger.error(msg)
                return {
                    "sql_query": "",
                    "sql_reasoning": "",
                    "sql_error": msg,
                    "sql_error_type": "generation",
                    "sql_error_message": "model returned empty SQL",
                }
            self.logger.info("Generated SQL draft successfully.")
            return {
                "sql_query": sql_query,
                "sql_reasoning": getattr(draft, "reasoning", ""),
                "sql_error": "",
                "sql_error_type": "",
                "sql_error_message": "",
            }
        except Exception as exc:
            msg = f"SQL generation failed: {exc}"
            self.logger.error(msg)
            return {
                "sql_query": "",
                "sql_reasoning": "",
                "sql_error": msg,
                "sql_error_type": "generation",
                "sql_error_message": str(exc),
            }

    def preflight_sql(
        self,
        *,
        sql_query: str,
        allowed_tables: List[str],
    ) -> Dict[str, Any]:
        if not allowed_tables:
            msg = "Allowed table context is empty; refusing to execute SQL."
            self.logger.warning(msg)
            return {
                "sql_rows": [],
                "sql_error": msg,
                "sql_error_type": "allowlist",
                "sql_error_message": msg,
            }

        guard_error = validate_readonly_sql(
            sql=sql_query,
            allowed_tables=allowed_tables,
        )
        if guard_error:
            err_type = classify_sql_error(guard_error)
            self.logger.warning("SQL rejected by guard: %s", guard_error)
            return {
                "sql_rows": [],
                "sql_error": guard_error,
                "sql_error_type": err_type,
                "sql_error_message": guard_error,
            }
        return {
            "sql_rows": [],
            "sql_error": "",
            "sql_error_type": "",
            "sql_error_message": "",
        }

    def execute_sql(
        self,
        *,
        sql_query: str,
        allowed_tables: List[str],
        skip_guard: bool = False,
    ) -> Dict[str, Any]:
        if not skip_guard:
            preflight = self.preflight_sql(
                sql_query=sql_query,
                allowed_tables=allowed_tables,
            )
            if preflight.get("sql_error"):
                return preflight

        try:
            rows = self.db.run_query(sql_query)
            self.logger.info("SQL executed successfully with %d rows.", len(rows))
            return {
                "sql_rows": rows,
                "sql_error": "",
                "sql_error_type": "",
                "sql_error_message": "",
            }
        except Exception as exc:
            msg = str(exc)
            err_type = classify_sql_error(msg)
            self.logger.error("SQL execution failed (%s): %s", err_type, msg)
            return {
                "sql_rows": [],
                "sql_error": msg,
                "sql_error_type": err_type,
                "sql_error_message": msg,
            }

    def repair_sql(
        self,
        *,
        question: str,
        failed_sql: str,
        sql_error: str,
        schema_context: str,
        allowed_tables: List[str],
        attempts: int,
        metadata_context: str = "",
        conversation_context: str = "",
    ) -> Dict[str, Any]:
        if not schema_context or not allowed_tables:
            msg = "Schema context is empty, cannot repair SQL."
            return {
                "sql_query": "",
                "last_failed_sql": failed_sql,
                "sql_reasoning": "",
                "attempts": attempts,
                "sql_error": msg,
                "sql_error_type": "schema_context",
                "sql_error_message": msg,
            }

        allowed_tables_text = ", ".join(allowed_tables)

        try:
            draft = self.sql_llm.invoke(
                [
                    SystemMessage(
                        content=SQL_REPAIR_SYSTEM_PROMPT.format(
                            schema_text=schema_context,
                            allowed_tables=allowed_tables_text,
                            metadata_context=(
                                metadata_context.strip() or "No metadata hints."
                            ),
                            conversation_context=(
                                conversation_context.strip()
                                or "No prior conversation context."
                            ),
                            row_limit=self.row_limit,
                        )
                    ),
                    HumanMessage(
                        content=(
                            f"User question:\n{question}\n\n"
                            f"Failed SQL:\n{failed_sql}\n\n"
                            f"Database error:\n{sql_error}"
                        )
                    ),
                ]
            )
            self.logger.info("SQL repaired on attempt=%d.", attempts)
            repaired_sql = normalize_sql(draft.sql)
            if not repaired_sql:
                msg = "SQL repair failed: model returned empty SQL."
                self.logger.error(msg)
                return {
                    "sql_query": "",
                    "last_failed_sql": failed_sql,
                    "sql_reasoning": "",
                    "attempts": attempts,
                    "sql_error": msg,
                    "sql_error_type": "repair",
                    "sql_error_message": "model returned empty SQL",
                }
            return {
                "sql_query": repaired_sql,
                "sql_reasoning": getattr(draft, "reasoning", ""),
                "attempts": attempts,
                "sql_error": "",
                "sql_error_type": "",
                "sql_error_message": "",
            }
        except Exception as exc:
            msg = f"SQL repair failed: {exc}"
            self.logger.error(msg)
            err_type = classify_sql_error(str(exc))
            return {
                "sql_query": "",
                "last_failed_sql": failed_sql,
                "sql_reasoning": "",
                "attempts": attempts,
                "sql_error": msg,
                "sql_error_type": err_type if err_type == "provider" else "repair",
                "sql_error_message": str(exc),
            }
