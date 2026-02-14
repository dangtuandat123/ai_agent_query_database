import logging
import re
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from ..db import PostgresClient
from ..prompts import SQL_GENERATOR_SYSTEM_PROMPT, SQL_REPAIR_SYSTEM_PROMPT
from ..sql_guard import normalize_sql, validate_readonly_sql


POSTGRES_URL_PASSWORD_PATTERN = re.compile(
    r"(?i)(postgres(?:ql)?://[^:\s/]+:)([^@/\s]+)(@)"
)
DSN_PASSWORD_PATTERN = re.compile(r"(?i)(password=)([^\s]+)")
API_KEY_PATTERN = re.compile(r"(?i)(api[_-]?key(?:\s*[:=]\s*|\s+))([^\s,;]+)")
BEARER_PATTERN = re.compile(r"(?i)(authorization:\s*bearer\s+)([^\s]+)")
SQL_START_PATTERN = re.compile(r"^(?:with|select)\b", re.IGNORECASE)


class SQLDraftOutput(BaseModel):
    sql: str = Field(..., description="A single valid PostgreSQL SELECT query")
    reasoning: str = Field("", description="Short explanation")


def _stringify_message_content(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    return str(content)


def redact_sensitive_text(text: str) -> str:
    redacted = POSTGRES_URL_PASSWORD_PATTERN.sub(r"\1***\3", text)
    redacted = DSN_PASSWORD_PATTERN.sub(r"\1***", redacted)
    redacted = API_KEY_PATTERN.sub(r"\1***", redacted)
    redacted = BEARER_PATTERN.sub(r"\1***", redacted)
    return redacted


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
        or "connection timed out" in lowered
        or "connection is lost" in lowered
        or "server closed the connection unexpectedly" in lowered
        or "could not translate host name" in lowered
        or "temporary failure in name resolution" in lowered
        or "name or service not known" in lowered
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
        raw_llm: Any | None = None,
    ):
        self.sql_llm = sql_llm
        self.raw_llm = raw_llm
        self.db = db
        self.logger = logger
        self.row_limit = row_limit
        self.sql_output_parser = PydanticOutputParser(pydantic_object=SQLDraftOutput)

    @staticmethod
    def _is_select_like_sql(sql: str) -> bool:
        return bool(SQL_START_PATTERN.match(sql.strip()))

    @staticmethod
    def _append_json_instructions(
        messages: List[Any], format_instructions: str
    ) -> List[Any]:
        if not messages:
            return messages
        first = messages[0]
        if isinstance(first, SystemMessage):
            enhanced = SystemMessage(
                content=(
                    f"{first.content}\n\n"
                    "Return valid JSON only.\n"
                    f"{format_instructions}"
                )
            )
            return [enhanced, *messages[1:]]
        return messages

    def _invoke_sql_draft_with_fallback(self, messages: List[Any]) -> Any:
        try:
            return self.sql_llm.invoke(messages)
        except Exception as structured_exc:
            err_type = classify_sql_error(str(structured_exc))
            if err_type in {"provider", "connection"}:
                raise

            self.logger.warning(
                "Structured SQL output failed; attempting parser fallback: %s",
                redact_sensitive_text(str(structured_exc)),
            )
            if self.raw_llm is None:
                raise

            fallback_messages = self._append_json_instructions(
                messages,
                self.sql_output_parser.get_format_instructions(),
            )
            raw_response = self.raw_llm.invoke(fallback_messages)
            raw_content = _stringify_message_content(raw_response)
            try:
                return self.sql_output_parser.parse(raw_content)
            except Exception:
                sql_candidate = normalize_sql(raw_content)
                if sql_candidate and self._is_select_like_sql(sql_candidate):
                    return SQLDraftOutput(
                        sql=sql_candidate,
                        reasoning="Fallback parsed from raw model output.",
                    )
                raise structured_exc

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
        messages = [
            SystemMessage(
                content=SQL_GENERATOR_SYSTEM_PROMPT.format(
                    schema_text=schema_context,
                    allowed_tables=allowed_tables_text,
                    metadata_context=(metadata_context.strip() or "No metadata hints."),
                    conversation_context=(
                        conversation_context.strip() or "No prior conversation context."
                    ),
                    row_limit=self.row_limit,
                )
            ),
            HumanMessage(content=question),
        ]
        try:
            draft = self._invoke_sql_draft_with_fallback(messages)
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
            raw_message = str(exc)
            err_type = classify_sql_error(raw_message)
            safe_message = redact_sensitive_text(raw_message)
            msg = f"SQL generation failed: {safe_message}"
            self.logger.error(msg)
            return {
                "sql_query": "",
                "sql_reasoning": "",
                "sql_error": msg,
                "sql_error_type": (
                    err_type if err_type in {"provider", "connection"} else "generation"
                ),
                "sql_error_message": safe_message,
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
            raw_message = str(exc)
            err_type = classify_sql_error(raw_message)
            safe_message = redact_sensitive_text(raw_message)
            self.logger.error("SQL execution failed (%s): %s", err_type, safe_message)
            return {
                "sql_rows": [],
                "sql_error": safe_message,
                "sql_error_type": err_type,
                "sql_error_message": safe_message,
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
        messages = [
            SystemMessage(
                content=SQL_REPAIR_SYSTEM_PROMPT.format(
                    schema_text=schema_context,
                    allowed_tables=allowed_tables_text,
                    metadata_context=(metadata_context.strip() or "No metadata hints."),
                    conversation_context=(
                        conversation_context.strip() or "No prior conversation context."
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

        try:
            draft = self._invoke_sql_draft_with_fallback(messages)
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
            raw_message = str(exc)
            err_type = classify_sql_error(raw_message)
            safe_message = redact_sensitive_text(raw_message)
            msg = f"SQL repair failed: {safe_message}"
            self.logger.error(msg)
            return {
                "sql_query": "",
                "last_failed_sql": failed_sql,
                "sql_reasoning": "",
                "attempts": attempts,
                "sql_error": msg,
                "sql_error_type": (
                    err_type if err_type in {"provider", "connection"} else "repair"
                ),
                "sql_error_message": safe_message,
            }
