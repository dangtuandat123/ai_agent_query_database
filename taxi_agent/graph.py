import json
import logging
from pathlib import Path
import threading
from typing import Any, Dict, Literal, Optional, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .config import Settings
from .db import PostgresClient
from .prompts import (
    ANSWER_SYSTEM_PROMPT,
    INTENT_SYSTEM_PROMPT,
    ROUTER_SYSTEM_PROMPT,
)
from .retrieval import SchemaRetriever, SchemaRetrieverConfig
from .services.language_service import (
    empty_question_message,
    error_after_retry_message,
    fallback_no_data_message,
    fallback_success_message,
    internal_error_message,
    unsupported_message,
)
from .services.metadata_service import MetadataContextService
from .services.schema_service import SchemaService
from .services.sql_service import SQLService
from .types import AgentResult, DashboardState


class RouteDecision(BaseModel):
    route: Literal["sql", "unsupported"] = Field(
        ..., description="Route name: sql or unsupported"
    )
    reason: str = Field(..., description="Short reason for route")


class SQLDraft(BaseModel):
    sql: str = Field(..., description="A single valid PostgreSQL SELECT query")
    reasoning: str = Field("", description="Short explanation")


class IntentDecision(BaseModel):
    intent: Literal["sql_query", "sql_followup", "unsupported"] = Field(
        ...,
        description="Task intent classification.",
    )
    reason: str = Field(..., description="Short reason for selected intent")


def _stringify_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    return str(content)


class TaxiDashboardAgent:
    def __init__(
        self,
        settings: Settings,
        *,
        db_client: Optional[PostgresClient] = None,
        llm: Optional[Any] = None,
        embedding_model: Optional[OpenAIEmbeddings] = None,
        schema_retriever: Optional[SchemaRetriever] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.settings = settings
        self.logger = logger or logging.getLogger(__name__)
        self.default_headers = self._build_openrouter_headers(settings)
        self.db_logger = self.logger.getChild("db")
        self.schema_logger = self.logger.getChild("schema")
        self.sql_logger = self.logger.getChild("sql")

        self.db = db_client or PostgresClient(
            dsn=settings.postgres_dsn,
            row_limit=settings.query_row_limit,
            query_timeout_ms=settings.query_timeout_ms,
            default_schema=settings.db_schema,
            connect_timeout_seconds=settings.db_connect_timeout_seconds,
            logger=self.db_logger,
        )

        self.llm = llm or self._build_llm(settings)
        self.embedding_model = embedding_model or self._build_embedding_model(settings)
        self.schema_retriever = schema_retriever or SchemaRetriever(
            embedding_model=self.embedding_model,
            config=SchemaRetrieverConfig(
                top_k_tables=settings.schema_top_k_tables,
                search_type=settings.schema_retriever_search_type,
                fetch_k=settings.schema_retriever_fetch_k,
            ),
        )

        self.router_llm = self.llm.with_structured_output(RouteDecision)
        self.intent_llm = self.llm.with_structured_output(IntentDecision)
        self.sql_llm = self.llm.with_structured_output(SQLDraft)

        self.schema_service = SchemaService(
            db=self.db,
            schema_retriever=self.schema_retriever,
            db_schema=settings.db_schema,
            max_columns_per_table=settings.schema_max_columns_per_table,
            context_max_chars=settings.schema_context_max_chars,
            full_context_max_chars=settings.schema_full_context_max_chars,
            top_k_tables=settings.schema_top_k_tables,
            cache_ttl_seconds=settings.schema_cache_ttl_seconds,
            logger=self.schema_logger,
        )
        self.sql_service = SQLService(
            sql_llm=self.sql_llm,
            db=self.db,
            logger=self.sql_logger,
            row_limit=settings.query_row_limit,
        )
        self.metadata_service = MetadataContextService(max_chars=3000)
        self._conversation_memory: Dict[str, Dict[str, str]] = {}
        self._max_memory_threads = settings.memory_max_threads
        self._memory_lock = threading.Lock()

        self.graph = self._build_graph()

    @staticmethod
    def _normalize_thread_id(thread_id: str) -> str:
        normalized = (thread_id or "").strip()
        return normalized or "default"

    @staticmethod
    def _truncate_prompt_piece(value: str, max_chars: int = 1200) -> str:
        text = value.strip()
        if len(text) <= max_chars:
            return text
        if max_chars <= 3:
            return text[:max_chars]
        return text[: max_chars - 3].rstrip() + "..."

    def _remember_success_turn(
        self,
        *,
        thread_id: str,
        question: str,
        sql_query: str,
        final_answer: str,
    ) -> None:
        with self._memory_lock:
            if thread_id in self._conversation_memory:
                # Move existing thread to the end (most recent).
                self._conversation_memory.pop(thread_id, None)
            elif len(self._conversation_memory) >= self._max_memory_threads:
                oldest_thread_id = next(iter(self._conversation_memory))
                self._conversation_memory.pop(oldest_thread_id, None)

            self._conversation_memory[thread_id] = {
                "question": question,
                "sql_query": sql_query,
                "final_answer": final_answer,
            }

    def _build_openrouter_headers(self, settings: Settings) -> Optional[Dict[str, str]]:
        default_headers: Dict[str, str] = {}
        if settings.openrouter_site_url:
            default_headers["HTTP-Referer"] = settings.openrouter_site_url
        if settings.openrouter_app_name:
            default_headers["X-Title"] = settings.openrouter_app_name
        return default_headers or None

    def _build_llm(self, settings: Settings) -> ChatOpenAI:
        return ChatOpenAI(
            model=settings.openrouter_model,
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            temperature=0,
            default_headers=self.default_headers,
        )

    def _build_embedding_model(self, settings: Settings) -> Optional[OpenAIEmbeddings]:
        if not settings.enable_schema_embeddings:
            return None

        try:
            return OpenAIEmbeddings(
                model=settings.openrouter_embedding_model,
                api_key=settings.openrouter_api_key,
                base_url=settings.openrouter_base_url,
                default_headers=self.default_headers,
                tiktoken_enabled=False,
                check_embedding_ctx_length=False,
            )
        except Exception as exc:
            self.logger.warning(
                "Failed to initialize embedding model; fallback to BM25-only retrieval: %s",
                exc,
            )
            return None

    def _prepare_schema_context(self, state: DashboardState) -> DashboardState:
        question = state["question"]
        schema_result = self.schema_service.build_for_question(question)
        return {
            "schema_error": schema_result.schema_error,
            "schema_overview": schema_result.schema_overview,
            "schema_context": schema_result.schema_context,
            "schema_context_full": schema_result.schema_context_full,
            "allowed_tables": schema_result.allowed_tables,
            "all_allowed_tables": schema_result.all_allowed_tables,
        }

    def _route_question(self, state: DashboardState) -> DashboardState:
        if state.get("schema_error"):
            return {
                "route": "unsupported",
                "route_reason": state["schema_error"],
                "attempts": state.get("attempts", 0),
            }

        question = state["question"]
        schema_overview = state.get("schema_overview", "No schema overview available.")
        try:
            decision = self.router_llm.invoke(
                [
                    SystemMessage(
                        content=ROUTER_SYSTEM_PROMPT.format(
                            schema_overview=schema_overview
                        )
                    ),
                    HumanMessage(content=question),
                ]
            )
            self.logger.info("Route decision=%s", decision.route)
            return {
                "route": decision.route,
                "route_reason": decision.reason,
                "attempts": state.get("attempts", 0),
            }
        except Exception as exc:
            self.logger.error("Router error: %s", exc)
            return {
                "route": "unsupported",
                "route_reason": f"Router error: {exc}",
                "attempts": state.get("attempts", 0),
            }

    def _build_metadata_context(self, state: DashboardState) -> DashboardState:
        context = self.metadata_service.build(
            question=state["question"],
            allowed_tables=state.get("allowed_tables", []),
            schema_context=state.get("schema_context", ""),
        )
        return {"metadata_context": context}

    def _build_previous_context_text(self, state: DashboardState) -> str:
        previous_question = self._truncate_prompt_piece(
            state.get("previous_question", "")
        )
        previous_sql_query = self._truncate_prompt_piece(
            state.get("previous_sql_query", "")
        )
        previous_final_answer = self._truncate_prompt_piece(
            state.get("previous_final_answer", ""),
            max_chars=800,
        )
        if not previous_question and not previous_sql_query and not previous_final_answer:
            return "No previous conversation context."
        return (
            f"Previous question: {previous_question or 'n/a'}\n"
            f"Previous SQL: {previous_sql_query or 'n/a'}\n"
            f"Previous answer summary: {previous_final_answer or 'n/a'}"
        )

    def _determine_intent(self, state: DashboardState) -> DashboardState:
        if state.get("route") != "sql":
            return {
                "intent": "unsupported",
                "intent_reason": state.get("route_reason", "Unsupported route."),
            }

        question = state["question"]
        previous_context = self._build_previous_context_text(state)
        try:
            decision = self.intent_llm.invoke(
                [
                    SystemMessage(
                        content=INTENT_SYSTEM_PROMPT.format(
                            question=question,
                            previous_context=previous_context,
                        )
                    ),
                    HumanMessage(content=question),
                ]
            )
            self.logger.info("Intent decision=%s", decision.intent)
            return {"intent": decision.intent, "intent_reason": decision.reason}
        except Exception as exc:
            self.logger.warning("Intent router failed, fallback to sql_query: %s", exc)
            lowered = question.lower()
            followup_hints = (
                "còn",
                "so sánh",
                "compare",
                "what about",
                "how about",
                "tiếp",
                "again",
                "same filter",
            )
            has_previous = bool(state.get("previous_question", "").strip())
            is_followup = has_previous and any(hint in lowered for hint in followup_hints)
            return {
                "intent": "sql_followup" if is_followup else "sql_query",
                "intent_reason": "Heuristic fallback",
            }

    def _generate_sql(self, state: DashboardState) -> DashboardState:
        self.logger.info("Generating SQL.")
        conversation_context = self._build_previous_context_text(state)
        return self.sql_service.generate_sql(
            question=state["question"],
            schema_context=state.get("schema_context", ""),
            allowed_tables=state.get("allowed_tables", []),
            metadata_context=state.get("metadata_context", ""),
            conversation_context=conversation_context,
        )

    def _security_check(self, state: DashboardState) -> DashboardState:
        existing_error = state.get("sql_error", "")
        sql_query = state.get("sql_query", "")
        if existing_error and not sql_query:
            return {
                "sql_error": existing_error,
                "sql_error_type": state.get("sql_error_type", "generation"),
                "sql_error_message": state.get("sql_error_message", existing_error),
            }

        self.logger.info("Security preflight for SQL.")
        return self.sql_service.preflight_sql(
            sql_query=sql_query,
            allowed_tables=state.get("allowed_tables", []),
        )

    def _execute_sql(self, state: DashboardState) -> DashboardState:
        existing_error = state.get("sql_error", "")
        sql_query = state.get("sql_query", "")
        if existing_error and not sql_query:
            self.logger.warning(
                "Skipping SQL execution because SQL generation already failed."
            )
            return {
                "sql_rows": [],
                "sql_error": existing_error,
                "sql_error_type": state.get("sql_error_type", "generation"),
                "sql_error_message": state.get(
                    "sql_error_message",
                    existing_error,
                ),
            }

        self.logger.info("Executing SQL.")
        return self.sql_service.execute_sql(
            sql_query=sql_query,
            allowed_tables=state.get("allowed_tables", []),
            skip_guard=True,
        )

    def _repair_sql(self, state: DashboardState) -> DashboardState:
        attempts = state.get("attempts", 0) + 1
        sql_error = state.get("sql_error", "Unknown SQL error")
        error_type = state.get("sql_error_type", "")
        schema_context = state.get("schema_context", "")
        allowed_tables = state.get("allowed_tables", [])
        failed_sql = state.get("sql_query", "")

        should_expand = error_type == "allowlist" or (
            "outside allowed schema context" in sql_error.lower()
        )
        if should_expand:
            schema_context = state.get("schema_context_full", schema_context)
            allowed_tables = state.get("all_allowed_tables", allowed_tables)
            self.logger.info(
                "Repair attempt=%d with expanded full schema context.", attempts
            )
        else:
            self.logger.info("Repair attempt=%d with retrieved schema context.", attempts)

        if not failed_sql.strip():
            self.logger.info(
                "Repair attempt=%d has no failed SQL; regenerating SQL.",
                attempts,
            )
            conversation_context = self._build_previous_context_text(state)
            result = self.sql_service.generate_sql(
                question=state["question"],
                schema_context=schema_context,
                allowed_tables=allowed_tables,
                metadata_context=state.get("metadata_context", ""),
                conversation_context=conversation_context,
            )
            result["attempts"] = attempts
            result["allowed_tables"] = allowed_tables
            return result

        conversation_context = self._build_previous_context_text(state)
        result = self.sql_service.repair_sql(
            question=state["question"],
            failed_sql=failed_sql,
            sql_error=sql_error,
            schema_context=schema_context,
            allowed_tables=allowed_tables,
            attempts=attempts,
            metadata_context=state.get("metadata_context", ""),
            conversation_context=conversation_context,
        )
        result["allowed_tables"] = allowed_tables
        return result

    def _answer_user(self, state: DashboardState) -> DashboardState:
        question = state["question"]
        sql_query = state.get("sql_query", "")
        rows = state.get("sql_rows", [])
        rows_preview = rows[:20]

        try:
            response = self.llm.invoke(
                [
                    SystemMessage(content=ANSWER_SYSTEM_PROMPT),
                    HumanMessage(
                        content=(
                            f"User question:\n{question}\n\n"
                            f"Executed SQL:\n{sql_query}\n\n"
                            f"Rows(JSON):\n{json.dumps(rows_preview, default=str)}"
                        )
                    ),
                ]
            )
            return {"final_answer": _stringify_content(response.content)}
        except Exception as exc:
            self.logger.warning("Answer LLM failed, using fallback message: %s", exc)
            if not rows:
                return {"final_answer": fallback_no_data_message(question)}
            return {"final_answer": fallback_success_message(question, len(rows))}

    def _unsupported_answer(self, state: DashboardState) -> DashboardState:
        reason = (
            state.get("intent_reason")
            or state.get("route_reason")
            or "Unsupported question for current schema."
        )
        question = state.get("question", "")
        return {"final_answer": unsupported_message(question, reason)}

    def _error_answer(self, state: DashboardState) -> DashboardState:
        question = state["question"]
        sql_query = state.get("sql_query") or state.get("last_failed_sql", "")
        sql_error = state.get("sql_error", "Unknown error")
        return {
            "final_answer": error_after_retry_message(question, sql_query, sql_error),
        }

    def _after_route(self, state: DashboardState) -> str:
        if state.get("route") == "sql":
            return "sql_path"
        return "unsupported_path"

    def _after_intent(self, state: DashboardState) -> str:
        intent = state.get("intent")
        if intent in {"sql_query", "sql_followup"}:
            return "sql_path"
        return "unsupported_path"

    def _after_security(self, state: DashboardState) -> str:
        sql_error = state.get("sql_error")
        if not sql_error:
            return "execute"
        attempts = state.get("attempts", 0)
        if attempts < self.settings.max_sql_retries:
            return "retry"
        return "failed"

    def _after_execute(self, state: DashboardState) -> str:
        sql_error = state.get("sql_error")
        if not sql_error:
            return "success"

        attempts = state.get("attempts", 0)
        if attempts < self.settings.max_sql_retries:
            return "retry"
        return "failed"

    def _build_graph(self):
        builder = StateGraph(DashboardState)

        builder.add_node("prepare_schema_context", self._prepare_schema_context)
        builder.add_node("build_metadata_context", self._build_metadata_context)
        builder.add_node("route_question", self._route_question)
        builder.add_node("determine_intent", self._determine_intent)
        builder.add_node("generate_sql", self._generate_sql)
        builder.add_node("security_check", self._security_check)
        builder.add_node("execute_sql", self._execute_sql)
        builder.add_node("repair_sql", self._repair_sql)
        builder.add_node("answer_user", self._answer_user)
        builder.add_node("unsupported_answer", self._unsupported_answer)
        builder.add_node("error_answer", self._error_answer)

        builder.add_edge(START, "prepare_schema_context")
        builder.add_edge("prepare_schema_context", "build_metadata_context")
        builder.add_edge("build_metadata_context", "route_question")
        builder.add_conditional_edges(
            "route_question",
            self._after_route,
            {
                "sql_path": "determine_intent",
                "unsupported_path": "unsupported_answer",
            },
        )
        builder.add_conditional_edges(
            "determine_intent",
            self._after_intent,
            {
                "sql_path": "generate_sql",
                "unsupported_path": "unsupported_answer",
            },
        )
        builder.add_edge("generate_sql", "security_check")
        builder.add_conditional_edges(
            "security_check",
            self._after_security,
            {
                "execute": "execute_sql",
                "retry": "repair_sql",
                "failed": "error_answer",
            },
        )
        builder.add_conditional_edges(
            "execute_sql",
            self._after_execute,
            {
                "success": "answer_user",
                "retry": "repair_sql",
                "failed": "error_answer",
            },
        )
        builder.add_edge("repair_sql", "security_check")
        builder.add_edge("answer_user", END)
        builder.add_edge("unsupported_answer", END)
        builder.add_edge("error_answer", END)

        return builder.compile()

    def ask(self, question: str, thread_id: str = "default") -> AgentResult:
        normalized_thread_id = self._normalize_thread_id(thread_id)
        clean_question = question.strip()
        if not clean_question:
            self.logger.warning("Received empty question.")
            return cast(
                AgentResult,
                {
                    "question": "",
                    "thread_id": normalized_thread_id,
                    "route": "unsupported",
                    "route_reason": "Empty question.",
                    "attempts": 0,
                    "final_answer": empty_question_message(),
                },
            )

        with self._memory_lock:
            previous_turn = dict(self._conversation_memory.get(normalized_thread_id, {}))
        initial_state: DashboardState = {
            "question": clean_question,
            "thread_id": normalized_thread_id,
            "attempts": 0,
            "previous_question": previous_turn.get("question", ""),
            "previous_sql_query": previous_turn.get("sql_query", ""),
            "previous_final_answer": previous_turn.get("final_answer", ""),
        }
        try:
            raw_result = self.graph.invoke(initial_state)
            result = cast(DashboardState, dict(raw_result))
            result.setdefault("thread_id", normalized_thread_id)
            if (
                result.get("route") == "sql"
                and not result.get("sql_error")
                and result.get("sql_query")
            ):
                self._remember_success_turn(
                    thread_id=normalized_thread_id,
                    question=clean_question,
                    sql_query=str(result.get("sql_query", "")),
                    final_answer=str(result.get("final_answer", "")),
                )
            return cast(AgentResult, result)
        except Exception as exc:
            self.logger.exception("Graph execution failed: %s", exc)
            message = str(exc)
            return cast(
                AgentResult,
                {
                    "question": clean_question,
                    "thread_id": normalized_thread_id,
                    "route": "unsupported",
                    "route_reason": f"Internal graph error: {message}",
                    "attempts": 0,
                    "sql_error": message,
                    "sql_error_type": "internal",
                    "sql_error_message": message,
                    "final_answer": internal_error_message(clean_question),
                },
            )

    def get_workflow_mermaid(self) -> str:
        return self.graph.get_graph().draw_mermaid()

    def save_workflow_mermaid(self, file_path: str = "agent_workflow.mmd") -> str:
        mermaid = self.get_workflow_mermaid()
        path = Path(file_path)
        path.write_text(mermaid, encoding="utf-8")
        return str(path.resolve())

    def clear_thread_memory(self, thread_id: Optional[str] = None) -> None:
        with self._memory_lock:
            if thread_id is None:
                self._conversation_memory.clear()
                return
            normalized_thread_id = self._normalize_thread_id(thread_id)
            self._conversation_memory.pop(normalized_thread_id, None)
