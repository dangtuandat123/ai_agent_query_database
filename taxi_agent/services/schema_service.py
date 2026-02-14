import logging
import threading
from dataclasses import dataclass
from time import monotonic
from typing import List, Sequence

from ..db import PostgresClient
from ..redaction import redact_sensitive_text
from ..retrieval import SchemaRetriever
from ..schema import TableSchema, build_schema_context, build_schema_overview


@dataclass(frozen=True)
class SchemaContextResult:
    schema_error: str
    schema_overview: str
    schema_context: str
    schema_context_full: str
    allowed_tables: List[str]
    all_allowed_tables: List[str]


class SchemaService:
    def __init__(
        self,
        db: PostgresClient,
        schema_retriever: SchemaRetriever,
        *,
        db_schema: str,
        max_columns_per_table: int,
        context_max_chars: int,
        full_context_max_chars: int,
        top_k_tables: int,
        cache_ttl_seconds: int,
        logger: logging.Logger,
    ):
        self.db = db
        self.schema_retriever = schema_retriever
        self.db_schema = db_schema
        self.max_columns_per_table = max_columns_per_table
        self.context_max_chars = context_max_chars
        self.full_context_max_chars = full_context_max_chars
        self.top_k_tables = top_k_tables
        self.cache_ttl_seconds = cache_ttl_seconds
        self.logger = logger

        self._cached_tables: List[TableSchema] = []
        self._cache_loaded: bool = False
        self._cache_expiry: float = 0.0
        self._cache_lock = threading.RLock()
        self._cached_schema_overview: str = "No schema overview available."
        self._cached_all_allowed_tables: List[str] = []
        self._cached_full_context: str = ""

    @staticmethod
    def _truncate_text(value: str, max_chars: int) -> str:
        if max_chars <= 0 or len(value) <= max_chars:
            return value
        if max_chars <= 3:
            return value[:max_chars]
        return value[: max_chars - 3].rstrip() + "..."

    def _build_context_with_langchain(
        self,
        tables: Sequence[TableSchema],
        max_chars: int,
    ) -> str:
        get_table_info = getattr(self.db, "get_table_info", None)
        if not callable(get_table_info):
            return ""

        try:
            table_names = [table.table_name for table in tables]
            context = get_table_info(table_names, table_schema=self.db_schema)
            return self._truncate_text(context, max_chars)
        except Exception as exc:
            safe_message = redact_sensitive_text(str(exc))
            self.logger.warning(
                "LangChain SQLDatabase table info unavailable; fallback to custom context: %s",
                safe_message,
            )
            return ""

    def _build_context_with_fallback(
        self,
        tables: Sequence[TableSchema],
        max_chars: int,
    ) -> str:
        library_context = self._build_context_with_langchain(tables, max_chars)
        if library_context:
            return library_context
        return build_schema_context(
            tables=tables,
            max_columns_per_table=self.max_columns_per_table,
            max_chars=max_chars,
        )

    def _is_cache_valid(self) -> bool:
        return self._cache_loaded and monotonic() < self._cache_expiry

    def _load_all_tables(self) -> List[TableSchema]:
        with self._cache_lock:
            if self._is_cache_valid():
                self.logger.info("Schema cache hit (%d tables).", len(self._cached_tables))
                return list(self._cached_tables)

            self.logger.info("Schema cache miss; loading from PostgreSQL.")
            tables = self.db.get_table_schemas(table_schema=self.db_schema)
            self._cached_tables = list(tables)
            self._cache_loaded = True
            self._cache_expiry = monotonic() + self.cache_ttl_seconds
            self._cached_schema_overview = build_schema_overview(tables=self._cached_tables)
            self._cached_all_allowed_tables = self._build_allowlist(self._cached_tables)
            self._cached_full_context = ""
            self.logger.info(
                "Loaded %d tables for schema '%s'.",
                len(self._cached_tables),
                self.db_schema,
            )
            return list(self._cached_tables)

    def invalidate_cache(self) -> None:
        with self._cache_lock:
            self._cached_tables = []
            self._cache_loaded = False
            self._cache_expiry = 0.0
            self._cached_schema_overview = "No schema overview available."
            self._cached_all_allowed_tables = []
            self._cached_full_context = ""

    @staticmethod
    def _build_allowlist(tables: Sequence[TableSchema]) -> List[str]:
        allowed: List[str] = []
        for table in tables:
            allowed.append(table.table_name.lower())
            allowed.append(table.full_name.lower())
        return sorted(set(allowed))

    def _get_cached_schema_overview(self) -> str:
        with self._cache_lock:
            return self._cached_schema_overview

    def _get_cached_all_allowed_tables(self) -> List[str]:
        with self._cache_lock:
            return list(self._cached_all_allowed_tables)

    def _get_full_schema_context(self, all_tables: Sequence[TableSchema]) -> str:
        with self._cache_lock:
            if self._cached_full_context:
                return self._cached_full_context

            self._cached_full_context = self._build_context_with_fallback(
                tables=all_tables,
                max_chars=self.full_context_max_chars,
            )
            return self._cached_full_context

    def build_for_question(self, question: str) -> SchemaContextResult:
        try:
            all_tables = self._load_all_tables()
        except Exception as exc:
            safe_message = redact_sensitive_text(str(exc))
            self.logger.error("Failed to load schema from PostgreSQL: %s", safe_message)
            return SchemaContextResult(
                schema_error=f"Cannot read schema from PostgreSQL: {safe_message}",
                schema_overview="No schema overview available.",
                schema_context="No schema context available.",
                schema_context_full="No schema context available.",
                allowed_tables=[],
                all_allowed_tables=[],
            )

        if not all_tables:
            return SchemaContextResult(
                schema_error="No tables found in configured DB schema.",
                schema_overview="No tables found.",
                schema_context="No schema context available.",
                schema_context_full="No schema context available.",
                allowed_tables=[],
                all_allowed_tables=[],
            )

        try:
            self.schema_retriever.refresh(all_tables)
            relevant_tables = self.schema_retriever.retrieve_tables(question)
            if not relevant_tables:
                self.logger.warning(
                    "Schema retriever returned no tables; fallback to top-k tables."
                )
                relevant_tables = all_tables[: self.top_k_tables]
            self.logger.info(
                "Schema retrieval selected %d/%d tables.",
                len(relevant_tables),
                len(all_tables),
            )
        except Exception as exc:
            safe_message = redact_sensitive_text(str(exc))
            self.logger.warning(
                "Schema retrieval failed, fallback to top-k tables: %s",
                safe_message,
            )
            relevant_tables = all_tables[: self.top_k_tables]

        allowed_tables = self._build_allowlist(relevant_tables)
        all_allowed_tables = self._get_cached_all_allowed_tables()
        if not all_allowed_tables:
            all_allowed_tables = self._build_allowlist(all_tables)

        schema_context = self._build_context_with_fallback(
            tables=relevant_tables,
            max_chars=self.context_max_chars,
        )
        schema_context_full = self._get_full_schema_context(all_tables)
        schema_overview = self._get_cached_schema_overview()

        return SchemaContextResult(
            schema_error="",
            schema_overview=schema_overview,
            schema_context=schema_context,
            schema_context_full=schema_context_full,
            allowed_tables=allowed_tables,
            all_allowed_tables=all_allowed_tables,
        )
