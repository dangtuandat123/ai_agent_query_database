import logging
from typing import Any, Dict, List, Tuple

import psycopg
from langchain_community.utilities import SQLDatabase
from psycopg.rows import dict_row

from .schema import ColumnSchema, TableSchema
from .sql_guard import normalize_sql, validate_readonly_sql


class PostgresClient:
    def __init__(
        self,
        dsn: str,
        row_limit: int = 100,
        query_timeout_ms: int = 30000,
        default_schema: str = "public",
        logger: logging.Logger | None = None,
    ):
        self.dsn = dsn
        self.row_limit = row_limit
        self.query_timeout_ms = query_timeout_ms
        self.default_schema = default_schema
        self.logger = logger or logging.getLogger(__name__)
        self._sql_db_by_schema: Dict[str, SQLDatabase] = {}

    def _get_sql_database(self, table_schema: str) -> SQLDatabase:
        cached = self._sql_db_by_schema.get(table_schema)
        if cached is not None:
            return cached

        sql_db = SQLDatabase.from_uri(
            self.dsn,
            schema=table_schema,
            sample_rows_in_table_info=0,
        )
        self._sql_db_by_schema[table_schema] = sql_db
        return sql_db

    def run_query(self, sql: str) -> List[Dict[str, Any]]:
        with psycopg.connect(self.dsn) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                # Defense-in-depth: even if SQL guard misses something,
                # this transaction is forced read-only.
                cur.execute("SET TRANSACTION READ ONLY")
                # Pin search_path to the configured analytics schema to avoid
                # accidentally reading same-named tables from another schema.
                cur.execute(
                    "SELECT set_config('search_path', %s, true)",
                    (self.default_schema,),
                )
                # Apply timeout only for this transaction.
                cur.execute(
                    "SELECT set_config('statement_timeout', %s, true)",
                    (str(self.query_timeout_ms),),
                )
                cur.execute(sql)
                rows = cur.fetchmany(self.row_limit)
                return list(rows)

    def check_connection(self) -> Tuple[bool, str]:
        try:
            with psycopg.connect(self.dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
            return True, "ok"
        except Exception as exc:  # pragma: no cover
            return False, str(exc)

    def get_table_schemas(self, table_schema: str = "public") -> List[TableSchema]:
        query = """
            SELECT
                table_schema,
                table_name,
                column_name,
                data_type,
                ordinal_position
            FROM information_schema.columns
            WHERE table_schema = %s
            ORDER BY table_name, ordinal_position
        """
        with psycopg.connect(self.dsn) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(query, (table_schema,))
                rows = cur.fetchall()

        grouped: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            key = f"{row['table_schema']}.{row['table_name']}"
            if key not in grouped:
                grouped[key] = {
                    "table_schema": row["table_schema"],
                    "table_name": row["table_name"],
                    "columns": [],
                }
            grouped[key]["columns"].append(
                ColumnSchema(
                    column_name=row["column_name"],
                    data_type=row["data_type"],
                    ordinal_position=int(row["ordinal_position"]),
                )
            )

        tables = [
            TableSchema(
                table_schema=item["table_schema"],
                table_name=item["table_name"],
                columns=item["columns"],
            )
            for item in grouped.values()
        ]
        tables.sort(key=lambda t: (t.table_schema, t.table_name))
        return tables

    def get_table_info(
        self,
        table_names: List[str],
        table_schema: str = "public",
    ) -> str:
        db = self._get_sql_database(table_schema)
        requested = sorted({name.strip() for name in table_names if name.strip()})
        if not requested:
            return ""
        return db.get_table_info(table_names=requested)


__all__ = [
    "PostgresClient",
    "normalize_sql",
    "validate_readonly_sql",
]
