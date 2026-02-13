from dataclasses import dataclass
from typing import List, Sequence


@dataclass(frozen=True)
class ColumnSchema:
    column_name: str
    data_type: str
    ordinal_position: int


@dataclass(frozen=True)
class TableSchema:
    table_schema: str
    table_name: str
    columns: List[ColumnSchema]

    @property
    def full_name(self) -> str:
        return f"{self.table_schema}.{self.table_name}"


def build_schema_overview(tables: Sequence[TableSchema], max_tables: int = 50) -> str:
    if not tables:
        return "No tables found."

    names = [t.full_name for t in tables[:max_tables]]
    overview = ", ".join(names)
    if len(tables) > max_tables:
        overview += f", ... (+{len(tables) - max_tables} tables)"
    return overview


def build_schema_context(
    tables: Sequence[TableSchema],
    max_columns_per_table: int,
    max_chars: int,
) -> str:
    if not tables:
        return "No schema available."

    lines = []
    for table in tables:
        lines.append(f"Table: {table.full_name}")
        lines.append("Columns:")
        for col in table.columns[:max_columns_per_table]:
            lines.append(f"- {col.column_name} ({col.data_type})")
        lines.append("")

    text = "\n".join(lines).strip()
    if max_chars > 0 and len(text) > max_chars:
        if max_chars <= 3:
            return text[:max_chars]
        return text[: max_chars - 3].rstrip() + "..."
    return text
