import re
from typing import List


TOKEN_PATTERN = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")
DATE_PATTERN = re.compile(r"\b\d{4}-\d{2}(?:-\d{2})?\b")
QUARTER_PATTERN = re.compile(r"\bq[1-4]\s*20\d{2}\b", re.IGNORECASE)
YEAR_MONTH_PATTERN = re.compile(r"\b(19|20)\d{2}[/-](0?[1-9]|1[0-2])\b")
QUOTED_VALUE_PATTERN = re.compile(r"[\"']([^\"']{1,80})[\"']")


def _tokenize(text: str) -> List[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def _truncate(value: str, max_chars: int) -> str:
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    if max_chars <= 3:
        return value[:max_chars]
    return value[: max_chars - 3].rstrip() + "..."


class MetadataContextService:
    """Build lightweight context hints before SQL generation."""

    def __init__(self, max_chars: int = 2000):
        self.max_chars = max_chars

    def build(
        self,
        *,
        question: str,
        allowed_tables: List[str],
        schema_context: str,
    ) -> str:
        question_lower = question.lower()
        question_tokens = set(_tokenize(question_lower))

        matched_tables: List[str] = []
        for table_name in allowed_tables:
            normalized = table_name.lower()
            short_name = normalized.split(".")[-1]
            if normalized in question_lower or short_name in question_tokens:
                matched_tables.append(normalized)

        schema_tokens = set(_tokenize(schema_context.lower()))
        likely_columns = sorted(
            {
                token
                for token in question_tokens
                if token in schema_tokens and token not in {"select", "from", "where"}
            }
        )[:20]

        detected_values = sorted(
            {
                *DATE_PATTERN.findall(question),
                *QUARTER_PATTERN.findall(question),
                *YEAR_MONTH_PATTERN.findall(question),
                *[item.strip() for item in QUOTED_VALUE_PATTERN.findall(question)],
            }
        )

        lines: List[str] = ["Metadata hints (pre-SQL):"]
        if matched_tables:
            lines.append("- matched tables: " + ", ".join(sorted(set(matched_tables))[:20]))
        else:
            lines.append("- matched tables: none (fallback to schema retrieval context)")

        if likely_columns:
            lines.append("- likely columns: " + ", ".join(likely_columns))
        else:
            lines.append("- likely columns: none (infer from schema context)")

        if detected_values:
            lines.append("- detected filter values: " + ", ".join(detected_values[:20]))
        else:
            lines.append("- detected filter values: none")

        lines.append("- policy: read-only analytics, single statement, allowed-table only")
        return _truncate("\n".join(lines), self.max_chars)

