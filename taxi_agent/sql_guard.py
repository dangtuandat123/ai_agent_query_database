import re
from typing import Any, List, Optional

import sqlparse
from sqlparse import tokens as T
from sqlparse.sql import Function, Identifier, IdentifierList, Parenthesis, TokenList


FORBIDDEN_SQL_PATTERN = re.compile(
    r"\b(insert|update|delete|drop|alter|truncate|grant|revoke|copy|merge|call|execute|create)\b",
    re.IGNORECASE,
)
SELECT_INTO_PATTERN = re.compile(
    r"\bselect\b[\s\S]*?\binto\b\s+(?:temp(?:orary)?\s+|unlogged\s+|table\s+)?"
    r"(?:\"?[a-zA-Z_][\w$]*\"?(?:\s*\.\s*\"?[a-zA-Z_][\w$]*\"?)?)",
    re.IGNORECASE,
)
LOCKING_PATTERN = re.compile(
    r"\bfor\s+(update|share|no\s+key\s+update|key\s+share)\b",
    re.IGNORECASE,
)
SQL_FENCE_PATTERN = re.compile(
    r"```(?:sql)?\s*([\s\S]*?)\s*```",
    re.IGNORECASE,
)
SQL_PREFIX_PATTERN = re.compile(r"^(?:sqlquery|sql)\s*:\s*", re.IGNORECASE)
SQL_START_PATTERN = re.compile(r"\b(select|with)\b", re.IGNORECASE)


def sanitize_sql(sql: str) -> str:
    cleaned = sql.strip()
    fence_match = SQL_FENCE_PATTERN.search(cleaned)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    cleaned = SQL_PREFIX_PATTERN.sub("", cleaned).strip()
    start_match = SQL_START_PATTERN.search(cleaned)
    if start_match and start_match.start() > 0:
        cleaned = cleaned[start_match.start() :].strip()
    return cleaned


def normalize_sql(sql: str) -> str:
    return sanitize_sql(sql).rstrip(";")


def _extract_cte_names(sql: str) -> set[str]:
    cte_names: set[str] = set()
    for statement in sqlparse.parse(sql):
        tokens = [t for t in statement.tokens if not t.is_whitespace]
        if not tokens:
            continue
        if str(tokens[0].value).upper() != "WITH":
            continue

        for token in tokens[1:]:
            # WITH RECURSIVE ...
            if (
                token.ttype in T.Keyword
                and str(token.value).upper() == "RECURSIVE"
            ):
                continue

            if isinstance(token, IdentifierList):
                for ident in token.get_identifiers():
                    name = ident.get_name() or ident.get_real_name()
                    if name:
                        cte_names.add(name.strip('"').lower())
                break

            if isinstance(token, Identifier):
                name = token.get_name() or token.get_real_name()
                if name:
                    cte_names.add(name.strip('"').lower())
                break

            if token.ttype in T.Keyword and str(token.value).upper() == "SELECT":
                # End of CTE header section.
                break
    return cte_names


def _extract_referenced_tables(sql: str) -> set[str]:
    def normalize_identifier(raw: str) -> str:
        return raw.strip().strip('"').lower()

    def extract_from_identifier(identifier: Identifier) -> set[str]:
        result = set()
        # Derived tables (FROM (SELECT ...) alias) should not be treated as
        # base tables in allowlist validation.
        has_subquery = any(
            isinstance(tok, Parenthesis) and "select" in tok.value.lower()
            for tok in identifier.tokens
        )
        if has_subquery:
            return result
        name = identifier.get_real_name() or identifier.get_name()
        parent = identifier.get_parent_name()
        if name:
            result.add(normalize_identifier(name))
            if parent:
                result.add(f"{normalize_identifier(parent)}.{normalize_identifier(name)}")
        return result

    def extract_table_targets(token: Any) -> set[str]:
        result = set()
        if token is None:
            return result
        if isinstance(token, IdentifierList):
            for ident in token.get_identifiers():
                if isinstance(ident, Identifier):
                    result.update(extract_from_identifier(ident))
        elif isinstance(token, Identifier):
            result.update(extract_from_identifier(token))
        elif getattr(token, "is_group", False):
            return result
        else:
            value = str(getattr(token, "value", "")).strip()
            if value and value not in {"(", ")"}:
                clean = normalize_identifier(value.split()[0])
                if clean:
                    result.add(clean)
        return result

    def walk_tokenlist(token_list: TokenList) -> set[str]:
        refs = set()
        tokens = [t for t in token_list.tokens if not t.is_whitespace]
        i = 0
        while i < len(tokens):
            token = tokens[i]
            value_upper = str(getattr(token, "value", "")).upper()

            if value_upper == "FROM" or "JOIN" in value_upper:
                next_token = tokens[i + 1] if i + 1 < len(tokens) else None
                refs.update(extract_table_targets(next_token))

            # Recursively scan nested subqueries but skip SQL function arguments
            # (e.g. EXTRACT(MONTH FROM pickup_datetime)).
            if token.is_group and not isinstance(token, Function):
                refs.update(walk_tokenlist(token))
            i += 1
        return refs

    refs = set()
    for statement in sqlparse.parse(sql):
        refs.update(walk_tokenlist(statement))

    expanded = set()
    for ref in refs:
        expanded.add(ref)
        if "." in ref:
            expanded.add(ref.split(".")[-1])
    return expanded


def _contains_token_type(ttype: Any, parent: Any) -> bool:
    while ttype is not None:
        if ttype == parent:
            return True
        ttype = ttype.parent
    return False


def _build_keyword_stream(sql: str) -> str:
    words: List[str] = []
    for statement in sqlparse.parse(sql):
        for token in statement.flatten():
            ttype = token.ttype
            if ttype is None:
                continue
            if _contains_token_type(ttype, T.Literal.String):
                continue
            if _contains_token_type(ttype, T.Comment):
                continue
            if (
                _contains_token_type(ttype, T.Keyword)
                or _contains_token_type(ttype, T.DML)
                or _contains_token_type(ttype, T.DDL)
            ):
                value = token.value.strip().lower()
                if value:
                    words.append(value)
    return " ".join(words)


def validate_readonly_sql(
    sql: str,
    allowed_tables: Optional[List[str]] = None,
) -> Optional[str]:
    if not sql or not sql.strip():
        return "SQL is empty."

    cleaned = normalize_sql(sql)
    no_comment_sql = sqlparse.format(cleaned, strip_comments=True).strip().lower()
    keyword_stream = _build_keyword_stream(cleaned)

    # Use sqlparse splitter to detect multiple statements while allowing
    # semicolons inside string literals.
    if len(sqlparse.split(cleaned)) > 1:
        return "Only one SQL statement is allowed."

    if not (no_comment_sql.startswith("select") or no_comment_sql.startswith("with")):
        return "Only SELECT queries are allowed."

    if SELECT_INTO_PATTERN.search(keyword_stream):
        return "SELECT INTO is not allowed."

    if LOCKING_PATTERN.search(keyword_stream):
        return "Locking clauses are not allowed."

    if FORBIDDEN_SQL_PATTERN.search(keyword_stream):
        return "Write or destructive SQL is not allowed."

    referenced_tables = _extract_referenced_tables(cleaned)
    cte_names = _extract_cte_names(cleaned)
    referenced_tables = referenced_tables.difference(cte_names)

    if not referenced_tables:
        return "Query must reference at least one table."

    if allowed_tables:
        allowed_lower = {t.lower() for t in allowed_tables}
        disallowed = [t for t in referenced_tables if t not in allowed_lower]
        if disallowed:
            return (
                "Query references table(s) outside allowed schema context: "
                + ", ".join(sorted(disallowed))
            )

    return None
