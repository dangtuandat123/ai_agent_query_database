from taxi_agent.sql_guard import (
    _extract_cte_names,
    normalize_sql,
    sanitize_sql,
    validate_readonly_sql,
)


def test_allow_simple_select() -> None:
    err = validate_readonly_sql("SELECT * FROM public.taxi_trip_data LIMIT 1")
    assert err is None


def test_allow_with_select() -> None:
    err = validate_readonly_sql(
        "WITH a AS (SELECT * FROM public.taxi_trip_data LIMIT 1) SELECT * FROM a"
    )
    assert err is None


def test_allow_with_recursive_cte() -> None:
    err = validate_readonly_sql(
        "WITH RECURSIVE nums AS ("
        "SELECT vendor_id AS n FROM public.taxi_trip_data LIMIT 1 "
        "UNION ALL "
        "SELECT n + 1 FROM nums WHERE n < 3"
        ") SELECT * FROM nums",
        allowed_tables=["public.taxi_trip_data", "taxi_trip_data"],
    )
    assert err is None


def test_allow_complex_cte_without_keyword_false_positive() -> None:
    sql = (
        "WITH base_trips AS ("
        "  SELECT pickup_location_id, dropoff_location_id, payment_type, total_amount "
        "  FROM public.taxi_trip_data "
        "  WHERE pickup_datetime >= '2018-04-01' AND pickup_datetime < '2018-07-01'"
        "), final_metrics AS ("
        "  SELECT pickup_location_id, dropoff_location_id, payment_type, "
        "         SUM(total_amount) AS total_revenue_q2 "
        "  FROM base_trips "
        "  GROUP BY 1,2,3"
        ") "
        "SELECT *, "
        "  DENSE_RANK() OVER (PARTITION BY payment_type ORDER BY total_revenue_q2 DESC) AS payment_rank "
        "FROM final_metrics "
        "ORDER BY total_revenue_q2 DESC "
        "LIMIT 20"
    )
    err = validate_readonly_sql(
        sql,
        allowed_tables=["public.taxi_trip_data", "taxi_trip_data"],
    )
    assert err is None


def test_deny_multi_statement() -> None:
    err = validate_readonly_sql("SELECT 1; SELECT 2")
    assert err == "Only one SQL statement is allowed."


def test_allow_semicolon_inside_literal() -> None:
    err = validate_readonly_sql("SELECT ';' AS marker FROM public.taxi_trip_data LIMIT 1")
    assert err is None


def test_sanitize_sql_code_fence() -> None:
    raw = "```sql\nSELECT * FROM public.taxi_trip_data LIMIT 1;\n```"
    assert sanitize_sql(raw) == "SELECT * FROM public.taxi_trip_data LIMIT 1;"
    assert normalize_sql(raw) == "SELECT * FROM public.taxi_trip_data LIMIT 1"


def test_sanitize_sql_code_fence_with_explanation() -> None:
    raw = "Here is your SQL:\n```sql\nSELECT * FROM public.taxi_trip_data LIMIT 1;\n```\nUse it."
    assert normalize_sql(raw) == "SELECT * FROM public.taxi_trip_data LIMIT 1"


def test_sanitize_sql_prefix() -> None:
    raw = "SQLQuery: SELECT * FROM public.taxi_trip_data LIMIT 1;"
    assert normalize_sql(raw) == "SELECT * FROM public.taxi_trip_data LIMIT 1"


def test_sanitize_sql_with_intro_text() -> None:
    raw = "The answer can be found by using SELECT * FROM public.taxi_trip_data LIMIT 1;"
    assert normalize_sql(raw) == "SELECT * FROM public.taxi_trip_data LIMIT 1"


def test_allow_keywords_inside_string_literal() -> None:
    err = validate_readonly_sql(
        "SELECT 'drop table x; for update' AS note FROM public.taxi_trip_data LIMIT 1"
    )
    assert err is None


def test_allow_keywords_inside_comment() -> None:
    err = validate_readonly_sql(
        "-- drop table x\nSELECT * FROM public.taxi_trip_data LIMIT 1"
    )
    assert err is None


def test_allow_subquery_alias_with_allowlist() -> None:
    err = validate_readonly_sql(
        "SELECT t.* FROM (SELECT * FROM public.taxi_trip_data LIMIT 5) t LIMIT 5",
        allowed_tables=["public.taxi_trip_data", "taxi_trip_data"],
    )
    assert err is None


def test_extract_cte_names_ignores_select_aliases() -> None:
    sql = (
        "WITH a AS (SELECT payment_type AS p FROM public.taxi_trip_data), "
        "b AS (SELECT p AS q FROM a) "
        "SELECT q AS final_alias FROM b"
    )
    assert _extract_cte_names(sql) == {"a", "b"}


def test_deny_destructive_sql() -> None:
    err = validate_readonly_sql("DELETE FROM public.taxi_trip_data")
    assert err == "Only SELECT queries are allowed."


def test_deny_select_into() -> None:
    err = validate_readonly_sql("SELECT * INTO temp t FROM public.taxi_trip_data")
    assert err == "SELECT INTO is not allowed."


def test_deny_locking_clause() -> None:
    err = validate_readonly_sql("SELECT * FROM public.taxi_trip_data FOR UPDATE")
    assert err == "Locking clauses are not allowed."


def test_allowlist_pass() -> None:
    err = validate_readonly_sql(
        "SELECT * FROM public.taxi_trip_data LIMIT 1",
        allowed_tables=["public.taxi_trip_data", "taxi_trip_data"],
    )
    assert err is None


def test_allowlist_fail() -> None:
    err = validate_readonly_sql(
        "SELECT * FROM public.other_table LIMIT 1",
        allowed_tables=["public.taxi_trip_data", "taxi_trip_data"],
    )
    assert err is not None
    assert "outside allowed schema context" in err
    assert "public.other_table" in err
    assert "other_table, public.other_table" not in err
