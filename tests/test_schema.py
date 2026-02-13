from taxi_agent.schema import ColumnSchema, TableSchema, build_schema_context


def _table() -> TableSchema:
    return TableSchema(
        table_schema="public",
        table_name="taxi_trip_data",
        columns=[
            ColumnSchema("vendor_id", "integer", 1),
            ColumnSchema("trip_distance", "double precision", 2),
        ],
    )


def test_build_schema_context_tiny_max_chars() -> None:
    text = build_schema_context([_table()], max_columns_per_table=10, max_chars=2)
    assert len(text) <= 2

    text = build_schema_context([_table()], max_columns_per_table=10, max_chars=3)
    assert len(text) <= 3


def test_build_schema_context_regular_truncation_has_ellipsis() -> None:
    text = build_schema_context([_table()], max_columns_per_table=10, max_chars=20)
    assert len(text) <= 20
    assert text.endswith("...")
