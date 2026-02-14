from taxi_agent.services.metadata_service import MetadataContextService


def test_metadata_context_detects_tables_columns_and_values() -> None:
    service = MetadataContextService(max_chars=2000)
    context = service.build(
        question=(
            "Trong Q2 2018, hãy tính total_amount theo payment_type "
            "cho bảng taxi_trip_data"
        ),
        allowed_tables=["public.taxi_trip_data", "taxi_trip_data"],
        schema_context=(
            "Table: public.taxi_trip_data\n"
            "Columns:\n"
            "- payment_type (integer)\n"
            "- total_amount (double precision)\n"
        ),
    )
    assert "matched tables" in context
    assert "taxi_trip_data" in context
    assert "likely columns" in context
    assert "payment_type" in context
    assert "total_amount" in context


def test_metadata_context_truncates() -> None:
    service = MetadataContextService(max_chars=50)
    context = service.build(
        question="How many rows?",
        allowed_tables=["public.taxi_trip_data"],
        schema_context="Table: public.taxi_trip_data",
    )
    assert len(context) <= 50
