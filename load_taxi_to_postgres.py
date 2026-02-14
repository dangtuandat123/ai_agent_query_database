from pathlib import Path
import os

from dotenv import load_dotenv
import psycopg


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS taxi_trip_data (
    vendor_id INTEGER,
    pickup_datetime TIMESTAMP,
    dropoff_datetime TIMESTAMP,
    passenger_count INTEGER,
    trip_distance DOUBLE PRECISION,
    rate_code INTEGER,
    store_and_fwd_flag TEXT,
    payment_type INTEGER,
    fare_amount DOUBLE PRECISION,
    extra DOUBLE PRECISION,
    mta_tax DOUBLE PRECISION,
    tip_amount DOUBLE PRECISION,
    tolls_amount DOUBLE PRECISION,
    imp_surcharge DOUBLE PRECISION,
    total_amount DOUBLE PRECISION,
    pickup_location_id INTEGER,
    dropoff_location_id INTEGER
);
"""


COPY_SQL = """
COPY taxi_trip_data (
    vendor_id,
    pickup_datetime,
    dropoff_datetime,
    passenger_count,
    trip_distance,
    rate_code,
    store_and_fwd_flag,
    payment_type,
    fare_amount,
    extra,
    mta_tax,
    tip_amount,
    tolls_amount,
    imp_surcharge,
    total_amount,
    pickup_location_id,
    dropoff_location_id
)
FROM STDIN WITH (FORMAT CSV, HEADER TRUE);
"""

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = PROJECT_ROOT / "dataset" / "taxi_trip_data.csv"


def _get_import_mode() -> str:
    mode = os.getenv("IMPORT_MODE", "fail_if_exists").strip().lower()
    supported = {"fail_if_exists", "truncate", "append"}
    if mode not in supported:
        raise ValueError(
            f"Unsupported IMPORT_MODE={mode!r}. Use one of: {', '.join(sorted(supported))}."
        )
    return mode


def _get_csv_path() -> Path:
    raw = os.getenv("TAXI_CSV_PATH", "").strip()
    if raw:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()
    return DEFAULT_CSV_PATH


def _get_connect_timeout_seconds() -> int:
    raw = os.getenv("DB_CONNECT_TIMEOUT_SECONDS", "10").strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"DB_CONNECT_TIMEOUT_SECONDS must be an integer, got: {raw!r}"
        ) from exc
    if value < 1:
        raise ValueError(f"DB_CONNECT_TIMEOUT_SECONDS must be >= 1, got: {value}")
    return value


def _count_rows(conn: psycopg.Connection) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM taxi_trip_data")
        value = cur.fetchone()
    return int(value[0] if value else 0)


def main() -> None:
    load_dotenv()
    postgres_dsn = os.getenv(
        "POSTGRES_DSN", "postgresql://postgres:postgres@localhost:5432/taxi_db"
    )
    connect_timeout_seconds = _get_connect_timeout_seconds()
    import_mode = _get_import_mode()
    csv_path = _get_csv_path()
    if not csv_path.exists():
        raise FileNotFoundError(f"Cannot find CSV file: {csv_path}")

    print("Connecting PostgreSQL...")
    with psycopg.connect(
        postgres_dsn,
        connect_timeout=connect_timeout_seconds,
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)
            conn.commit()

        existing_rows = _count_rows(conn)
        print(f"Current rows in taxi_trip_data: {existing_rows}")

        if existing_rows > 0 and import_mode == "fail_if_exists":
            raise RuntimeError(
                "Table taxi_trip_data already has data. "
                "Set IMPORT_MODE=truncate to reload from scratch or IMPORT_MODE=append to add more rows."
            )

        if import_mode == "truncate":
            print("Truncating existing data before import...")
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE taxi_trip_data")
                conn.commit()

        print("Copying CSV into taxi_trip_data table (this can take a while)...")
        with conn.cursor() as cur:
            with cur.copy(COPY_SQL) as copy:
                with csv_path.open("r", encoding="utf-8") as file_obj:
                    while True:
                        chunk = file_obj.read(1024 * 1024)
                        if not chunk:
                            break
                        copy.write(chunk)
            conn.commit()

    print(f"Done loading data into PostgreSQL (mode={import_mode}).")


if __name__ == "__main__":
    main()
