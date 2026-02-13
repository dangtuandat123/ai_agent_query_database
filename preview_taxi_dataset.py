from pathlib import Path
import os

import pandas as pd


COLUMN_DESCRIPTIONS = {
    "vendor_id": "TPEP vendor code. 1=Creative Mobile Technologies, 2=VeriFone Inc.",
    "pickup_datetime": "Trip pickup timestamp.",
    "dropoff_datetime": "Trip dropoff timestamp.",
    "passenger_count": "Number of passengers (driver-entered value).",
    "trip_distance": "Trip distance in miles from taximeter.",
    "rate_code": "Final rate code. 1=Standard, 2=JFK, 3=Newark, 4=Nassau/Westchester, 5=Negotiated, 6=Group ride.",
    "store_and_fwd_flag": "Y=stored then forwarded from vehicle memory, N=sent immediately.",
    "payment_type": "Payment method. 1=Card, 2=Cash, 3=No charge, 4=Dispute, 5=Unknown, 6=Voided trip.",
    "fare_amount": "Base fare amount (time + distance).",
    "extra": "Extra charges (e.g. rush hour / overnight).",
    "mta_tax": "MTA tax.",
    "tip_amount": "Tip amount.",
    "tolls_amount": "Toll amount.",
    "imp_surcharge": "Improvement surcharge.",
    "total_amount": "Total amount paid by passenger.",
    "pickup_location_id": "Pickup Taxi Zone ID.",
    "dropoff_location_id": "Dropoff Taxi Zone ID.",
}


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = PROJECT_ROOT / "dataset" / "taxi_trip_data.csv"


def _get_csv_path() -> Path:
    raw = os.getenv("TAXI_CSV_PATH", "").strip()
    if raw:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()
    return DEFAULT_CSV_PATH


def main() -> None:
    csv_path = _get_csv_path()
    if not csv_path.exists():
        raise FileNotFoundError(f"Cannot find file: {csv_path}")

    # Read only a small sample to inspect data shape quickly.
    df = pd.read_csv(
        csv_path,
        nrows=20,
        parse_dates=["pickup_datetime", "dropoff_datetime"],
    )

    print(f"File: {csv_path}")
    print(f"Sample rows: {len(df)}")
    print(f"Column count: {len(df.columns)}")

    print("\nColumns:")
    for col in df.columns:
        print(f"- {col}")

    print("\nColumn descriptions:")
    for col in df.columns:
        print(f"- {col}: {COLUMN_DESCRIPTIONS.get(col, 'No description available')}")

    print("\nFirst 5 rows:")
    print(df.head(5).to_string(index=False))

    print("\nDtypes by column:")
    print(df.dtypes.to_string())

    print("\nMissing values in sample (20 rows):")
    print(df.isna().sum().to_string())


if __name__ == "__main__":
    main()
