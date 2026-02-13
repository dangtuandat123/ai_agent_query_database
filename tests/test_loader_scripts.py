from pathlib import Path

import pytest

import load_taxi_to_postgres as loader
import preview_taxi_dataset as preview


def test_get_import_mode_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IMPORT_MODE", raising=False)
    assert loader._get_import_mode() == "fail_if_exists"


def test_get_import_mode_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMPORT_MODE", "invalid_mode")
    with pytest.raises(ValueError):
        loader._get_import_mode()


def test_loader_get_csv_path_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TAXI_CSV_PATH", raising=False)
    csv_path = loader._get_csv_path()
    assert csv_path.name == "taxi_trip_data.csv"
    assert csv_path.parent.name == "dataset"


def test_loader_get_csv_path_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAXI_CSV_PATH", "dataset/taxi_trip_data.csv")
    csv_path = loader._get_csv_path()
    assert csv_path == (loader.PROJECT_ROOT / "dataset" / "taxi_trip_data.csv").resolve()


def test_preview_get_csv_path_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TAXI_CSV_PATH", raising=False)
    csv_path = preview._get_csv_path()
    assert csv_path.name == "taxi_trip_data.csv"
    assert csv_path.parent.name == "dataset"


def test_preview_get_csv_path_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAXI_CSV_PATH", "dataset/taxi_trip_data.csv")
    csv_path = preview._get_csv_path()
    assert csv_path == (preview.PROJECT_ROOT / "dataset" / "taxi_trip_data.csv").resolve()


def test_smoke_run_alias_exists() -> None:
    assert Path("smoke_run.py").exists()
