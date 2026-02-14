import pytest

from main import parse_args


def test_parse_args_default_question() -> None:
    args = parse_args([])
    assert args.question == ""
    assert args.thread_id == "default"


def test_parse_args_custom_question() -> None:
    args = parse_args(["--question", "Xin chao"])
    assert args.question == "Xin chao"


def test_parse_args_custom_thread_id() -> None:
    args = parse_args(["--thread-id", "team-finance"])
    assert args.thread_id == "team-finance"


def test_parse_args_help_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as exc:
        parse_args(["--help"])
    assert exc.value.code == 0
