from taxi_agent.services.language_service import (
    empty_question_message,
    error_after_retry_message,
    fallback_no_data_message,
    fallback_success_message,
    is_probably_vietnamese,
    unsupported_message,
)


def test_is_probably_vietnamese_by_diacritics() -> None:
    assert is_probably_vietnamese("Trong thang nay co bao nhieu chuyen?")
    assert is_probably_vietnamese("T\u1ed5ng doanh thu l\u00e0 bao nhi\u00eau?")


def test_is_probably_vietnamese_english_false() -> None:
    assert not is_probably_vietnamese("How many trips are in March 2018?")
    assert not is_probably_vietnamese("Strong growth in revenue over time.")
    assert not is_probably_vietnamese("Haystack search in English text.")


def test_fallback_messages_vn() -> None:
    q = "Trong thang 3/2018 co bao nhieu chuyen?"
    assert "Kh\u00f4ng c\u00f3 d\u1eef li\u1ec7u" in fallback_no_data_message(q)
    assert "\u0110\u00e3 truy v\u1ea5n th\u00e0nh c\u00f4ng" in fallback_success_message(q, 3)
    assert "Kh\u00f4ng th\u1ec3 tr\u1ea3 l\u1eddi" in unsupported_message(q, "x")
    assert "Th\u1ef1c thi truy v\u1ea5n th\u1ea5t b\u1ea1i" in error_after_retry_message(
        q, "SELECT 1", "err"
    )


def test_fallback_messages_en() -> None:
    q = "How many trips are in March 2018?"
    assert "No matching data" in fallback_no_data_message(q)
    assert "Query succeeded" in fallback_success_message(q, 3)
    assert "I cannot answer" in unsupported_message(q, "x")
    assert "Query execution failed after retry" in error_after_retry_message(
        q, "SELECT 1", "err"
    )


def test_empty_question_message() -> None:
    msg = empty_question_message()
    assert "Câu hỏi đang trống" in msg
    assert "The question is empty" in msg
