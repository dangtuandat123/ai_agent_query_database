from taxi_agent.services.language_service import (
    empty_question_message,
    error_after_retry_message,
    fallback_no_data_message,
    fallback_success_message,
    internal_error_message,
    is_probably_vietnamese,
    normalize_for_matching,
    unsupported_message,
)


def test_is_probably_vietnamese_by_diacritics() -> None:
    assert is_probably_vietnamese("Trong thang nay co bao nhieu chuyen?")
    assert is_probably_vietnamese("Tổng doanh thu là bao nhiêu?")


def test_is_probably_vietnamese_english_false() -> None:
    assert not is_probably_vietnamese("How many trips are in March 2018?")
    assert not is_probably_vietnamese("Strong growth in revenue over time.")
    assert not is_probably_vietnamese("Haystack search in English text.")


def test_fallback_messages_vn() -> None:
    q = "Trong thang 3/2018 co bao nhieu chuyen?"
    assert "Không có dữ liệu" in fallback_no_data_message(q)
    assert "Đã truy vấn thành công" in fallback_success_message(q, 3)
    assert "Không thể trả lời" in unsupported_message(q, "x")
    assert "Thực thi truy vấn thất bại" in error_after_retry_message(
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


def test_internal_error_message_vn_en() -> None:
    assert "lỗi nội bộ" in internal_error_message("Trong thang 3/2018 co bao nhieu chuyen?")
    assert "internal error" in internal_error_message("How many trips in March 2018?")


def test_normalize_for_matching_strips_diacritics() -> None:
    assert normalize_for_matching("Còn so sánh tiếp") == "con so sanh tiep"
