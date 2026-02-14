import re
import unicodedata


VIETNAMESE_CHAR_PATTERN = re.compile(
    r"[\u0103\u00e2\u0111\u00ea\u00f4\u01a1\u01b0"
    r"\u00e1\u00e0\u1ea3\u00e3\u1ea1\u1eaf\u1eb1\u1eb3\u1eb5\u1eb7\u1ea5\u1ea7\u1ea9\u1eab\u1ead"
    r"\u00e9\u00e8\u1ebb\u1ebd\u1eb9\u1ebf\u1ec1\u1ec3\u1ec5\u1ec7"
    r"\u00ed\u00ec\u1ec9\u0129\u1ecb"
    r"\u00f3\u00f2\u1ecf\u00f5\u1ecd\u1ed1\u1ed3\u1ed5\u1ed7\u1ed9\u1edb\u1edd\u1edf\u1ee1\u1ee3"
    r"\u00fa\u00f9\u1ee7\u0169\u1ee5\u1ee9\u1eeb\u1eed\u1eef\u1ef1"
    r"\u00fd\u1ef3\u1ef7\u1ef9\u1ef5]"
)

VIETNAMESE_HINT_PATTERNS = [
    re.compile(r"\bbao\s+nhieu\b"),
    re.compile(r"\btrong\b"),
    re.compile(r"\bhay\b"),
    re.compile(r"\bcho\s+biet\b"),
    re.compile(r"\btinh\b"),
    re.compile(r"\bthang\b"),
    re.compile(r"\bnam\b"),
    re.compile(r"\bchuyen\b"),
    re.compile(r"\bdoanh\s+thu\b"),
    re.compile(r"\bdu\s+lieu\b"),
]


def normalize_for_matching(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text.lower())
    without_marks = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return without_marks.replace("đ", "d")


def is_probably_vietnamese(text: str) -> bool:
    lowered = text.lower()
    if VIETNAMESE_CHAR_PATTERN.search(lowered):
        return True

    normalized = re.sub(r"[^\w\s]", " ", lowered)
    return any(pattern.search(normalized) for pattern in VIETNAMESE_HINT_PATTERNS)


def fallback_no_data_message(question: str) -> str:
    if is_probably_vietnamese(question):
        return (
            "Không có dữ liệu phù hợp "
            "với điều kiện truy vấn."
        )
    return "No matching data was found for the query conditions."


def fallback_success_message(question: str, row_count: int) -> str:
    if is_probably_vietnamese(question):
        return (
            f"Đã truy vấn thành công {row_count} dòng "
            "(hiển thị tối đa theo cấu hình)."
        )
    return f"Query succeeded with {row_count} rows (showing up to configured limit)."


def unsupported_message(question: str, reason: str) -> str:
    if is_probably_vietnamese(question):
        return (
            "Không thể trả lời yêu cầu này "
            "từ schema PostgreSQL hiện tại.\n"
            f"Lý do: {reason}"
        )
    return (
        "I cannot answer this request from the current PostgreSQL schema.\n"
        f"Reason: {reason}"
    )


def error_after_retry_message(question: str, sql_query: str, sql_error: str) -> str:
    if is_probably_vietnamese(question):
        return (
            "Thực thi truy vấn thất bại sau khi đã "
            "thử sửa SQL.\n"
            f"Câu hỏi: {question}\n"
            f"SQL: {sql_query}\n"
            f"Lỗi: {sql_error}"
        )
    return (
        "Query execution failed after retry.\n"
        f"Question: {question}\n"
        f"SQL: {sql_query}\n"
        f"Error: {sql_error}"
    )


def empty_question_message() -> str:
    return (
        "Câu hỏi đang trống. Vui lòng nhập câu hỏi phân tích dữ liệu taxi.\n"
        "The question is empty. Please provide a taxi analytics question."
    )


def internal_error_message(question: str) -> str:
    if is_probably_vietnamese(question):
        return (
            "Hệ thống gặp lỗi nội bộ khi xử lý yêu cầu. "
            "Vui lòng thử lại sau."
        )
    return "The system encountered an internal error while processing your request. Please try again."
