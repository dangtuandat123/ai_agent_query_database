import re


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


def is_probably_vietnamese(text: str) -> bool:
    lowered = text.lower()
    if VIETNAMESE_CHAR_PATTERN.search(lowered):
        return True

    normalized = re.sub(r"[^\w\s]", " ", lowered)
    return any(pattern.search(normalized) for pattern in VIETNAMESE_HINT_PATTERNS)


def fallback_no_data_message(question: str) -> str:
    if is_probably_vietnamese(question):
        return (
            "Kh\u00f4ng c\u00f3 d\u1eef li\u1ec7u ph\u00f9 h\u1ee3p "
            "v\u1edbi \u0111i\u1ec1u ki\u1ec7n truy v\u1ea5n."
        )
    return "No matching data was found for the query conditions."


def fallback_success_message(question: str, row_count: int) -> str:
    if is_probably_vietnamese(question):
        return (
            f"\u0110\u00e3 truy v\u1ea5n th\u00e0nh c\u00f4ng {row_count} d\u00f2ng "
            "(hi\u1ec3n th\u1ecb t\u1ed1i \u0111a theo c\u1ea5u h\u00ecnh)."
        )
    return f"Query succeeded with {row_count} rows (showing up to configured limit)."


def unsupported_message(question: str, reason: str) -> str:
    if is_probably_vietnamese(question):
        return (
            "Kh\u00f4ng th\u1ec3 tr\u1ea3 l\u1eddi y\u00eau c\u1ea7u n\u00e0y "
            "t\u1eeb schema PostgreSQL hi\u1ec7n t\u1ea1i.\n"
            f"L\u00fd do: {reason}"
        )
    return (
        "I cannot answer this request from the current PostgreSQL schema.\n"
        f"Reason: {reason}"
    )


def error_after_retry_message(question: str, sql_query: str, sql_error: str) -> str:
    if is_probably_vietnamese(question):
        return (
            "Th\u1ef1c thi truy v\u1ea5n th\u1ea5t b\u1ea1i sau khi \u0111\u00e3 "
            "th\u1eed s\u1eeda SQL.\n"
            f"C\u00e2u h\u1ecfi: {question}\n"
            f"SQL: {sql_query}\n"
            f"L\u1ed7i: {sql_error}"
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
