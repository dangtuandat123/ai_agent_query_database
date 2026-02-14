import re


POSTGRES_URL_PASSWORD_PATTERN = re.compile(
    r"(?i)(postgres(?:ql)?://[^:\s/]+:)([^@/\s]+)(@)"
)
DSN_PASSWORD_PATTERN = re.compile(r"(?i)(password=)([^\s]+)")
API_KEY_PATTERN = re.compile(r"(?i)(api[_-]?key(?:\s*[:=]\s*|\s+))([^\s,;]+)")
BEARER_PATTERN = re.compile(r"(?i)(authorization:\s*bearer\s+)([^\s]+)")


def redact_sensitive_text(text: str) -> str:
    redacted = POSTGRES_URL_PASSWORD_PATTERN.sub(r"\1***\3", text)
    redacted = DSN_PASSWORD_PATTERN.sub(r"\1***", redacted)
    redacted = API_KEY_PATTERN.sub(r"\1***", redacted)
    redacted = BEARER_PATTERN.sub(r"\1***", redacted)
    return redacted

