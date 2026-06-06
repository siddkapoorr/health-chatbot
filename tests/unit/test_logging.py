import logging

from health_intake.logging_config import RedactionFilter


def _redact(message: str) -> str:
    record = logging.LogRecord("t", logging.INFO, __file__, 1, message, None, None)
    RedactionFilter().filter(record)
    return record.getMessage()


def test_redacts_email():
    assert "[REDACTED_EMAIL]" in _redact("contact jane@example.com now")


def test_redacts_date_of_birth():
    assert "[REDACTED_DOB]" in _redact("dob 1990-03-05")


def test_redacts_long_digit_runs():
    assert "[REDACTED_NUM]" in _redact("id 123456789")


def test_keeps_safe_text():
    assert _redact("advanced to step address") == "advanced to step address"
