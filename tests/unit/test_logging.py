import logging
from pathlib import Path

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


def test_configure_logging_creates_log_file_and_redacts(tmp_path):
    from health_intake.logging_config import configure_logging

    log_dir = tmp_path / "logs"
    configure_logging("DEBUG", log_dir)

    test_logger = logging.getLogger("test.configure")
    test_logger.info("Email: john@example.com DOB: 1990-03-05")

    log_file = log_dir / "session.log"
    assert log_file.exists()
    content = log_file.read_text()
    assert "[REDACTED_EMAIL]" in content
    assert "john@example.com" not in content
