"""Logging configuration with a PII redaction safety net.

Primary defense against PHI leakage is logging *events*, not raw field values.
This filter is a secondary safety net that masks anything that slips through.
"""

import logging
import os
import re
import traceback
from pathlib import Path

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b")
_LONG_NUM = re.compile(r"\b\d{5,}\b")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_PHONE = re.compile(r"\b\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}\b")


def _scrub(text: str) -> str:
    """Apply all redaction patterns to text."""
    text = _EMAIL.sub("[REDACTED_EMAIL]", text)
    text = _DATE.sub("[REDACTED_DOB]", text)
    text = _LONG_NUM.sub("[REDACTED_NUM]", text)
    text = _SSN.sub("[REDACTED_SSN]", text)
    text = _PHONE.sub("[REDACTED_PHONE]", text)
    return text


class RedactionFilter(logging.Filter):
    """Mask emails, dates of birth, SSNs, phone numbers, and long digit runs in log messages.

    Also redacts exception text and stack info to prevent PHI leakage in tracebacks.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        message = _scrub(message)
        record.msg = message
        record.args = None

        # Redact exception text and stack info — tracebacks can contain PHI if a
        # field value appears in an error message or repr.
        if record.exc_info and not record.exc_text:
            record.exc_text = "".join(traceback.format_exception(*record.exc_info))
        if record.exc_text:
            record.exc_text = _scrub(record.exc_text)
        if record.stack_info:
            record.stack_info = _scrub(record.stack_info)
        record.exc_info = None  # prevent formatter from re-rendering the raw traceback

        return True


def configure_logging(level: str, log_dir: Path) -> None:
    """Configure root logging to console + rotating file, with redaction."""
    log_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    handlers: list[logging.Handler] = [
        logging.StreamHandler(),
        logging.FileHandler(log_dir / "session.log"),
    ]
    os.chmod(log_dir / "session.log", 0o600)
    redaction = RedactionFilter()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    for handler in handlers:
        handler.addFilter(redaction)
        handler.setFormatter(formatter)
    logging.basicConfig(level=level.upper(), handlers=handlers, force=True)
