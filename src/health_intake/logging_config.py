"""Logging configuration with a PII redaction safety net.

Primary defense against PHI leakage is logging *events*, not raw field values.
This filter is a secondary safety net that masks anything that slips through.
"""

import logging
import re
from pathlib import Path

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b")
_LONG_NUM = re.compile(r"\b\d{5,}\b")


class RedactionFilter(logging.Filter):
    """Mask emails, dates of birth, and long digit runs in log messages."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        message = _EMAIL.sub("[REDACTED_EMAIL]", message)
        message = _DATE.sub("[REDACTED_DOB]", message)
        message = _LONG_NUM.sub("[REDACTED_NUM]", message)
        record.msg = message
        record.args = None
        return True


def configure_logging(level: str, log_dir: Path) -> None:
    """Configure root logging to console + rotating file, with redaction."""
    log_dir.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [
        logging.StreamHandler(),
        logging.FileHandler(log_dir / "session.log"),
    ]
    redaction = RedactionFilter()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    for handler in handlers:
        handler.addFilter(redaction)
        handler.setFormatter(formatter)
    logging.basicConfig(level=level.upper(), handlers=handlers, force=True)
