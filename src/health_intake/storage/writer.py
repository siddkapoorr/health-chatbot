"""Persist a completed IntakeRecord as JSON."""

import logging
from pathlib import Path

from health_intake.models.patient import IntakeRecord

logger = logging.getLogger(__name__)


def write_record(record: IntakeRecord, output_dir: Path) -> Path:
    """Write the record to ``output_dir/intake-<session_id>.json`` and return the path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"intake-{record.session_id}.json"
    path.write_text(record.model_dump_json(indent=2))
    logger.info("Wrote intake record for session %s", record.session_id)
    return path
