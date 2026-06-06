"""Persist a completed IntakeRecord as JSON."""

import logging
import os
from pathlib import Path

from health_intake.models.patient import IntakeRecord

logger = logging.getLogger(__name__)


def write_record(record: IntakeRecord, output_dir: Path) -> Path:
    """Write the record to ``output_dir/intake-<session_id>.json`` and return the path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    # Enforce 0o700 on every call — mkdir's mode is a no-op when exist_ok=True and
    # the directory already exists.
    os.chmod(output_dir, 0o700)

    path = output_dir / f"intake-{record.session_id}.json"
    # Open atomically with O_CREAT|O_EXCL|O_NOFOLLOW at mode 0o600 to prevent TOCTOU
    # races and symlink attacks before permissions are set.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(record.model_dump_json(indent=2))

    logger.info("Wrote intake record for session %s", record.session_id)
    return path
