"""Deterministic per-field validators. The LLM never decides validity."""

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Generic, TypeVar

T = TypeVar("T")

MAX_AGE_YEARS = 120
MAX_COMPLAINT_LEN = 1000
_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z\s'\-]*$")
_INSURANCE_ID_RE = re.compile(r"^[A-Za-z0-9\-]+$")
_DATE_FORMATS = ("%Y-%m-%d", "%m/%d/%Y", "%d %B %Y", "%B %d, %Y")


@dataclass(frozen=True)
class FieldResult(Generic[T]):
    ok: bool
    value: T | None = None
    error: str | None = None


def validate_full_name(raw: str) -> FieldResult[str]:
    cleaned = raw.strip()
    if not cleaned:
        return FieldResult(ok=False, error="A full name is required.")
    if not _NAME_RE.match(cleaned):
        msg = "Name may only contain letters, spaces, hyphens, apostrophes."
        return FieldResult(ok=False, error=msg)
    return FieldResult(ok=True, value=cleaned)


def validate_date_of_birth(raw: str) -> FieldResult[date]:
    cleaned = raw.strip()
    parsed: date | None = None
    for fmt in _DATE_FORMATS:
        try:
            parsed = datetime.strptime(cleaned, fmt).date()
            break
        except ValueError:
            continue
    if parsed is None:
        return FieldResult(ok=False, error="Please give a date of birth like 1990-03-05.")
    today = date.today()
    if parsed >= today:
        return FieldResult(ok=False, error="Date of birth must be in the past.")
    if (today.year - parsed.year) > MAX_AGE_YEARS:
        return FieldResult(ok=False, error="That date of birth doesn't look plausible.")
    return FieldResult(ok=True, value=parsed)


def validate_payer_name(raw: str) -> FieldResult[str]:
    cleaned = raw.strip()
    if not cleaned:
        return FieldResult(ok=False, error="An insurance payer name is required.")
    return FieldResult(ok=True, value=cleaned)


def validate_insurance_id(raw: str) -> FieldResult[str]:
    cleaned = raw.strip()
    if not cleaned:
        return FieldResult(ok=True, value=None)  # optional
    if not _INSURANCE_ID_RE.match(cleaned):
        msg = "Insurance ID may only contain letters, numbers, hyphens."
        return FieldResult(ok=False, error=msg)
    return FieldResult(ok=True, value=cleaned)


def validate_chief_complaint(raw: str) -> FieldResult[str]:
    cleaned = raw.strip()
    if not cleaned:
        return FieldResult(ok=False, error="Please describe the reason for your visit.")
    if len(cleaned) > MAX_COMPLAINT_LEN:
        return FieldResult(ok=False, error="That description is too long.")
    return FieldResult(ok=True, value=cleaned)
