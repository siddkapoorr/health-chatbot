"""Structured-output schema the LLM fills from user messages."""

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel


class FieldExtraction(BaseModel):
    """Fields the LLM may extract from the conversation. All optional.

    Raw strings only — Python validates/parses them. ``appointment_choice`` should be
    the slot id shown in the presented list when the user picks a time.
    """

    full_name: str | None = None
    date_of_birth: str | None = None
    payer_name: str | None = None
    insurance_id: str | None = None
    chief_complaint: str | None = None
    street: str | None = None
    city: str | None = None
    state: str | None = None
    zip_code: str | None = None
    appointment_choice: str | None = None
    user_asked_question: bool = False


from health_intake.models.conversation import RecordDraft  # noqa: E402
from health_intake.validation.fields import (  # noqa: E402
    validate_chief_complaint,
    validate_date_of_birth,
    validate_full_name,
    validate_insurance_id,
    validate_payer_name,
)


def apply_extraction(
    draft: RecordDraft, extraction: FieldExtraction
) -> tuple[RecordDraft, list[str]]:
    """Validate each provided field and return a new draft plus any error messages.

    Address validation and appointment matching are handled by the orchestrator
    because they require external services / runtime data. Raw address parts and the
    appointment choice are copied through here for the orchestrator to finalize.
    """
    updates: dict[str, object] = {}
    errors: list[str] = []

    def apply(raw: str | None, validator: Callable[[str], Any], key: str) -> None:
        if raw is None:
            return
        result = validator(raw)
        if result.ok:
            updates[key] = result.value
        elif result.error:
            errors.append(result.error)

    apply(extraction.full_name, validate_full_name, "full_name")
    apply(extraction.date_of_birth, validate_date_of_birth, "date_of_birth")
    apply(extraction.payer_name, validate_payer_name, "payer_name")
    apply(extraction.insurance_id, validate_insurance_id, "insurance_id")
    apply(extraction.chief_complaint, validate_chief_complaint, "chief_complaint")

    # Raw address parts pass through unvalidated here; the API check happens in the
    # orchestrator once all four parts are present.
    for key in ("street", "city", "state", "zip_code"):
        value = getattr(extraction, key)
        if value:
            updates[key] = value.strip()

    return draft.model_copy(update=updates), errors
