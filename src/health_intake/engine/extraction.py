"""Structured-output schema the LLM fills from user messages."""

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
