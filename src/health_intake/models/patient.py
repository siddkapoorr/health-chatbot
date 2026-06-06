"""Pydantic domain models for the final intake record."""

from datetime import date, datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from health_intake.models.conversation import RecordDraft


class Address(BaseModel):
    street: str
    city: str
    state: str
    zip_code: str
    validated: bool = False
    formatted: str | None = None


class Insurance(BaseModel):
    payer_name: str
    insurance_id: str | None = None


class PatientInfo(BaseModel):
    full_name: str
    date_of_birth: date


class Appointment(BaseModel):
    slot_id: str
    provider_name: str
    specialty: str
    start_time: datetime


class IntakeRecord(BaseModel):
    session_id: str
    created_at: datetime
    patient: PatientInfo
    insurance: Insurance
    chief_complaint: str
    address: Address
    appointment: Appointment


def build_record(
    session_id: str, draft: "RecordDraft", appointment: Appointment
) -> IntakeRecord:
    """Assemble a complete IntakeRecord from a fully-collected draft.

    Assumes the draft has passed all step gates; callers must check completeness first.
    """
    assert draft.full_name and draft.date_of_birth and draft.payer_name
    assert draft.chief_complaint and draft.street and draft.city
    assert draft.state and draft.zip_code

    return IntakeRecord(
        session_id=session_id,
        created_at=datetime.now(),
        patient=PatientInfo(full_name=draft.full_name, date_of_birth=draft.date_of_birth),
        insurance=Insurance(payer_name=draft.payer_name, insurance_id=draft.insurance_id),
        chief_complaint=draft.chief_complaint,
        address=Address(
            street=draft.street,
            city=draft.city,
            state=draft.state,
            zip_code=draft.zip_code,
            validated=draft.address_validated,
            formatted=draft.address_formatted,
        ),
        appointment=appointment,
    )
