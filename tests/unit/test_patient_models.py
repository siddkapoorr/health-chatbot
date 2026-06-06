from datetime import date, datetime

from health_intake.models.patient import (
    Address,
    Appointment,
    Insurance,
    PatientInfo,
    build_record,
)
from health_intake.models.conversation import RecordDraft


def test_build_record_from_complete_draft():
    draft = RecordDraft(
        full_name="Jane Doe",
        date_of_birth=date(1990, 3, 5),
        payer_name="Acme Health",
        insurance_id="AH123",
        chief_complaint="sore throat",
        street="1600 Amphitheatre Pkwy",
        city="Mountain View",
        state="CA",
        zip_code="94043",
        address_validated=True,
        address_formatted="1600 Amphitheatre Pkwy, Mountain View, CA 94043, USA",
        slot_id="chen-20260607-0900",
    )
    appointment = Appointment(
        slot_id="chen-20260607-0900",
        provider_name="Dr. Alice Chen",
        specialty="Family Medicine",
        start_time=datetime(2026, 6, 7, 9, 0),
    )

    record = build_record("sess-1", draft, appointment)

    assert record.patient == PatientInfo(full_name="Jane Doe", date_of_birth=date(1990, 3, 5))
    assert record.insurance == Insurance(payer_name="Acme Health", insurance_id="AH123")
    assert record.address.formatted.endswith("USA")
    assert record.appointment.provider_name == "Dr. Alice Chen"
    assert record.session_id == "sess-1"
