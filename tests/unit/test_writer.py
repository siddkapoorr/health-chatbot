import json
from datetime import date, datetime

from health_intake.models.patient import (
    Address,
    Appointment,
    Insurance,
    IntakeRecord,
    PatientInfo,
)
from health_intake.storage.writer import write_record


def _record() -> IntakeRecord:
    return IntakeRecord(
        session_id="sess-42",
        created_at=datetime(2026, 6, 6, 12, 0),
        patient=PatientInfo(full_name="Jane Doe", date_of_birth=date(1990, 3, 5)),
        insurance=Insurance(payer_name="Acme", insurance_id=None),
        chief_complaint="cough",
        address=Address(street="1 Main", city="Town", state="CA", zip_code="90001"),
        appointment=Appointment(
            slot_id="chen-1", provider_name="Dr. Alice Chen",
            specialty="Family Medicine", start_time=datetime(2026, 6, 7, 9, 0),
        ),
    )


def test_write_record_creates_json_file(tmp_path):
    path = write_record(_record(), tmp_path)

    assert path.exists()
    assert path.name == "intake-sess-42.json"
    data = json.loads(path.read_text())
    assert data["patient"]["full_name"] == "Jane Doe"
