from datetime import datetime

from health_intake.appointments.provider import available_slots
from health_intake.cli import run_session
from health_intake.engine.extraction import FieldExtraction
from health_intake.engine.orchestrator import Orchestrator
from health_intake.llm.client import FakeLLMClient

from tests.conftest import FakeAddressValidator, make_settings

NOW = datetime(2026, 6, 6, 12, 0)


def test_run_session_drives_until_complete(tmp_path):
    settings = make_settings(tmp_path)
    slot = available_slots(NOW)[0]
    llm = FakeLLMClient(
        extractions=[
            FieldExtraction(full_name="Jane Doe", date_of_birth="1990-03-05"),
            FieldExtraction(payer_name="Acme"),
            FieldExtraction(chief_complaint="cough"),
            FieldExtraction(
                street="1600 Amphitheatre Pkwy", city="Mountain View", state="CA", zip_code="94043"
            ),
            FieldExtraction(appointment_choice=slot.slot_id),
        ],
        replies=["a", "b", "c", "d", "done"],
    )
    orch = Orchestrator(llm, FakeAddressValidator(), settings, now_fn=lambda: NOW)

    addr = "1600 Amphitheatre Pkwy, Mountain View, CA 94043"
    scripted_inputs = iter(["Jane Doe 1990-03-05", "Acme", "cough", addr, slot.slot_id])
    outputs: list[str] = []

    record = run_session(orch, input_fn=lambda _: next(scripted_inputs), output_fn=outputs.append)

    assert record is not None
    assert record.patient.full_name == "Jane Doe"
    assert any("done" in line for line in outputs)
