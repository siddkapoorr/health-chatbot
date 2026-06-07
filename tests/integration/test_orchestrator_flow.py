from datetime import datetime

from health_intake.appointments.provider import available_slots
from health_intake.engine.extraction import FieldExtraction
from health_intake.engine.orchestrator import Orchestrator
from health_intake.llm.client import FakeLLMClient
from health_intake.models.conversation import ConversationState, RecordDraft, Step

from tests.conftest import FakeAddressValidator, make_settings

NOW = datetime(2026, 6, 6, 12, 0)


def _orchestrator(llm: object, settings: object) -> Orchestrator:
    return Orchestrator(
        llm=llm,
        address_validator=FakeAddressValidator(),
        settings=settings,
        now_fn=lambda: NOW,
    )


def test_happy_path_collects_record_and_writes_json(tmp_path):
    settings = make_settings(tmp_path)
    slot = available_slots(NOW)[0]
    scripted = [
        FieldExtraction(full_name="Jane Doe", date_of_birth="1990-03-05"),
        FieldExtraction(payer_name="Acme Health", insurance_id="AH123"),
        FieldExtraction(chief_complaint="sore throat for 3 days"),
        FieldExtraction(
            street="1600 Amphitheatre Pkwy", city="Mountain View", state="CA", zip_code="94043"
        ),
        FieldExtraction(appointment_choice=slot.slot_id),
    ]
    replies = ["ok1", "ok2", "ok3", "ok4", "Thanks, all set!"]
    llm = FakeLLMClient(extractions=scripted, replies=replies)
    orch = _orchestrator(llm, settings)

    state, _ = orch.start()
    inputs = [
        "I'm Jane Doe, born 1990-03-05",
        "Acme Health, AH123",
        "sore throat for 3 days",
        "1600 Amphitheatre Pkwy, Mountain View, CA 94043",
        f"I'll take {slot.slot_id}",
    ]
    result = None
    for text in inputs:
        result = orch.handle_turn(state, text)
        state = result.state

    assert result.is_complete
    assert result.record is not None
    assert result.record.patient.full_name == "Jane Doe"
    assert (tmp_path / f"intake-{result.record.session_id}.json").exists()


def test_invalid_dob_blocks_advance(tmp_path):
    settings = make_settings(tmp_path)
    llm = FakeLLMClient(
        extractions=[FieldExtraction(date_of_birth="2999-01-01")],
        replies=["Please re-check your date of birth."],
    )
    orch = _orchestrator(llm, settings)
    state = ConversationState(
        session_id="s1", current_step=Step.PATIENT_INFO, draft=RecordDraft(full_name="Jane")
    )

    result = orch.handle_turn(state, "born in 2999")

    assert not result.is_complete
    assert result.state.current_step == Step.PATIENT_INFO  # did not advance
    assert result.state.draft.date_of_birth is None
