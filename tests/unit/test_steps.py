from datetime import date

from health_intake.engine.steps import ORDERED_STEPS, is_step_satisfied, next_step
from health_intake.models.conversation import RecordDraft, Step


def test_patient_info_requires_name_and_dob() -> None:
    assert not is_step_satisfied(Step.PATIENT_INFO, RecordDraft(full_name="Jane"))
    draft = RecordDraft(full_name="Jane", date_of_birth=date(1990, 1, 1))
    assert is_step_satisfied(Step.PATIENT_INFO, draft)


def test_insurance_id_is_optional() -> None:
    assert is_step_satisfied(Step.INSURANCE, RecordDraft(payer_name="Acme"))


def test_address_requires_validation_flag() -> None:
    draft = RecordDraft(street="1 Main", city="Town", state="CA", zip_code="90001")
    assert not is_step_satisfied(Step.ADDRESS, draft)
    assert is_step_satisfied(Step.ADDRESS, draft.model_copy(update={"address_validated": True}))


def test_next_step_advances_in_order() -> None:
    assert next_step(Step.GREETING) == Step.PATIENT_INFO
    assert next_step(Step.CONFIRMATION) == Step.CONFIRMATION  # terminal
    assert ORDERED_STEPS[0] == Step.GREETING
