from health_intake.llm.prompts import SYSTEM_PROMPT, build_situation
from health_intake.models.conversation import Step


def test_system_prompt_sets_role_and_safety():
    assert "intake" in SYSTEM_PROMPT.lower()
    assert "do not give medical advice" in SYSTEM_PROMPT.lower()


def test_build_situation_includes_validation_errors():
    situation = build_situation(
        step=Step.PATIENT_INFO, errors=["Date of birth must be in the past."], advancing=False
    )
    assert "Date of birth must be in the past." in situation
    assert "patient_info" in situation


def test_build_situation_signals_advance():
    situation = build_situation(step=Step.INSURANCE, errors=[], advancing=True)
    assert "advance" in situation.lower()
