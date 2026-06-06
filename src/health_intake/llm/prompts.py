"""System prompt and the per-turn 'situation' message that steers generation."""

from health_intake.models.conversation import Step

SYSTEM_PROMPT = (
    "You are a warm, professional medical front-desk assistant collecting patient "
    "intake information in a terminal chat. Be concise and friendly. Ask for one thing "
    "at a time unless the patient volunteers more. Do not give medical advice or "
    "diagnoses. Never invent information the patient did not provide."
)

_STEP_GOALS = {
    Step.GREETING: "Greet the patient and ask for their full name to begin.",
    Step.PATIENT_INFO: "Collect the patient's full name and date of birth.",
    Step.INSURANCE: "Collect the insurance payer name (insurance ID is optional).",
    Step.CHIEF_COMPLAINT: "Ask the reason for the visit / chief complaint.",
    Step.ADDRESS: "Collect the full mailing address (street, city, state, ZIP).",
    Step.APPOINTMENT: "Help the patient pick one of the listed appointment slots.",
    Step.CONFIRMATION: "Confirm everything is collected and thank the patient.",
}


def build_situation(step: Step, errors: list[str], advancing: bool, extra: str = "") -> str:
    """Build a deterministic instruction describing the real validation outcome.

    The generation call must reflect this exactly, so the assistant message can never
    contradict what Python actually validated.
    """
    lines = [f"Current step: {step.value}.", _STEP_GOALS[step]]
    if errors:
        lines.append(
            "The patient's last input had problems. Re-ask, mentioning: " + " ".join(errors)
        )
    elif advancing:
        lines.append(
            "All required info for this step is valid. Acknowledge and advance to the next topic."
        )
    if extra:
        lines.append(extra)
    return "\n".join(lines)
