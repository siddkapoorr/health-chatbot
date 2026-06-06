"""Ordered steps and the deterministic gates that decide when each is satisfied."""

from health_intake.models.conversation import RecordDraft, Step

ORDERED_STEPS: tuple[Step, ...] = (
    Step.GREETING,
    Step.PATIENT_INFO,
    Step.INSURANCE,
    Step.CHIEF_COMPLAINT,
    Step.ADDRESS,
    Step.APPOINTMENT,
    Step.CONFIRMATION,
)


def is_step_satisfied(step: Step, draft: RecordDraft) -> bool:
    """Determine if a step's requirements are met in the draft.

    GREETING: always satisfied (entry state)
    PATIENT_INFO: requires full_name and date_of_birth
    INSURANCE: requires payer_name
    CHIEF_COMPLAINT: requires chief_complaint
    ADDRESS: requires address_validated flag (not just address fields)
    APPOINTMENT: requires slot_id
    CONFIRMATION: delegates to is_record_complete()
    """
    if step == Step.GREETING:
        return True
    if step == Step.PATIENT_INFO:
        return bool(draft.full_name and draft.date_of_birth)
    if step == Step.INSURANCE:
        return bool(draft.payer_name)
    if step == Step.CHIEF_COMPLAINT:
        return bool(draft.chief_complaint)
    if step == Step.ADDRESS:
        return draft.address_validated
    if step == Step.APPOINTMENT:
        return bool(draft.slot_id)
    return is_record_complete(draft)  # CONFIRMATION


def is_record_complete(draft: RecordDraft) -> bool:
    """Check if all non-trivial steps are satisfied."""
    return all(
        is_step_satisfied(step, draft)
        for step in ORDERED_STEPS
        if step not in (Step.GREETING, Step.CONFIRMATION)
    )


def next_step(step: Step) -> Step:
    """Advance to the next step in the ordered sequence.

    Terminal at CONFIRMATION: returns itself.
    """
    index = ORDERED_STEPS.index(step)
    if index + 1 < len(ORDERED_STEPS):
        return ORDERED_STEPS[index + 1]
    return step  # terminal
