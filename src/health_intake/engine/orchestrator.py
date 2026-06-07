"""Deterministic state machine driving the intake conversation."""

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from health_intake.appointments.provider import available_slots as get_available_slots
from health_intake.appointments.provider import format_slots, get_slot
from health_intake.config import Settings
from health_intake.engine.extraction import apply_extraction
from health_intake.engine.steps import is_step_satisfied, next_step
from health_intake.llm.client import LLMClient
from health_intake.llm.prompts import SYSTEM_PROMPT, build_situation
from health_intake.models.conversation import ChatMessage, ConversationState, RecordDraft, Step
from health_intake.models.patient import Appointment, IntakeRecord, build_record
from health_intake.storage.writer import write_record
from health_intake.validation.address import AddressValidator

logger = logging.getLogger(__name__)

_GREETING = "Hello! I'm here to help check you in. To start, could you tell me your full name?"


@dataclass
class TurnResult:
    state: ConversationState
    message: str
    is_complete: bool
    record: IntakeRecord | None = None


class Orchestrator:
    def __init__(
        self,
        llm: LLMClient,
        address_validator: AddressValidator,
        settings: Settings,
        now_fn: Callable[[], datetime] = datetime.now,
    ) -> None:
        self._llm = llm
        self._address = address_validator
        self._settings = settings
        self._now = now_fn

    def start(self) -> tuple[ConversationState, str]:
        state = ConversationState(
            session_id=uuid.uuid4().hex[:12],
            current_step=Step.PATIENT_INFO,
            draft=RecordDraft(),
            messages=(ChatMessage(role="assistant", content=_GREETING),),
        )
        return state, _GREETING

    def handle_turn(self, state: ConversationState, user_input: str) -> TurnResult:
        state = state.with_message(ChatMessage(role="user", content=user_input))
        try:
            extraction = self._llm.extract(SYSTEM_PROMPT, state.messages)
        except Exception:  # noqa: BLE001
            logger.exception("Extraction call failed")
            return self._reply(state, "Sorry, I had trouble there. Could you say that again?")

        draft, errors = apply_extraction(state.draft, extraction)
        draft, address_errors = self._finalize_address(state.current_step, draft)
        draft, appt_errors = self._finalize_appointment(state.current_step, draft, extraction)
        errors = errors + address_errors + appt_errors
        state = state.with_draft(draft)

        if not errors:
            state = self._advance(state)

        if state.current_step == Step.CONFIRMATION:
            return self._complete(state)

        situation = build_situation(
            step=state.current_step,
            errors=errors,
            advancing=not errors,
            extra=self._appointment_listing(state.current_step),
        )
        message = self._safe_generate(state, situation)
        return self._reply(state, message)

    def _advance(self, state: ConversationState) -> ConversationState:
        while state.current_step != Step.CONFIRMATION and is_step_satisfied(
            state.current_step, state.draft
        ):
            state = state.with_step(next_step(state.current_step))
        return state

    def _finalize_address(self, step: Step, draft: RecordDraft) -> tuple[RecordDraft, list[str]]:
        if step != Step.ADDRESS or draft.address_validated:
            return draft, []
        if not all((draft.street, draft.city, draft.state, draft.zip_code)):
            return draft, []
        result = self._address.validate(
            draft.street or "", draft.city or "", draft.state or "", draft.zip_code or ""
        )
        if result.ok:
            return draft.model_copy(
                update={"address_validated": True, "address_formatted": result.formatted}
            ), []
        detail = f" Unconfirmed: {', '.join(result.missing)}." if result.missing else ""
        return draft, [(result.error or "Address could not be validated.") + detail]

    def _finalize_appointment(
        self, step: Step, draft: RecordDraft, extraction: object
    ) -> tuple[RecordDraft, list[str]]:
        if step != Step.APPOINTMENT:
            return draft, []
        choice = getattr(extraction, "appointment_choice", None)
        if not choice:
            return draft, []
        slot = get_slot(self._now(), choice)
        if slot is None:
            return draft, ["That appointment slot isn't available. Please pick one from the list."]
        return draft.model_copy(update={"slot_id": slot.slot_id}), []

    def _appointment_listing(self, step: Step) -> str:
        if step != Step.APPOINTMENT:
            return ""
        return "Available slots to present:\n" + format_slots(get_available_slots(self._now()))

    def _complete(self, state: ConversationState) -> TurnResult:
        slot = get_slot(self._now(), state.draft.slot_id or "")
        assert slot is not None
        appointment = Appointment(
            slot_id=slot.slot_id,
            provider_name=slot.provider_name,
            specialty=slot.specialty,
            start_time=slot.start_time,
        )
        record = build_record(state.session_id, state.draft, appointment)
        write_record(record, self._settings.output_dir)
        situation = build_situation(step=Step.CONFIRMATION, errors=[], advancing=True)
        message = self._safe_generate(state, situation)
        state = state.with_message(ChatMessage(role="assistant", content=message))
        return TurnResult(state=state, message=message, is_complete=True, record=record)

    def _safe_generate(self, state: ConversationState, situation: str) -> str:
        try:
            return self._llm.generate(SYSTEM_PROMPT, state.messages, situation)
        except Exception:  # noqa: BLE001
            logger.exception("Generation call failed")
            return "Thanks — let's continue."

    def _reply(self, state: ConversationState, message: str) -> TurnResult:
        state = state.with_message(ChatMessage(role="assistant", content=message))
        return TurnResult(state=state, message=message, is_complete=False)
