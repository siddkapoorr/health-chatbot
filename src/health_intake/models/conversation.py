"""Immutable conversation state and the in-progress record draft."""

from datetime import date
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class Step(str, Enum):
    GREETING = "greeting"
    PATIENT_INFO = "patient_info"
    INSURANCE = "insurance"
    CHIEF_COMPLAINT = "chief_complaint"
    ADDRESS = "address"
    APPOINTMENT = "appointment"
    CONFIRMATION = "confirmation"


class RecordDraft(BaseModel):
    """All-optional working copy of collected values; frozen for immutability."""

    model_config = ConfigDict(frozen=True)

    full_name: str | None = None
    date_of_birth: date | None = None
    payer_name: str | None = None
    insurance_id: str | None = None
    chief_complaint: str | None = None
    street: str | None = None
    city: str | None = None
    state: str | None = None
    zip_code: str | None = None
    address_validated: bool = False
    address_formatted: str | None = None
    slot_id: str | None = None


class ConversationState(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_id: str
    current_step: Step
    draft: RecordDraft
    messages: tuple[ChatMessage, ...] = ()

    def with_message(self, message: ChatMessage) -> "ConversationState":
        return self.model_copy(update={"messages": (*self.messages, message)})

    def with_draft(self, draft: RecordDraft) -> "ConversationState":
        return self.model_copy(update={"draft": draft})

    def with_step(self, step: Step) -> "ConversationState":
        return self.model_copy(update={"current_step": step})
