from health_intake.models.conversation import (
    ChatMessage,
    ConversationState,
    RecordDraft,
    Step,
)


def test_state_updates_are_immutable():
    state = ConversationState(session_id="s1", current_step=Step.GREETING, draft=RecordDraft())

    updated = state.with_message(ChatMessage(role="user", content="hi"))

    assert state.messages == ()  # original unchanged
    assert updated.messages[0].content == "hi"
    assert updated is not state


def test_with_draft_replaces_draft():
    state = ConversationState(session_id="s1", current_step=Step.PATIENT_INFO, draft=RecordDraft())

    updated = state.with_draft(state.draft.model_copy(update={"full_name": "Jane"}))

    assert state.draft.full_name is None
    assert updated.draft.full_name == "Jane"
