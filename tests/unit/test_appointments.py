from datetime import datetime

from health_intake.appointments.provider import available_slots, format_slots, get_slot

NOW = datetime(2026, 6, 6, 12, 0)


def test_all_available_slots_are_in_the_future():
    slots = available_slots(NOW)
    assert slots
    assert all(slot.start_time > NOW for slot in slots)


def test_get_slot_returns_matching_slot():
    slots = available_slots(NOW)
    target = slots[0]
    assert get_slot(NOW, target.slot_id) == target


def test_get_slot_returns_none_for_unknown_id():
    assert get_slot(NOW, "does-not-exist") is None


def test_format_slots_produces_numbered_list():
    slots = available_slots(NOW)
    formatted = format_slots(slots)
    assert formatted
    lines = formatted.split("\n")
    assert len(lines) == len(slots)
    # Verify first line starts with "1." and contains slot info
    assert lines[0].startswith("1.")
    assert "[" in lines[0] and "]" in lines[0]
    assert "(" in lines[0] and ")" in lines[0]
    assert "at" in lines[0]
