"""Mock provider directory and appointment availability.

Availability rules (also documented in README):
- Slots are generated for the next two business mornings/afternoons.
- A slot is only selectable if its start time is in the future.
- A slot id that is not in the generated set is rejected.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

_PROVIDERS = (
    ("Dr. Alice Chen", "Family Medicine"),
    ("Dr. Ben Okafor", "Internal Medicine"),
)
_HOURS = (9, 14)
_DAYS_AHEAD = (1, 2)


@dataclass(frozen=True)
class Slot:
    slot_id: str
    provider_name: str
    specialty: str
    start_time: datetime


def _generate(now: datetime) -> tuple[Slot, ...]:
    base = now.replace(minute=0, second=0, microsecond=0)
    slots: list[Slot] = []
    for day in _DAYS_AHEAD:
        for hour in _HOURS:
            start = (base + timedelta(days=day)).replace(hour=hour)
            for name, specialty in _PROVIDERS:
                last = name.split()[-1].lower()
                slot_id = f"{last}-{start:%Y%m%d-%H%M}"
                slots.append(Slot(slot_id, name, specialty, start))
    return tuple(slots)


def available_slots(now: datetime) -> tuple[Slot, ...]:
    return tuple(slot for slot in _generate(now) if slot.start_time > now)


def get_slot(now: datetime, slot_id: str) -> Slot | None:
    return next((slot for slot in available_slots(now) if slot.slot_id == slot_id), None)


def format_slots(slots: tuple[Slot, ...]) -> str:
    """Human-readable numbered list for presenting choices to the user / LLM."""
    return "\n".join(
        f"{i}. [{slot.slot_id}] {slot.provider_name} ({slot.specialty}) — "
        f"{slot.start_time:%a %b %d at %I:%M %p}"
        for i, slot in enumerate(slots, start=1)
    )
