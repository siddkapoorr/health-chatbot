"""Terminal I/O loop. Core loop is decoupled from real stdin/stdout for testing."""

import logging
from collections.abc import Callable

from rich.console import Console
from rich.table import Table

from health_intake.engine.orchestrator import Orchestrator
from health_intake.models.patient import IntakeRecord

logger = logging.getLogger(__name__)
_EXIT_WORDS = {"quit", "exit"}


def run_session(
    orchestrator: Orchestrator,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> IntakeRecord | None:
    """Drive a full intake session. Returns the completed record, or None if aborted."""
    state, greeting = orchestrator.start()
    output_fn(greeting)

    while True:
        try:
            user_input = input_fn("> ").strip()
        except (EOFError, KeyboardInterrupt):
            output_fn("\nSession ended.")
            return None
        if user_input.lower() in _EXIT_WORDS:
            output_fn("Session ended. Take care!")
            return None
        if not user_input:
            continue

        result = orchestrator.handle_turn(state, user_input)
        state = result.state
        output_fn(result.message)
        if result.is_complete and result.record is not None:
            output_fn(_render_summary(result.record))
            return result.record


def _render_summary(record: IntakeRecord) -> str:
    table = Table(title="Appointment Confirmation")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Patient", record.patient.full_name)
    table.add_row("Date of birth", record.patient.date_of_birth.isoformat())
    table.add_row("Insurance", record.insurance.payer_name)
    table.add_row("Insurance ID", record.insurance.insurance_id or "—")
    table.add_row("Chief complaint", record.chief_complaint)
    table.add_row("Address", record.address.formatted or record.address.street)
    table.add_row("Physician", record.appointment.provider_name)
    table.add_row("Appointment", record.appointment.start_time.strftime("%a %b %d at %I:%M %p"))
    console = Console()
    with console.capture() as capture:
        console.print(table)
    return capture.get()
