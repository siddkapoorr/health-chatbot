"""Application entry point: wire config, logging, LLM, validator, and run the session."""

import logging
import sys

from health_intake.cli import run_session
from health_intake.config import get_settings
from health_intake.engine.orchestrator import Orchestrator
from health_intake.llm.client import OpenAIClient
from health_intake.logging_config import configure_logging
from health_intake.validation.address import GoogleAddressValidator, SkipAddressValidator

logger = logging.getLogger(__name__)


def main() -> int:
    try:
        settings = get_settings()
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    configure_logging(settings.log_level, log_dir=settings.output_dir.parent / "logs")

    google_key = (
        settings.google_maps_api_key.get_secret_value() if settings.google_maps_api_key else ""
    )
    validator = (
        SkipAddressValidator()
        if settings.skip_address_validation
        else GoogleAddressValidator(api_key=google_key)
    )
    orchestrator = Orchestrator(
        llm=OpenAIClient(
            api_key=settings.openai_api_key.get_secret_value(),
            model=settings.openai_model,
        ),
        address_validator=validator,
        settings=settings,
    )

    try:
        run_session(orchestrator, input_fn=input, output_fn=print)
    except Exception:  # noqa: BLE001
        logger.exception("Unexpected error during session")
        print("Sorry, something went wrong. Please try again later.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
