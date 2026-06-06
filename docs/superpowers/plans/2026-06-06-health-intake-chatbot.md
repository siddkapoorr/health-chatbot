# Health Intake Chatbot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a terminal LLM chatbot that conducts a forgiving, human-like patient intake conversation while deterministically validating each step before advancing, then writes a structured JSON record and shows a confirmation summary.

**Architecture:** A deterministic Python state machine (the orchestrator) owns the step flow and all validation. The LLM (OpenAI) does only two jobs per turn: extract structured field values from free text (Structured Outputs) and phrase the next natural reply. Python — never the LLM — decides validity and advancement.

**Tech Stack:** Python 3.11+, Poetry, Pydantic v2, pydantic-settings, OpenAI SDK, httpx (Google Address Validation API), tenacity, rich, pytest + pytest-cov + respx, ruff, mypy.

---

## File Structure

```
src/health_intake/
  __main__.py            # entry point: wires everything, runs the CLI loop
  cli.py                 # terminal I/O loop (testable via injected input/output fns)
  config.py              # pydantic-settings Settings + get_settings()
  logging_config.py      # logging setup + PII RedactionFilter
  models/
    patient.py           # Address, Insurance, PatientInfo, Appointment, IntakeRecord, build_record
    conversation.py      # ChatMessage, Step, RecordDraft, ConversationState (frozen/immutable)
  validation/
    fields.py            # FieldResult + per-field validators
    address.py           # AddressResult, AddressValidator Protocol, Google + Skip impls
  appointments/
    provider.py          # Slot, generate_slots, available_slots, get_slot
  llm/
    client.py            # ChatMessage->OpenAI; LLMClient Protocol, OpenAIClient, FakeLLMClient
    prompts.py           # system prompt + per-step guidance + situation builder
  engine/
    extraction.py        # FieldExtraction schema + apply_extraction
    steps.py             # ORDERED_STEPS, is_step_satisfied, next_step
    orchestrator.py      # TurnResult, Orchestrator.start/handle_turn
  storage/
    writer.py            # write_record
tests/
  unit/                  # one test file per module
  integration/           # full scripted conversation through the orchestrator
  conftest.py            # fixtures: FakeLLMClient builders, fake address validator
```

Immutability rule: `ConversationState` and `RecordDraft` are frozen Pydantic models; updates return copies via `model_copy`. Validators and results are frozen dataclasses.

---

## Task 1: Project scaffold & tooling

**Files:**
- Create: `pyproject.toml`, `src/health_intake/__init__.py`, `tests/__init__.py`, `tests/unit/__init__.py`, `tests/integration/__init__.py`
- Create: `tests/unit/test_smoke.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_smoke.py`:
```python
def test_package_imports():
    import health_intake

    assert health_intake.__version__ == "0.1.0"
```

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[tool.poetry]
name = "health-intake"
version = "0.1.0"
description = "Terminal LLM chatbot for patient intake"
authors = ["Your Name <you@example.com>"]
readme = "README.md"
packages = [{ include = "health_intake", from = "src" }]

[tool.poetry.dependencies]
python = "^3.11"
openai = "^1.40"
pydantic = "^2.7"
pydantic-settings = "^2.3"
httpx = "^0.27"
tenacity = "^8.4"
rich = "^13.7"

[tool.poetry.group.dev.dependencies]
pytest = "^8.2"
pytest-cov = "^5.0"
pytest-mock = "^3.14"
respx = "^0.21"
ruff = "^0.5"
mypy = "^1.10"

[tool.poetry.scripts]
health-intake = "health_intake.__main__:main"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

[tool.pytest.ini_options]
addopts = "--cov=health_intake --cov-report=term-missing"
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]

[tool.mypy]
python_version = "3.11"
strict = true
mypy_path = "src"
```

- [ ] **Step 3: Create package files**

`src/health_intake/__init__.py`:
```python
"""Terminal LLM chatbot for patient intake."""

__version__ = "0.1.0"
```

Create empty `tests/__init__.py`, `tests/unit/__init__.py`, `tests/integration/__init__.py`.

- [ ] **Step 4: Install and run the test**

Run: `poetry install && poetry run pytest tests/unit/test_smoke.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml poetry.lock src tests
git commit -m "chore: scaffold poetry project and tooling"
```

---

## Task 2: Configuration

**Files:**
- Create: `src/health_intake/config.py`
- Test: `tests/unit/test_config.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_config.py`:
```python
import pytest

from health_intake.config import Settings


def test_settings_load_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "g-test")

    settings = Settings()

    assert settings.openai_api_key == "sk-test"
    assert settings.openai_model == "gpt-4o-mini"
    assert settings.skip_address_validation is False


def test_missing_google_key_requires_skip_flag(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("GOOGLE_MAPS_API_KEY", raising=False)

    with pytest.raises(ValueError, match="GOOGLE_MAPS_API_KEY"):
        Settings()
```

- [ ] **Step 2: Run to verify it fails**

Run: `poetry run pytest tests/unit/test_config.py -v`
Expected: FAIL (`ModuleNotFoundError: health_intake.config`)

- [ ] **Step 3: Implement `config.py`**

```python
"""Application configuration loaded from environment / .env."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated runtime configuration. Fails fast on missing required keys."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = Field(..., min_length=1)
    openai_model: str = "gpt-4o-mini"
    google_maps_api_key: str | None = None
    skip_address_validation: bool = False
    log_level: str = "INFO"
    output_dir: Path = Path("./output")

    @model_validator(mode="after")
    def _require_google_key_unless_skipped(self) -> "Settings":
        if not self.skip_address_validation and not self.google_maps_api_key:
            raise ValueError(
                "GOOGLE_MAPS_API_KEY is required unless SKIP_ADDRESS_VALIDATION=true"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
```

- [ ] **Step 4: Run to verify it passes**

Run: `poetry run pytest tests/unit/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/health_intake/config.py tests/unit/test_config.py
git commit -m "feat: add validated settings with fail-fast key checks"
```

---

## Task 3: Logging & PII redaction

**Files:**
- Create: `src/health_intake/logging_config.py`
- Test: `tests/unit/test_logging.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_logging.py`:
```python
import logging

from health_intake.logging_config import RedactionFilter


def _redact(message: str) -> str:
    record = logging.LogRecord("t", logging.INFO, __file__, 1, message, None, None)
    RedactionFilter().filter(record)
    return record.getMessage()


def test_redacts_email():
    assert "[REDACTED_EMAIL]" in _redact("contact jane@example.com now")


def test_redacts_date_of_birth():
    assert "[REDACTED_DOB]" in _redact("dob 1990-03-05")


def test_redacts_long_digit_runs():
    assert "[REDACTED_NUM]" in _redact("id 123456789")


def test_keeps_safe_text():
    assert _redact("advanced to step address") == "advanced to step address"
```

- [ ] **Step 2: Run to verify it fails**

Run: `poetry run pytest tests/unit/test_logging.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement `logging_config.py`**

```python
"""Logging configuration with a PII redaction safety net.

Primary defense against PHI leakage is logging *events*, not raw field values.
This filter is a secondary safety net that masks anything that slips through.
"""

import logging
import re
from pathlib import Path

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b")
_LONG_NUM = re.compile(r"\b\d{5,}\b")


class RedactionFilter(logging.Filter):
    """Mask emails, dates of birth, and long digit runs in log messages."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        message = _EMAIL.sub("[REDACTED_EMAIL]", message)
        message = _DATE.sub("[REDACTED_DOB]", message)
        message = _LONG_NUM.sub("[REDACTED_NUM]", message)
        record.msg = message
        record.args = None
        return True


def configure_logging(level: str, log_dir: Path) -> None:
    """Configure root logging to console + rotating file, with redaction."""
    log_dir.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [
        logging.StreamHandler(),
        logging.FileHandler(log_dir / "session.log"),
    ]
    redaction = RedactionFilter()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    for handler in handlers:
        handler.addFilter(redaction)
        handler.setFormatter(formatter)
    logging.basicConfig(level=level.upper(), handlers=handlers, force=True)
```

- [ ] **Step 4: Run to verify it passes**

Run: `poetry run pytest tests/unit/test_logging.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/health_intake/logging_config.py tests/unit/test_logging.py
git commit -m "feat: add logging with PII redaction filter"
```

---

## Task 4: Domain models

**Files:**
- Create: `src/health_intake/models/__init__.py`, `src/health_intake/models/patient.py`
- Test: `tests/unit/test_patient_models.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_patient_models.py`:
```python
from datetime import date, datetime

from health_intake.models.patient import (
    Address,
    Appointment,
    Insurance,
    PatientInfo,
    build_record,
)
from health_intake.models.conversation import RecordDraft


def test_build_record_from_complete_draft():
    draft = RecordDraft(
        full_name="Jane Doe",
        date_of_birth=date(1990, 3, 5),
        payer_name="Acme Health",
        insurance_id="AH123",
        chief_complaint="sore throat",
        street="1600 Amphitheatre Pkwy",
        city="Mountain View",
        state="CA",
        zip_code="94043",
        address_validated=True,
        address_formatted="1600 Amphitheatre Pkwy, Mountain View, CA 94043, USA",
        slot_id="chen-20260607-0900",
    )
    appointment = Appointment(
        slot_id="chen-20260607-0900",
        provider_name="Dr. Alice Chen",
        specialty="Family Medicine",
        start_time=datetime(2026, 6, 7, 9, 0),
    )

    record = build_record("sess-1", draft, appointment)

    assert record.patient == PatientInfo(full_name="Jane Doe", date_of_birth=date(1990, 3, 5))
    assert record.insurance == Insurance(payer_name="Acme Health", insurance_id="AH123")
    assert record.address.formatted.endswith("USA")
    assert record.appointment.provider_name == "Dr. Alice Chen"
    assert record.session_id == "sess-1"
```

- [ ] **Step 2: Run to verify it fails**

Run: `poetry run pytest tests/unit/test_patient_models.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement models**

Create empty `src/health_intake/models/__init__.py`.

`src/health_intake/models/patient.py`:
```python
"""Pydantic domain models for the final intake record."""

from datetime import date, datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from health_intake.models.conversation import RecordDraft


class Address(BaseModel):
    street: str
    city: str
    state: str
    zip_code: str
    validated: bool = False
    formatted: str | None = None


class Insurance(BaseModel):
    payer_name: str
    insurance_id: str | None = None


class PatientInfo(BaseModel):
    full_name: str
    date_of_birth: date


class Appointment(BaseModel):
    slot_id: str
    provider_name: str
    specialty: str
    start_time: datetime


class IntakeRecord(BaseModel):
    session_id: str
    created_at: datetime
    patient: PatientInfo
    insurance: Insurance
    chief_complaint: str
    address: Address
    appointment: Appointment


def build_record(
    session_id: str, draft: "RecordDraft", appointment: Appointment
) -> IntakeRecord:
    """Assemble a complete IntakeRecord from a fully-collected draft.

    Assumes the draft has passed all step gates; callers must check completeness first.
    """
    assert draft.full_name and draft.date_of_birth and draft.payer_name
    assert draft.chief_complaint and draft.street and draft.city
    assert draft.state and draft.zip_code

    return IntakeRecord(
        session_id=session_id,
        created_at=datetime.now(),
        patient=PatientInfo(full_name=draft.full_name, date_of_birth=draft.date_of_birth),
        insurance=Insurance(payer_name=draft.payer_name, insurance_id=draft.insurance_id),
        chief_complaint=draft.chief_complaint,
        address=Address(
            street=draft.street,
            city=draft.city,
            state=draft.state,
            zip_code=draft.zip_code,
            validated=draft.address_validated,
            formatted=draft.address_formatted,
        ),
        appointment=appointment,
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `poetry run pytest tests/unit/test_patient_models.py -v`
Expected: PASS (after Task 5 defines `RecordDraft`; if running this task alone, implement Task 5 first — they are paired)

> Note: Tasks 4 and 5 are interdependent (`build_record` consumes `RecordDraft`). Implement Task 5's `conversation.py` before running Task 4's test.

- [ ] **Step 5: Commit**

```bash
git add src/health_intake/models
git commit -m "feat: add patient domain models and record builder"
```

---

## Task 5: Conversation state (immutable)

**Files:**
- Create: `src/health_intake/models/conversation.py`
- Test: `tests/unit/test_conversation_state.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_conversation_state.py`:
```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `poetry run pytest tests/unit/test_conversation_state.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement `conversation.py`**

```python
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `poetry run pytest tests/unit/test_conversation_state.py tests/unit/test_patient_models.py -v`
Expected: PASS (both Task 4 and Task 5 tests now green)

- [ ] **Step 5: Commit**

```bash
git add src/health_intake/models/conversation.py tests/unit/test_conversation_state.py tests/unit/test_patient_models.py
git commit -m "feat: add immutable conversation state and record draft"
```

---

## Task 6: Field validators

**Files:**
- Create: `src/health_intake/validation/__init__.py`, `src/health_intake/validation/fields.py`
- Test: `tests/unit/test_field_validators.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_field_validators.py`:
```python
from datetime import date

from health_intake.validation.fields import (
    validate_chief_complaint,
    validate_date_of_birth,
    validate_full_name,
    validate_insurance_id,
    validate_payer_name,
)


def test_valid_full_name():
    result = validate_full_name("Mary-Jane O'Connor")
    assert result.ok
    assert result.value == "Mary-Jane O'Connor"


def test_rejects_empty_name():
    result = validate_full_name("   ")
    assert not result.ok
    assert "name" in result.error.lower()


def test_valid_dob_parsed():
    result = validate_date_of_birth("1990-03-05")
    assert result.ok
    assert result.value == date(1990, 3, 5)


def test_rejects_future_dob():
    result = validate_date_of_birth("2999-01-01")
    assert not result.ok


def test_rejects_unparseable_dob():
    result = validate_date_of_birth("not a date")
    assert not result.ok


def test_optional_insurance_id_blank_is_ok():
    result = validate_insurance_id("")
    assert result.ok
    assert result.value is None


def test_payer_and_complaint_required():
    assert not validate_payer_name("").ok
    assert not validate_chief_complaint("").ok
    assert validate_chief_complaint("sore throat for 3 days").ok
```

- [ ] **Step 2: Run to verify it fails**

Run: `poetry run pytest tests/unit/test_field_validators.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement `fields.py`**

Create empty `src/health_intake/validation/__init__.py`.

```python
"""Deterministic per-field validators. The LLM never decides validity."""

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Generic, TypeVar

T = TypeVar("T")

MAX_AGE_YEARS = 120
MAX_COMPLAINT_LEN = 1000
_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z\s'\-]*$")
_INSURANCE_ID_RE = re.compile(r"^[A-Za-z0-9\-]+$")
_DATE_FORMATS = ("%Y-%m-%d", "%m/%d/%Y", "%d %B %Y", "%B %d, %Y")


@dataclass(frozen=True)
class FieldResult(Generic[T]):
    ok: bool
    value: T | None = None
    error: str | None = None


def validate_full_name(raw: str) -> FieldResult[str]:
    cleaned = raw.strip()
    if not cleaned:
        return FieldResult(ok=False, error="A full name is required.")
    if not _NAME_RE.match(cleaned):
        return FieldResult(ok=False, error="Name may only contain letters, spaces, hyphens, apostrophes.")
    return FieldResult(ok=True, value=cleaned)


def validate_date_of_birth(raw: str) -> FieldResult[date]:
    cleaned = raw.strip()
    parsed: date | None = None
    for fmt in _DATE_FORMATS:
        try:
            parsed = datetime.strptime(cleaned, fmt).date()
            break
        except ValueError:
            continue
    if parsed is None:
        return FieldResult(ok=False, error="Please give a date of birth like 1990-03-05.")
    today = date.today()
    if parsed >= today:
        return FieldResult(ok=False, error="Date of birth must be in the past.")
    if (today.year - parsed.year) > MAX_AGE_YEARS:
        return FieldResult(ok=False, error="That date of birth doesn't look plausible.")
    return FieldResult(ok=True, value=parsed)


def validate_payer_name(raw: str) -> FieldResult[str]:
    cleaned = raw.strip()
    if not cleaned:
        return FieldResult(ok=False, error="An insurance payer name is required.")
    return FieldResult(ok=True, value=cleaned)


def validate_insurance_id(raw: str) -> FieldResult[str]:
    cleaned = raw.strip()
    if not cleaned:
        return FieldResult(ok=True, value=None)  # optional
    if not _INSURANCE_ID_RE.match(cleaned):
        return FieldResult(ok=False, error="Insurance ID may only contain letters, numbers, hyphens.")
    return FieldResult(ok=True, value=cleaned)


def validate_chief_complaint(raw: str) -> FieldResult[str]:
    cleaned = raw.strip()
    if not cleaned:
        return FieldResult(ok=False, error="Please describe the reason for your visit.")
    if len(cleaned) > MAX_COMPLAINT_LEN:
        return FieldResult(ok=False, error="That description is too long.")
    return FieldResult(ok=True, value=cleaned)
```

- [ ] **Step 4: Run to verify it passes**

Run: `poetry run pytest tests/unit/test_field_validators.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/health_intake/validation tests/unit/test_field_validators.py
git commit -m "feat: add deterministic field validators"
```

---

## Task 7: Address validation

**Files:**
- Create: `src/health_intake/validation/address.py`
- Test: `tests/unit/test_address_validation.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_address_validation.py`:
```python
import httpx
import respx

from health_intake.validation.address import (
    GoogleAddressValidator,
    SkipAddressValidator,
)

_URL = "https://addressvalidation.googleapis.com/v1:validateAddress"


def test_skip_validator_assembles_formatted():
    result = SkipAddressValidator().validate("1 Main St", "Springfield", "IL", "62704")
    assert result.ok
    assert "Springfield" in result.formatted


@respx.mock
def test_google_validator_accepts_complete_address():
    respx.post(_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "result": {
                    "verdict": {"addressComplete": True, "hasUnconfirmedComponents": False},
                    "address": {"formattedAddress": "1600 Amphitheatre Pkwy, Mountain View, CA 94043, USA"},
                }
            },
        )
    )
    validator = GoogleAddressValidator(api_key="g-test")
    result = validator.validate("1600 Amphitheatre Pkwy", "Mountain View", "CA", "94043")
    assert result.ok
    assert result.formatted.endswith("USA")


@respx.mock
def test_google_validator_reports_unconfirmed():
    respx.post(_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "result": {
                    "verdict": {"addressComplete": False, "hasUnconfirmedComponents": True},
                    "address": {
                        "formattedAddress": "Nowhere",
                        "unconfirmedComponentTypes": ["route", "postal_code"],
                    },
                }
            },
        )
    )
    result = GoogleAddressValidator(api_key="g-test").validate("x", "y", "z", "00000")
    assert not result.ok
    assert "route" in result.missing
```

- [ ] **Step 2: Run to verify it fails**

Run: `poetry run pytest tests/unit/test_address_validation.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement `address.py`**

```python
"""Address validation via Google Address Validation API, with a skip fallback."""

import logging
from dataclasses import dataclass, field
from typing import Protocol

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

_ENDPOINT = "https://addressvalidation.googleapis.com/v1:validateAddress"
_TIMEOUT = 10.0


@dataclass(frozen=True)
class AddressResult:
    ok: bool
    formatted: str | None = None
    missing: tuple[str, ...] = field(default_factory=tuple)
    error: str | None = None


class AddressValidator(Protocol):
    def validate(self, street: str, city: str, state: str, zip_code: str) -> AddressResult: ...


class SkipAddressValidator:
    """Offline fallback: accepts the address as-is. Documented dev-only behavior."""

    def validate(self, street: str, city: str, state: str, zip_code: str) -> AddressResult:
        formatted = f"{street}, {city}, {state} {zip_code}"
        return AddressResult(ok=True, formatted=formatted)


class GoogleAddressValidator:
    def __init__(self, api_key: str, client: httpx.Client | None = None) -> None:
        self._api_key = api_key
        self._client = client or httpx.Client(timeout=_TIMEOUT)

    @retry(
        retry=retry_if_exception_type(httpx.TransportError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=5),
        reraise=True,
    )
    def _post(self, payload: dict) -> httpx.Response:
        return self._client.post(_ENDPOINT, params={"key": self._api_key}, json=payload)

    def validate(self, street: str, city: str, state: str, zip_code: str) -> AddressResult:
        payload = {
            "address": {
                "regionCode": "US",
                "addressLines": [street],
                "locality": city,
                "administrativeArea": state,
                "postalCode": zip_code,
            }
        }
        try:
            response = self._post(payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("Address validation request failed: %s", type(exc).__name__)
            return AddressResult(ok=False, error="Could not reach the address validation service.")

        result = response.json().get("result", {})
        verdict = result.get("verdict", {})
        address = result.get("address", {})
        if verdict.get("addressComplete") and not verdict.get("hasUnconfirmedComponents"):
            return AddressResult(ok=True, formatted=address.get("formattedAddress"))

        missing = tuple(address.get("unconfirmedComponentTypes", []))
        return AddressResult(
            ok=False,
            missing=missing,
            error="The address could not be fully confirmed.",
        )
```

- [ ] **Step 4: Run to verify it passes**

Run: `poetry run pytest tests/unit/test_address_validation.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/health_intake/validation/address.py tests/unit/test_address_validation.py
git commit -m "feat: add Google address validation with skip fallback"
```

---

## Task 8: Appointments (mock providers & slots)

**Files:**
- Create: `src/health_intake/appointments/__init__.py`, `src/health_intake/appointments/provider.py`
- Test: `tests/unit/test_appointments.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_appointments.py`:
```python
from datetime import datetime

from health_intake.appointments.provider import available_slots, get_slot

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
```

- [ ] **Step 2: Run to verify it fails**

Run: `poetry run pytest tests/unit/test_appointments.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement `provider.py`**

Create empty `src/health_intake/appointments/__init__.py`.

```python
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `poetry run pytest tests/unit/test_appointments.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/health_intake/appointments tests/unit/test_appointments.py
git commit -m "feat: add mock provider directory and slot availability"
```

---

## Task 9: Storage writer

**Files:**
- Create: `src/health_intake/storage/__init__.py`, `src/health_intake/storage/writer.py`
- Test: `tests/unit/test_writer.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_writer.py`:
```python
import json
from datetime import date, datetime

from health_intake.models.patient import (
    Address,
    Appointment,
    Insurance,
    IntakeRecord,
    PatientInfo,
)
from health_intake.storage.writer import write_record


def _record() -> IntakeRecord:
    return IntakeRecord(
        session_id="sess-42",
        created_at=datetime(2026, 6, 6, 12, 0),
        patient=PatientInfo(full_name="Jane Doe", date_of_birth=date(1990, 3, 5)),
        insurance=Insurance(payer_name="Acme", insurance_id=None),
        chief_complaint="cough",
        address=Address(street="1 Main", city="Town", state="CA", zip_code="90001"),
        appointment=Appointment(
            slot_id="chen-1", provider_name="Dr. Alice Chen",
            specialty="Family Medicine", start_time=datetime(2026, 6, 7, 9, 0),
        ),
    )


def test_write_record_creates_json_file(tmp_path):
    path = write_record(_record(), tmp_path)

    assert path.exists()
    assert path.name == "intake-sess-42.json"
    data = json.loads(path.read_text())
    assert data["patient"]["full_name"] == "Jane Doe"
```

- [ ] **Step 2: Run to verify it fails**

Run: `poetry run pytest tests/unit/test_writer.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement `writer.py`**

Create empty `src/health_intake/storage/__init__.py`.

```python
"""Persist a completed IntakeRecord as JSON."""

import logging
from pathlib import Path

from health_intake.models.patient import IntakeRecord

logger = logging.getLogger(__name__)


def write_record(record: IntakeRecord, output_dir: Path) -> Path:
    """Write the record to ``output_dir/intake-<session_id>.json`` and return the path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"intake-{record.session_id}.json"
    path.write_text(record.model_dump_json(indent=2))
    logger.info("Wrote intake record for session %s", record.session_id)
    return path
```

- [ ] **Step 4: Run to verify it passes**

Run: `poetry run pytest tests/unit/test_writer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/health_intake/storage tests/unit/test_writer.py
git commit -m "feat: add JSON record writer"
```

---

## Task 10: LLM client + fake

**Files:**
- Create: `src/health_intake/llm/__init__.py`, `src/health_intake/llm/client.py`, `src/health_intake/engine/__init__.py`, `src/health_intake/engine/extraction.py`
- Test: `tests/unit/test_fake_llm.py`

> `FieldExtraction` lives in `engine/extraction.py`; the client returns it, so create the schema first.

- [ ] **Step 1: Write the failing test**

`tests/unit/test_fake_llm.py`:
```python
from health_intake.engine.extraction import FieldExtraction
from health_intake.llm.client import FakeLLMClient
from health_intake.models.conversation import ChatMessage


def test_fake_client_returns_scripted_extraction_and_reply():
    client = FakeLLMClient(
        extractions=[FieldExtraction(full_name="Jane Doe")],
        replies=["Thanks, Jane!"],
    )
    messages = [ChatMessage(role="user", content="I'm Jane Doe")]

    extraction = client.extract("system", messages)
    reply = client.generate("system", messages, "ask for DOB")

    assert extraction.full_name == "Jane Doe"
    assert reply == "Thanks, Jane!"
```

- [ ] **Step 2: Run to verify it fails**

Run: `poetry run pytest tests/unit/test_fake_llm.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement extraction schema and client**

Create empty `src/health_intake/engine/__init__.py`.

`src/health_intake/engine/extraction.py` (schema only for now; `apply_extraction` added in Task 13):
```python
"""Structured-output schema the LLM fills from user messages."""

from pydantic import BaseModel


class FieldExtraction(BaseModel):
    """Fields the LLM may extract from the conversation. All optional.

    Raw strings only — Python validates/parses them. ``appointment_choice`` should be
    the slot id shown in the presented list when the user picks a time.
    """

    full_name: str | None = None
    date_of_birth: str | None = None
    payer_name: str | None = None
    insurance_id: str | None = None
    chief_complaint: str | None = None
    street: str | None = None
    city: str | None = None
    state: str | None = None
    zip_code: str | None = None
    appointment_choice: str | None = None
    user_asked_question: bool = False
```

Create empty `src/health_intake/llm/__init__.py`.

`src/health_intake/llm/client.py`:
```python
"""LLM client protocol, OpenAI implementation, and a scripted fake for tests."""

import logging
from collections.abc import Sequence
from typing import Protocol

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from health_intake.engine.extraction import FieldExtraction
from health_intake.models.conversation import ChatMessage

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    def extract(self, system: str, messages: Sequence[ChatMessage]) -> FieldExtraction: ...

    def generate(self, system: str, messages: Sequence[ChatMessage], situation: str) -> str: ...


def _to_openai(system: str, messages: Sequence[ChatMessage]) -> list[dict]:
    payload: list[dict] = [{"role": "system", "content": system}]
    payload.extend({"role": m.role, "content": m.content} for m in messages if m.role != "system")
    return payload


class OpenAIClient:
    def __init__(self, api_key: str, model: str) -> None:
        self._client = OpenAI(api_key=api_key)
        self._model = model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
    def extract(self, system: str, messages: Sequence[ChatMessage]) -> FieldExtraction:
        completion = self._client.beta.chat.completions.parse(
            model=self._model,
            messages=_to_openai(system, messages),
            response_format=FieldExtraction,
            temperature=0,
        )
        parsed = completion.choices[0].message.parsed
        return parsed or FieldExtraction()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
    def generate(self, system: str, messages: Sequence[ChatMessage], situation: str) -> str:
        convo = _to_openai(system, messages)
        convo.append({"role": "system", "content": f"Situation: {situation}"})
        completion = self._client.chat.completions.create(
            model=self._model, messages=convo, temperature=0.3
        )
        return completion.choices[0].message.content or ""


class FakeLLMClient:
    """Deterministic scripted client for tests. Pops scripted values per call."""

    def __init__(self, extractions: list[FieldExtraction], replies: list[str]) -> None:
        self._extractions = list(extractions)
        self._replies = list(replies)

    def extract(self, system: str, messages: Sequence[ChatMessage]) -> FieldExtraction:
        return self._extractions.pop(0) if self._extractions else FieldExtraction()

    def generate(self, system: str, messages: Sequence[ChatMessage], situation: str) -> str:
        return self._replies.pop(0) if self._replies else "(no reply)"
```

- [ ] **Step 4: Run to verify it passes**

Run: `poetry run pytest tests/unit/test_fake_llm.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/health_intake/llm src/health_intake/engine tests/unit/test_fake_llm.py
git commit -m "feat: add LLM client protocol, OpenAI impl, fake, extraction schema"
```

---

## Task 11: Prompts & situation builder

**Files:**
- Create: `src/health_intake/llm/prompts.py`
- Test: `tests/unit/test_prompts.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_prompts.py`:
```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `poetry run pytest tests/unit/test_prompts.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement `prompts.py`**

```python
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
        lines.append("The patient's last input had problems. Re-ask, mentioning: " + " ".join(errors))
    elif advancing:
        lines.append("All required info for this step is valid. Acknowledge and advance to the next topic.")
    if extra:
        lines.append(extra)
    return "\n".join(lines)
```

- [ ] **Step 4: Run to verify it passes**

Run: `poetry run pytest tests/unit/test_prompts.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/health_intake/llm/prompts.py tests/unit/test_prompts.py
git commit -m "feat: add system prompt and situation builder"
```

---

## Task 12: Step gates

**Files:**
- Create: `src/health_intake/engine/steps.py`
- Test: `tests/unit/test_steps.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_steps.py`:
```python
from health_intake.engine.steps import ORDERED_STEPS, is_step_satisfied, next_step
from health_intake.models.conversation import RecordDraft, Step


def test_patient_info_requires_name_and_dob():
    from datetime import date

    assert not is_step_satisfied(Step.PATIENT_INFO, RecordDraft(full_name="Jane"))
    draft = RecordDraft(full_name="Jane", date_of_birth=date(1990, 1, 1))
    assert is_step_satisfied(Step.PATIENT_INFO, draft)


def test_insurance_id_is_optional():
    assert is_step_satisfied(Step.INSURANCE, RecordDraft(payer_name="Acme"))


def test_address_requires_validation_flag():
    draft = RecordDraft(street="1 Main", city="Town", state="CA", zip_code="90001")
    assert not is_step_satisfied(Step.ADDRESS, draft)
    assert is_step_satisfied(Step.ADDRESS, draft.model_copy(update={"address_validated": True}))


def test_next_step_advances_in_order():
    assert next_step(Step.GREETING) == Step.PATIENT_INFO
    assert next_step(Step.CONFIRMATION) == Step.CONFIRMATION  # terminal
    assert ORDERED_STEPS[0] == Step.GREETING
```

- [ ] **Step 2: Run to verify it fails**

Run: `poetry run pytest tests/unit/test_steps.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement `steps.py`**

```python
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
    return all(
        is_step_satisfied(step, draft)
        for step in ORDERED_STEPS
        if step not in (Step.GREETING, Step.CONFIRMATION)
    )


def next_step(step: Step) -> Step:
    index = ORDERED_STEPS.index(step)
    if index + 1 < len(ORDERED_STEPS):
        return ORDERED_STEPS[index + 1]
    return step  # terminal
```

- [ ] **Step 4: Run to verify it passes**

Run: `poetry run pytest tests/unit/test_steps.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/health_intake/engine/steps.py tests/unit/test_steps.py
git commit -m "feat: add ordered steps and validation gates"
```

---

## Task 13: Orchestrator (the state machine)

**Files:**
- Modify: `src/health_intake/engine/extraction.py` (add `apply_extraction`)
- Create: `src/health_intake/engine/orchestrator.py`
- Test: `tests/unit/test_extraction_apply.py`, `tests/integration/test_orchestrator_flow.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Write the failing unit test for `apply_extraction`**

`tests/unit/test_extraction_apply.py`:
```python
from health_intake.engine.extraction import FieldExtraction, apply_extraction
from health_intake.models.conversation import RecordDraft


def test_apply_valid_fields_updates_draft():
    draft = RecordDraft()
    extraction = FieldExtraction(full_name="Jane Doe", date_of_birth="1990-03-05")

    new_draft, errors = apply_extraction(draft, extraction)

    assert new_draft.full_name == "Jane Doe"
    assert new_draft.date_of_birth is not None
    assert errors == []
    assert draft.full_name is None  # original unchanged (immutability)


def test_apply_invalid_dob_records_error_and_skips_field():
    draft = RecordDraft()
    extraction = FieldExtraction(date_of_birth="not a date")

    new_draft, errors = apply_extraction(draft, extraction)

    assert new_draft.date_of_birth is None
    assert any("date of birth" in e.lower() for e in errors)
```

- [ ] **Step 2: Run to verify it fails**

Run: `poetry run pytest tests/unit/test_extraction_apply.py -v`
Expected: FAIL (`ImportError: cannot import name 'apply_extraction'`)

- [ ] **Step 3: Add `apply_extraction` to `extraction.py`**

Append to `src/health_intake/engine/extraction.py`:
```python
from health_intake.models.conversation import RecordDraft
from health_intake.validation.fields import (
    validate_chief_complaint,
    validate_date_of_birth,
    validate_full_name,
    validate_insurance_id,
    validate_payer_name,
)


def apply_extraction(
    draft: RecordDraft, extraction: FieldExtraction
) -> tuple[RecordDraft, list[str]]:
    """Validate each provided field and return a new draft plus any error messages.

    Address validation and appointment matching are handled by the orchestrator
    because they require external services / runtime data. Raw address parts and the
    appointment choice are copied through here for the orchestrator to finalize.
    """
    updates: dict[str, object] = {}
    errors: list[str] = []

    def apply(raw: str | None, validator, key: str) -> None:
        if raw is None:
            return
        result = validator(raw)
        if result.ok:
            updates[key] = result.value
        elif result.error:
            errors.append(result.error)

    apply(extraction.full_name, validate_full_name, "full_name")
    apply(extraction.date_of_birth, validate_date_of_birth, "date_of_birth")
    apply(extraction.payer_name, validate_payer_name, "payer_name")
    apply(extraction.insurance_id, validate_insurance_id, "insurance_id")
    apply(extraction.chief_complaint, validate_chief_complaint, "chief_complaint")

    # Raw address parts pass through unvalidated here; the API check happens in the
    # orchestrator once all four parts are present.
    for key in ("street", "city", "state", "zip_code"):
        value = getattr(extraction, key)
        if value:
            updates[key] = value.strip()

    return draft.model_copy(update=updates), errors
```

- [ ] **Step 4: Run the unit test to verify it passes**

Run: `poetry run pytest tests/unit/test_extraction_apply.py -v`
Expected: PASS

- [ ] **Step 5: Write `conftest.py` fixtures**

`tests/conftest.py`:
```python
from health_intake.config import Settings
from health_intake.validation.address import AddressResult


class FakeAddressValidator:
    def __init__(self, ok: bool = True) -> None:
        self._ok = ok

    def validate(self, street: str, city: str, state: str, zip_code: str) -> AddressResult:
        if self._ok:
            return AddressResult(ok=True, formatted=f"{street}, {city}, {state} {zip_code}, USA")
        return AddressResult(ok=False, missing=("postal_code",), error="Unconfirmed address.")


def make_settings(tmp_path) -> Settings:
    return Settings(
        openai_api_key="sk-test",
        skip_address_validation=True,
        output_dir=tmp_path,
    )
```

- [ ] **Step 6: Write the failing integration test**

`tests/integration/test_orchestrator_flow.py`:
```python
from datetime import datetime

from health_intake.appointments.provider import available_slots
from health_intake.engine.extraction import FieldExtraction
from health_intake.engine.orchestrator import Orchestrator
from health_intake.llm.client import FakeLLMClient
from health_intake.models.conversation import ConversationState, RecordDraft, Step
from tests.conftest import FakeAddressValidator, make_settings

NOW = datetime(2026, 6, 6, 12, 0)


def _orchestrator(llm, settings):
    return Orchestrator(
        llm=llm,
        address_validator=FakeAddressValidator(),
        settings=settings,
        now_fn=lambda: NOW,
    )


def test_happy_path_collects_record_and_writes_json(tmp_path):
    settings = make_settings(tmp_path)
    slot = available_slots(NOW)[0]
    scripted = [
        FieldExtraction(full_name="Jane Doe", date_of_birth="1990-03-05"),
        FieldExtraction(payer_name="Acme Health", insurance_id="AH123"),
        FieldExtraction(chief_complaint="sore throat for 3 days"),
        FieldExtraction(street="1600 Amphitheatre Pkwy", city="Mountain View", state="CA", zip_code="94043"),
        FieldExtraction(appointment_choice=slot.slot_id),
    ]
    replies = ["ok1", "ok2", "ok3", "ok4", "Thanks, all set!"]
    llm = FakeLLMClient(extractions=scripted, replies=replies)
    orch = _orchestrator(llm, settings)

    state, _ = orch.start()
    inputs = [
        "I'm Jane Doe, born 1990-03-05",
        "Acme Health, AH123",
        "sore throat for 3 days",
        "1600 Amphitheatre Pkwy, Mountain View, CA 94043",
        f"I'll take {slot.slot_id}",
    ]
    result = None
    for text in inputs:
        result = orch.handle_turn(state, text)
        state = result.state

    assert result.is_complete
    assert result.record is not None
    assert result.record.patient.full_name == "Jane Doe"
    assert (tmp_path / f"intake-{result.record.session_id}.json").exists()


def test_invalid_dob_blocks_advance(tmp_path):
    settings = make_settings(tmp_path)
    llm = FakeLLMClient(
        extractions=[FieldExtraction(date_of_birth="2999-01-01")],
        replies=["Please re-check your date of birth."],
    )
    orch = _orchestrator(llm, settings)
    state = ConversationState(
        session_id="s1", current_step=Step.PATIENT_INFO, draft=RecordDraft(full_name="Jane")
    )

    result = orch.handle_turn(state, "born in 2999")

    assert not result.is_complete
    assert result.state.current_step == Step.PATIENT_INFO  # did not advance
    assert result.state.draft.date_of_birth is None
```

- [ ] **Step 7: Run to verify the integration test fails**

Run: `poetry run pytest tests/integration/test_orchestrator_flow.py -v`
Expected: FAIL (`ModuleNotFoundError: health_intake.engine.orchestrator`)

- [ ] **Step 8: Implement `orchestrator.py`**

```python
"""Deterministic state machine driving the intake conversation."""

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from health_intake.appointments.provider import format_slots, get_slot
from health_intake.appointments.provider import available_slots as get_available_slots
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

_GREETING = (
    "Hello! I'm here to help check you in. To start, could you tell me your full name?"
)


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
        except Exception:  # noqa: BLE001 - surfaced to user, logged with stack
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

    # --- helpers -------------------------------------------------------------

    def _advance(self, state: ConversationState) -> ConversationState:
        while state.current_step != Step.CONFIRMATION and is_step_satisfied(
            state.current_step, state.draft
        ):
            state = state.with_step(next_step(state.current_step))
        return state

    def _finalize_address(
        self, step: Step, draft: RecordDraft
    ) -> tuple[RecordDraft, list[str]]:
        if step != Step.ADDRESS or draft.address_validated:
            return draft, []
        if not all((draft.street, draft.city, draft.state, draft.zip_code)):
            return draft, []  # still collecting parts; not an error yet
        result = self._address.validate(draft.street, draft.city, draft.state, draft.zip_code)
        if result.ok:
            return draft.model_copy(
                update={"address_validated": True, "address_formatted": result.formatted}
            ), []
        detail = f" Unconfirmed: {', '.join(result.missing)}." if result.missing else ""
        return draft, [(result.error or "Address could not be validated.") + detail]

    def _finalize_appointment(
        self, step: Step, draft: RecordDraft, extraction
    ) -> tuple[RecordDraft, list[str]]:
        if step != Step.APPOINTMENT or not extraction.appointment_choice:
            return draft, []
        slot = get_slot(self._now(), extraction.appointment_choice)
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
```

- [ ] **Step 9: Run all orchestrator tests to verify they pass**

Run: `poetry run pytest tests/unit/test_extraction_apply.py tests/integration/test_orchestrator_flow.py -v`
Expected: PASS (happy path writes JSON; invalid DOB does not advance)

- [ ] **Step 10: Commit**

```bash
git add src/health_intake/engine tests/conftest.py tests/unit/test_extraction_apply.py tests/integration/test_orchestrator_flow.py
git commit -m "feat: add orchestrator state machine with validation gates"
```

---

## Task 14: CLI loop & entry point

**Files:**
- Create: `src/health_intake/cli.py`, `src/health_intake/__main__.py`
- Test: `tests/unit/test_cli.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_cli.py`:
```python
from datetime import datetime

from health_intake.appointments.provider import available_slots
from health_intake.cli import run_session
from health_intake.engine.extraction import FieldExtraction
from health_intake.engine.orchestrator import Orchestrator
from health_intake.llm.client import FakeLLMClient
from tests.conftest import FakeAddressValidator, make_settings

NOW = datetime(2026, 6, 6, 12, 0)


def test_run_session_drives_until_complete(tmp_path):
    settings = make_settings(tmp_path)
    slot = available_slots(NOW)[0]
    llm = FakeLLMClient(
        extractions=[
            FieldExtraction(full_name="Jane Doe", date_of_birth="1990-03-05"),
            FieldExtraction(payer_name="Acme"),
            FieldExtraction(chief_complaint="cough"),
            FieldExtraction(street="1600 Amphitheatre Pkwy", city="Mountain View", state="CA", zip_code="94043"),
            FieldExtraction(appointment_choice=slot.slot_id),
        ],
        replies=["a", "b", "c", "d", "done"],
    )
    orch = Orchestrator(llm, FakeAddressValidator(), settings, now_fn=lambda: NOW)

    scripted_inputs = iter(["Jane Doe 1990-03-05", "Acme", "cough", "1600 Amphitheatre Pkwy, Mountain View, CA 94043", slot.slot_id])
    outputs: list[str] = []

    record = run_session(orch, input_fn=lambda _: next(scripted_inputs), output_fn=outputs.append)

    assert record is not None
    assert record.patient.full_name == "Jane Doe"
    assert any("done" in line for line in outputs)
```

- [ ] **Step 2: Run to verify it fails**

Run: `poetry run pytest tests/unit/test_cli.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement `cli.py`**

```python
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
```

- [ ] **Step 4: Implement `__main__.py`**

`src/health_intake/__main__.py`:
```python
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

    validator = (
        SkipAddressValidator()
        if settings.skip_address_validation
        else GoogleAddressValidator(api_key=settings.google_maps_api_key or "")
    )
    orchestrator = Orchestrator(
        llm=OpenAIClient(api_key=settings.openai_api_key, model=settings.openai_model),
        address_validator=validator,
        settings=settings,
    )

    try:
        run_session(orchestrator, input_fn=input, output_fn=print)
    except Exception:  # noqa: BLE001 - last-resort guard
        logger.exception("Unexpected error during session")
        print("Sorry, something went wrong. Please try again later.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run to verify it passes**

Run: `poetry run pytest tests/unit/test_cli.py -v`
Expected: PASS

- [ ] **Step 6: Run the full suite with coverage**

Run: `poetry run pytest`
Expected: PASS, coverage ≥ 80%. If any module is under-covered, add a focused unit test before continuing.

- [ ] **Step 7: Commit**

```bash
git add src/health_intake/cli.py src/health_intake/__main__.py tests/unit/test_cli.py
git commit -m "feat: add CLI session loop and entry point"
```

---

## Task 15: README, env example, lint/type pass, GitHub

**Files:**
- Create: `README.md`, `.env.example`
- Verify: lint, types, tests

- [ ] **Step 1: Create `.env.example`**

```bash
# Required
OPENAI_API_KEY=sk-your-key-here

# Required unless SKIP_ADDRESS_VALIDATION=true
GOOGLE_MAPS_API_KEY=your-google-key-here

# Optional (defaults shown)
OPENAI_MODEL=gpt-4o-mini
LOG_LEVEL=INFO
OUTPUT_DIR=./output
SKIP_ADDRESS_VALIDATION=false
```

- [ ] **Step 2: Create `README.md`**

Include these sections (write real content, no placeholders):
- **Overview** — what the bot does; synthetic-data-only, real-patient-minded.
- **Architecture** — deterministic orchestrator + LLM extraction/generation layer; link to the design spec; the two-call-per-turn rationale.
- **Setup** — `poetry install`, copy `.env.example` to `.env`, where to get an OpenAI key and how to enable the Google Address Validation API (free tier). Mention `SKIP_ADDRESS_VALIDATION=true` to run without Google.
- **Run** — `poetry run health-intake`; how to quit (`quit`/`exit`); where output JSON and logs land.
- **Appointment availability rules** — slots generated for the next two days at 9am/2pm per provider; only future slots selectable; unknown slot ids rejected; selection echoed in confirmation. (Required by spec.)
- **Privacy & PHI handling** — PII redacted from logs; `output/`, `logs/`, `.env` gitignored; minimal data sent to the LLM; what would change for real HIPAA (encryption at rest, access controls, BAAs, retention, audit logging).
- **Testing** — `poetry run pytest`; coverage target; all external services mocked, no live calls.
- **Project layout** — short tree mirroring the File Structure section above.

- [ ] **Step 3: Run lint and type checks; fix issues**

Run: `poetry run ruff check . && poetry run ruff format --check . && poetry run mypy src`
Expected: no errors. Fix any reported issues, then re-run.

- [ ] **Step 4: Run the full suite a final time**

Run: `poetry run pytest`
Expected: PASS, coverage ≥ 80%.

- [ ] **Step 5: Commit**

```bash
git add README.md .env.example
git commit -m "docs: add README and env example"
```

- [ ] **Step 6: Create the GitHub repo and push**

If `gh` is authenticated:
```bash
gh repo create health-chatbot --private --source=. --remote=origin --push
```
Otherwise, create the repo in the GitHub UI and:
```bash
git remote add origin git@github.com:<you>/health-chatbot.git
git push -u origin main
```

---

## Self-Review (completed during plan authoring)

**Spec coverage:** every spec section maps to a task — config (T2), logging/privacy (T3), models (T4–5), validators (T6), Google address API (T7), mock appointments + documented availability (T8, T15), JSON persistence (T9), OpenAI integration + structured outputs (T10), prompts/forgiving UX (T11), validation gates "right info before advancing" (T12–13), confirmation summary (T14), README/privacy/HIPAA notes + GitHub (T15), testing 80%+ throughout.

**Placeholder scan:** no TBD/TODO; every code step contains complete code; README content is enumerated rather than left vague.

**Type consistency:** `RecordDraft`, `FieldExtraction`, `FieldResult`, `AddressResult`, `Slot`, `ConversationState`, `TurnResult`, and `LLMClient` signatures are used identically across tasks. `apply_extraction`, `is_step_satisfied`, `get_slot`, `available_slots`, `format_slots`, `build_record`, `build_situation`, and `write_record` signatures match their definitions and call sites.
