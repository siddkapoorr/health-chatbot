# Health Intake Chatbot — Design Spec

**Date:** 2026-06-06
**Status:** Approved (pending written-spec review)

## 1. Purpose & Context

A terminal-based, LLM-powered chatbot that conducts a patient intake conversation the
way a human front-desk agent would: it accepts messy, natural input, responds
dynamically, and guarantees each step has valid information before moving on.

This is a **take-home project**. It will only ever use **synthetic data**, but it is
designed with **real-patient concerns in mind** — privacy/PII handling, input
validation, graceful error handling, and clear documentation — so an evaluator can see
production-minded engineering.

### Non-goals
- No real PHI, no real EHR/EMR integration, no real booking system.
- No multi-provider LLM abstraction (OpenAI only; a thin client wrapper is kept for
  testability, not multi-vendor support).
- No database — structured output is written to JSON files.

## 2. Requirements

### Information to collect
| Group | Field | Required | Validation |
|---|---|---|---|
| Patient | full_name | yes | non-empty; letters / spaces / hyphens / apostrophes |
| Patient | date_of_birth | yes | valid date, in the past, plausible age (0–120) |
| Insurance | payer_name | yes | non-empty |
| Insurance | insurance_id | no | if provided, basic alphanumeric format |
| Medical | chief_complaint | yes | non-empty free text, sane length |
| Demographics | address (street, city, state, zip) | yes | required components present → validated via **Google Address Validation API**; notify user of invalid/missing parts |
| Appointment | selected slot (provider + datetime) | yes | must match an available mock slot |

### Behavioral requirements
- **Robust & forgiving UX:** user may give info out of order, supply multiple fields in
  one message, correct earlier answers, and ask clarifying questions.
- **Deterministic validation gates:** the system must confirm each step's required
  fields are valid before advancing. Validity is decided by Python, never the LLM.
- **Appointment availability** rules are documented in the README and enforced in code.
- **Confirmation:** at the end, display a summary (appointment date/time, assigned
  physician, all collected info) and persist it.

## 3. Architecture

**Approach: Deterministic orchestrator + LLM as an extraction/generation layer.**

Python owns a state machine of ordered steps and all validation. The LLM does exactly
two jobs: (1) extract/update structured field values from free text, and (2) phrase the
next natural-language reply. The LLM never decides whether data is valid or whether to
advance.

### Step flow
```
greeting → patient_info → insurance → chief_complaint → address → appointment → confirmation
```

`ConversationState` holds a partial `IntakeRecord` (all fields optional during
collection), the current step, and message history.

### Two-phase turn protocol (both deterministic)
1. **Extraction call** (LLM, OpenAI Structured Outputs, temperature 0): given history +
   the fields currently being filled, return a strict-JSON `FieldExtraction` containing
   any confidently-parsed fields (supports multi-field answers and corrections to
   earlier fields) plus a `meta` flag indicating whether the user asked a question.
2. **Validate & decide** (Python): merge updates → run validators. On the address step,
   call the Google Address Validation API; on the appointment step, check the chosen
   slot is real and available. Compute which required fields are missing/invalid and
   whether to advance. **Python alone owns the gate.**
3. **Generation call** (LLM, temperature ~0.3): given the real situation (confirmed
   fields, exact validation error if any, what to ask next, or that we are advancing),
   produce a warm, natural reply. This guarantees the message always matches the actual
   validation outcome.

Two LLM calls per turn is a deliberate trade-off: slightly more cost/latency in exchange
for full determinism and reliability. Can be collapsed to one call later if desired.

## 4. Module structure

```
health-chatbot/
├── pyproject.toml              # Poetry; [tool.poetry.scripts] entry point
├── README.md
├── .env.example
├── .gitignore                  # .env, output/, logs/, __pycache__
├── src/health_intake/
│   ├── __main__.py             # entry: `poetry run health-intake`
│   ├── cli.py                  # terminal I/O loop (rich)
│   ├── config.py               # pydantic-settings: keys, model, paths; fail-fast on missing keys
│   ├── logging_config.py       # logging setup + PII redaction filter
│   ├── models/
│   │   ├── patient.py          # PatientInfo, Insurance, Address, Appointment, IntakeRecord
│   │   └── conversation.py     # ConversationState, Step enum, StepStatus
│   ├── engine/
│   │   ├── orchestrator.py     # state machine: drives steps, validation gates, advancement
│   │   ├── steps.py            # ordered step definitions + required fields per step
│   │   └── extraction.py       # build LLM request, parse structured output → field updates
│   ├── llm/
│   │   ├── client.py           # thin OpenAI wrapper (Protocol + impl), retry, timeout
│   │   └── prompts.py          # system prompt, per-step instructions, JSON schemas
│   ├── validation/
│   │   ├── fields.py           # name, DOB, payer, insurance-id, complaint validators
│   │   └── address.py          # Google Address Validation client + graceful fallback
│   ├── appointments/
│   │   └── provider.py         # mock providers/slots, availability rules, choice validation
│   └── storage/
│       └── writer.py           # write IntakeRecord → ./output/intake-<id>.json
└── tests/
    ├── unit/
    ├── integration/
    └── conftest.py             # FakeLLMClient, FakeAddressClient, fixtures
```

Each file has a single clear responsibility; none should approach the 800-line limit.
Adding a new field = model entry + validator + schema field.

## 5. Data model

Pydantic v2 models in `models/patient.py`:
- `Address(street, city, state, zip, validated: bool, formatted: str | None)`
- `Insurance(payer_name, insurance_id: str | None)`
- `PatientInfo(full_name, date_of_birth: date)`
- `Appointment(provider_name, specialty, start_time, slot_id)`
- `IntakeRecord(session_id, created_at, patient, insurance, chief_complaint, address, appointment)`

`models/conversation.py`:
- `Step` enum (ordered), `StepStatus`
- `ConversationState(current_step, record_draft, messages, ...)` — immutable updates
  (return new copies, never mutate in place).

## 6. External integrations

### Google Address Validation API
- `POST https://addressvalidation.googleapis.com/v1:validateAddress?key=...` via httpx.
- Inspect `result.verdict` (`addressComplete`, `hasUnconfirmedComponents`) and
  `result.address` missing/unconfirmed components.
- On incomplete/unconfirmed: report exactly which parts are problematic and re-ask. On
  success, store the API's `formattedAddress`.
- Requires `GOOGLE_MAPS_API_KEY` with the Address Validation API enabled (free tier
  suitable for dev). Graceful handling of missing key / quota / network errors. A
  documented `SKIP_ADDRESS_VALIDATION` env flag allows offline development.
- Mocked in tests via `respx`.

### Appointments (mock)
- `appointments/provider.py` defines providers (name, specialty) each with predefined
  datetime slots.
- **Availability rules (documented in README, enforced in code):** no past slots, no
  nonexistent slots; a slot selected during the session is treated as booked.
- Slots presented as a readable list. LLM maps a natural choice ("the 2pm with Dr. Lee",
  "option 2") to a slot id; Python validates against the real available set.

## 7. Error handling, logging & privacy

### Error handling
- Every external call (OpenAI, Google) wrapped with explicit handling; transient errors
  retried with backoff (tenacity).
- Invalid user input never crashes the app — it re-prompts.
- Top-level CLI guard catches unexpected exceptions → friendly message + logged stack
  trace; non-zero exit.

### Logging
- stdlib `logging`, configurable level, console + `./logs`, correlated by session id.
- **PII redaction filter** scrubs name, DOB, address, and insurance id from all log
  output. Events are logged, not raw PHI.

### Privacy (real-patient-minded, synthetic-data-only)
- PHI never logged in cleartext.
- `output/`, `logs/`, and `.env` are gitignored; `.env.example` documents required keys.
- Minimal data sent to the LLM.
- README section on data classification and what would change for real HIPAA compliance:
  encryption at rest, access controls, BAAs with processors, retention policy, audit
  logging.

## 8. Configuration

`config.py` uses `pydantic-settings` reading from `.env`:
- `OPENAI_API_KEY` (required)
- `OPENAI_MODEL` (default `gpt-4o-mini`)
- `GOOGLE_MAPS_API_KEY` (required unless `SKIP_ADDRESS_VALIDATION=true`)
- `LOG_LEVEL` (default `INFO`)
- `OUTPUT_DIR` (default `./output`)
- `SKIP_ADDRESS_VALIDATION` (default `false`)

Required keys are validated at startup; missing keys fail fast with a clear message.

## 9. Testing & tooling

- **pytest**, target 80%+ coverage (pytest-cov).
- **Unit:** validators, models, appointment logic, extraction parsing, redaction filter,
  prompt building.
- **Integration:** a scripted full conversation through the orchestrator using
  `FakeLLMClient` + `FakeAddressClient` → asserts the final `IntakeRecord`, the written
  JSON output, and that gates block advancement (e.g. an invalid DOB cannot proceed).
- **No real API calls in tests** — all external services mocked.
- **Tooling:** ruff (lint + format), mypy (types), pytest-cov; optional pre-commit.

### Dependencies (Poetry)
- Runtime: `openai`, `pydantic`, `pydantic-settings`, `httpx`, `tenacity`, `rich`.
- Dev: `pytest`, `pytest-cov`, `pytest-mock`, `respx`, `ruff`, `mypy`.
- Entry point: `[tool.poetry.scripts] health-intake = "health_intake.__main__:main"`.

## 10. Locked decisions
- LLM provider: **OpenAI**, default model `gpt-4o-mini`, configurable.
- Terminal UX: `rich`.
- Persistence: timestamped JSON file per session in `./output`.
- Version control: GitHub. Local `git init` + `.gitignore` + initial commit; remote
  created via `gh` if authenticated, otherwise commands provided.
