# Health Intake Chatbot

A terminal-based LLM chatbot that conducts patient intake conversations the way a human front-desk agent would: it accepts messy, natural input, responds dynamically, and guarantees each step has valid information before moving on.

Built with Python, Poetry, and OpenAI. Designed with real-patient privacy concerns in mind (PII-redacted logging, atomic file writes, restricted permissions) even though it only processes synthetic data.

## Architecture

```
User ──▶ CLI loop (rich) ──▶ Orchestrator (state machine)
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
              LLM extract    Python validate   LLM generate
              (temp=0,        (all gates)       (temp=0.3,
               structured                        natural
               outputs)                          reply)
```

**Two-phase turn protocol per user message:**
1. **Extraction call** (LLM, temperature 0, Structured Outputs) — parse any fields from free text into a typed schema
2. **Validate & advance** (Python only) — run validators, call Google Address Validation, check slot availability, decide whether to advance
3. **Generation call** (LLM, temperature 0.3) — phrase a warm reply that accurately reflects what Python validated

Python owns all validation gates. The LLM never decides whether data is valid or whether to advance.

### Step flow
```
greeting → patient_info → insurance → chief_complaint → address → appointment → confirmation
```

State is fully immutable: `ConversationState` and `RecordDraft` are frozen Pydantic v2 models; all updates return new copies via `model_copy(update={...})`.

## Setup

### Prerequisites
- Python 3.11+
- [Poetry](https://python-poetry.org/docs/#installation)
- An OpenAI API key (with access to `gpt-4o-mini` or your chosen model)
- A Google Maps API key with the **Address Validation API** enabled (or set `SKIP_ADDRESS_VALIDATION=true` for offline development)

### Install

```bash
git clone <repo-url>
cd health-chatbot
poetry install
```

### Configure

```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes | — | OpenAI API key |
| `GOOGLE_MAPS_API_KEY` | Unless skipped | — | Google Maps Platform key |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | OpenAI model ID |
| `LOG_LEVEL` | No | `INFO` | Python logging level |
| `OUTPUT_DIR` | No | `./output` | Directory for session JSON files |
| `SKIP_ADDRESS_VALIDATION` | No | `false` | Set `true` to bypass the Google API |

The app validates required keys at startup and exits with a clear error message if any are missing.

## Run

```bash
poetry run health-intake
```

Type `quit` or `exit` at any prompt to end the session without saving.

## Appointment Availability Rules

Mock appointments are generated at runtime from two providers:

| Provider | Specialty |
|---|---|
| Dr. Alice Chen | Family Medicine |
| Dr. Ben Okafor | Internal Medicine |

**Rules (enforced in code):**
- Slots are generated for **1 and 2 days ahead** of the current time at **09:00 and 14:00 local time**
- Slots in the past are never shown and cannot be selected
- Slot IDs encode provider, date, and time: `chen-20260607-0900`
- A slot selected during a session is treated as booked for that session

The LLM maps natural choices ("the 2pm with Dr. Chen", "option 2") to a slot ID; Python validates the ID against the live available set before accepting it.

## Project Layout

```
health-chatbot/
├── pyproject.toml              # Poetry config, entry point, tool config
├── .env.example                # Required environment variables
├── src/health_intake/
│   ├── __main__.py             # Entry point: poetry run health-intake
│   ├── cli.py                  # Terminal I/O loop, confirmation summary (rich)
│   ├── config.py               # pydantic-settings: keys, model, paths; fail-fast validation
│   ├── logging_config.py       # Logging setup + PII redaction filter
│   ├── models/
│   │   ├── conversation.py     # ConversationState, Step enum (frozen Pydantic)
│   │   └── patient.py          # IntakeRecord, PatientInfo, Address, Appointment, etc.
│   ├── engine/
│   │   ├── orchestrator.py     # State machine: drives steps and validation gates
│   │   ├── steps.py            # Step ordering and satisfaction checks
│   │   └── extraction.py       # FieldExtraction schema + apply_extraction
│   ├── llm/
│   │   ├── client.py           # LLMClient Protocol, OpenAIClient, FakeLLMClient
│   │   └── prompts.py          # System prompt, per-step situation builder
│   ├── validation/
│   │   ├── fields.py           # Name, DOB, payer, insurance-id, complaint validators
│   │   └── address.py          # Google Address Validation client + skip fallback
│   ├── appointments/
│   │   └── provider.py         # Mock providers, slot generation, slot lookup
│   └── storage/
│       └── writer.py           # Atomic PHI-safe JSON write to ./output/
└── tests/
    ├── conftest.py             # FakeAddressValidator, make_settings
    ├── unit/                   # Validators, models, appointments, redaction, prompts
    └── integration/            # Full conversation through orchestrator with fakes
```

## Testing

```bash
# Run all tests with coverage
poetry run pytest

# Unit tests only
poetry run pytest tests/unit/

# Integration tests only
poetry run pytest tests/integration/

# HTML coverage report
poetry run pytest --cov-report=html && open htmlcov/index.html
```

Target: **80% minimum coverage**. All external services (OpenAI, Google Address Validation) are mocked — no real API calls in the test suite.

**Test strategy:**
- **Unit tests** — validators, models, appointment logic, PII redaction, prompt building, field extraction
- **Integration tests** — scripted full conversations through `Orchestrator` using `FakeLLMClient` + `FakeAddressValidator`; assert final `IntakeRecord`, written JSON, and that validation gates block invalid input (e.g. a future DOB cannot advance)

## Privacy & PHI Handling

This project processes only synthetic data. It is designed with real-patient concerns in mind:

| Concern | Implementation |
|---|---|
| PII never logged in cleartext | `RedactionFilter` scrubs emails, dates, long digit strings, SSNs, and phone numbers from every log record including exception tracebacks |
| PHI files restricted at creation | Output JSON opened with `O_CREAT\|O_EXCL\|O_NOFOLLOW` at mode `0o600`; directory enforced at `0o700` on every write |
| API keys never leak in repr | `pydantic.SecretStr`; callers must use `.get_secret_value()` |
| Secrets not committed | `.env`, `output/`, `logs/` are gitignored |
| Minimal data to LLM | Only current conversation history and field context; no bulk PHI |

**What would change for real HIPAA compliance:**
- Encryption at rest for all PHI (AES-256 or equivalent)
- Signed Business Associate Agreements (BAAs) with OpenAI and Google
- Access controls and audit logging on every PHI read/write
- A defined data retention and deletion policy
- TLS certificate pinning for all external API calls
- Regular penetration testing and security audits

## Linting & Type Checking

```bash
poetry run ruff check .
poetry run ruff format --check .
poetry run mypy src
```
