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
