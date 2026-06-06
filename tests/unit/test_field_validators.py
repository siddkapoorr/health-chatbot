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
