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
