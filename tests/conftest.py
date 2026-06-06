from pathlib import Path

from health_intake.config import Settings
from health_intake.validation.address import AddressResult


class FakeAddressValidator:
    def __init__(self, ok: bool = True) -> None:
        self._ok = ok

    def validate(self, street: str, city: str, state: str, zip_code: str) -> AddressResult:
        if self._ok:
            return AddressResult(ok=True, formatted=f"{street}, {city}, {state} {zip_code}, USA")
        return AddressResult(ok=False, missing=("postal_code",), error="Unconfirmed address.")


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        openai_api_key="sk-test",
        skip_address_validation=True,
        output_dir=tmp_path,
    )
