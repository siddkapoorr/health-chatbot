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
