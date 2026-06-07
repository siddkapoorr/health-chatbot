"""Application configuration loaded from environment / .env."""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated runtime configuration. Fails fast on missing required keys."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: SecretStr
    openai_model: str = "gpt-4o-mini"
    google_maps_api_key: SecretStr | None = None
    skip_address_validation: bool = False
    log_level: str = "INFO"
    output_dir: Path = Path("./output")

    @model_validator(mode="after")
    def _require_google_key_unless_skipped(self) -> "Settings":
        if not self.skip_address_validation and not self.google_maps_api_key:
            raise ValueError("GOOGLE_MAPS_API_KEY is required unless SKIP_ADDRESS_VALIDATION=true")
        return self


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    # pydantic-settings injects env vars at __init__ time; mypy sees no positional args
    return Settings()  # type: ignore[call-arg]
