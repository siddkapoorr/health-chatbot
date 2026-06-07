import pytest
from health_intake.config import Settings


def _isolated_settings(**kwargs) -> Settings:
    """Construct Settings without reading the local .env file."""
    return Settings(_env_file=None, **kwargs)  # type: ignore[call-arg]


def test_settings_load_from_env():
    settings = _isolated_settings(
        openai_api_key="sk-test",
        google_maps_api_key="g-test",
    )

    assert settings.openai_api_key.get_secret_value() == "sk-test"
    assert settings.openai_model == "gpt-4o-mini"
    assert settings.skip_address_validation is False


def test_missing_google_key_requires_skip_flag():
    with pytest.raises(ValueError, match="GOOGLE_MAPS_API_KEY"):
        _isolated_settings(openai_api_key="sk-test")
