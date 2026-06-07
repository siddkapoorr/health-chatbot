import pytest
from health_intake.config import Settings


def test_settings_load_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "g-test")

    settings = Settings()

    assert settings.openai_api_key.get_secret_value() == "sk-test"
    assert settings.openai_model == "gpt-4o-mini"
    assert settings.skip_address_validation is False


def test_missing_google_key_requires_skip_flag(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("GOOGLE_MAPS_API_KEY", raising=False)

    with pytest.raises(ValueError, match="GOOGLE_MAPS_API_KEY"):
        Settings()
