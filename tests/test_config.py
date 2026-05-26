import logging

import pytest


def test_defaults_without_yaml(clean_config_env):
    settings = clean_config_env.Settings()
    assert settings.models.llm_provider == "ollama"
    assert settings.models.llm_assistant == "llama3.2"
    assert settings.models.llm_digestion == "llama3.2"
    assert settings.models.llm_summary == "llama3.2"
    assert settings.models.llm_api_key is None


def test_transcription_defaults_cpu_int8(clean_config_env):
    settings = clean_config_env.Settings()
    assert settings.transcription.device == "cpu"
    assert settings.transcription.compute_type == "int8"
    assert settings.transcription.model_size == "small"


def test_diarization_max_speakers_default(clean_config_env):
    settings = clean_config_env.Settings()
    assert settings.diarization.max_speakers == 16


def test_env_override(clean_config_env, monkeypatch):
    monkeypatch.setenv("BAILIFF_MODELS__LLM_PROVIDER", "openai")
    monkeypatch.setenv("BAILIFF_MODELS__LLM_API_KEY", "sk-test")
    settings = clean_config_env.Settings()
    assert settings.models.llm_provider == "openai"
    assert settings.models.llm_api_key is not None
    assert settings.models.llm_api_key.get_secret_value() == "sk-test"


def test_validator_warns_when_non_ollama_without_key(clean_config_env, monkeypatch, caplog):
    monkeypatch.setenv("BAILIFF_MODELS__LLM_PROVIDER", "openai")
    with caplog.at_level(logging.WARNING, logger="bailiff.core.config"):
        settings = clean_config_env.Settings()
    assert settings.models.llm_provider == "openai"
    assert settings.models.llm_api_key is None
    assert any("llm_api_key" in r.message for r in caplog.records)


def test_validator_silent_for_ollama_without_key(clean_config_env, caplog):
    with caplog.at_level(logging.WARNING, logger="bailiff.core.config"):
        settings = clean_config_env.Settings()
    assert settings.models.llm_provider == "ollama"
    assert not any("llm_api_key" in r.message for r in caplog.records)
