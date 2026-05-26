import logging
from typing import Optional

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    YamlConfigSettingsSource,
)

logger = logging.getLogger("bailiff.core.config")


class AppConfig(BaseSettings):
    log_file: str = "bailiff.log"
    log_level: str = "INFO"
    data_dir: str = "data"


class AudioConfig(BaseSettings):
    sample_rate: int = 16000
    chunk_size: int = 512
    vad_threshold: float = 0.5
    silence_limit: float = 1.0


class ModelsConfig(BaseSettings):
    llm_provider: str = "ollama"
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: Optional[SecretStr] = None
    llm_assistant: str = "llama3.2"
    llm_digestion: str = "llama3.2"
    llm_summary: str = "llama3.2"
    voice_embedding: str = "speechbrain/spkrec-ecapa-voxceleb"

    @model_validator(mode="after")
    def _warn_missing_api_key(self):
        if self.llm_provider != "ollama" and self.llm_api_key is None:
            logger.warning(
                "llm_provider=%s but llm_api_key is not set; calls will fail at runtime",
                self.llm_provider,
            )
        return self


class DiarizationConfig(BaseSettings):
    threshold: float = 0.5
    inertia_weight: float = 0.1
    merge_timeout: float = 8.0
    segment_timeout: float = 3.0
    max_speakers: int = Field(default=16)


class TranscriptionConfig(BaseSettings):
    model_size: str = "small"
    device: str = "cpu"
    compute_type: str = "int8"
    language: Optional[str] = None


class Settings(BaseSettings):
    app: AppConfig = Field(default_factory=AppConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    diarization: DiarizationConfig = Field(default_factory=DiarizationConfig)
    transcription: TranscriptionConfig = Field(default_factory=TranscriptionConfig)

    class Config:
        env_prefix = "BAILIFF_"
        env_nested_delimiter = "__"
        env_file = ".env"
        env_file_encoding = "utf-8"
        yaml_file = "config.yaml"
        extra = "ignore"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )


_settings_instance: Optional[Settings] = None


def load_settings() -> Settings:
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance


settings = load_settings()
