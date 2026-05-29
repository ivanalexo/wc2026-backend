from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Aplicación
    app_env: str  = "development"
    app_debug: bool = True
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    # Base de datos
    database_url: str

    # CORS — string separado por comas para evitar problemas de parsing
    # .env: CORS_ORIGINS=http://localhost:3000,https://mi-dominio.com
    cors_origins: str = "http://localhost:3000"

    # ML
    artifacts_dir: Path = Path("artifacts")

    # Squads (opcional)
    football_data_api_key: str | None = None

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()