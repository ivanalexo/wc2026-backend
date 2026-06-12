from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    app_env: str  = "development"
    app_debug: bool = True
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    database_url: str

    cors_origins: str = "http://localhost:3000"

    # ML
    artifacts_dir: Path = Path("artifacts")

    # API Football (https://v3.football.api-sports.io)
    api_football_key: str | None = None

    # football-data.org v4  (https://www.football-data.org)
    football_data_org_key: str | None = None

    # Clave compartida para el endpoint de sync (Railway cron → /api/v1/admin/sync)
    sync_secret: str | None = None

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