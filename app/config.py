from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
    )

    app_env: str = 'development'
    app_debug: bool = True
    app_host: str = '0.0.0.0'
    app_port: int = 8000

    database_url: str

    cors_origins: list[str] = ['http://localhost:3000']

    @field_validator('cors_origins', mode='before')
    @classmethod
    def parse_cors_origins(cls, v: str | list) -> list[str]:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v
    
    artifacts_dir: Path = Path('artifacts')

    @property
    def is_production(self) -> bool:
        return self.app_env == 'production'

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()