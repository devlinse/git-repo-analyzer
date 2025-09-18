"""Configuration management for Git Repository Analyzer."""

from typing import Optional
from pydantic import SecretStr
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # API Keys
    github_token: Optional[SecretStr] = None
    openai_api_key: Optional[SecretStr] = None
    google_ai_api_key: Optional[SecretStr] = None
    azure_openai_api_key: Optional[SecretStr] = None
    azure_openai_endpoint: Optional[str] = None

    # Application settings
    max_repo_size_mb: int = 500
    clone_timeout_seconds: int = 300
    analysis_timeout_seconds: int = 180
    temp_dir: Path = Path("temp_repos")

    # UI settings
    app_title: str = "Git Repository Analyzer"
    app_icon: str = "🔍"

    class Config:
        env_file = ".env"
        secrets_dir = ".streamlit"


# Global settings instance
settings = Settings()