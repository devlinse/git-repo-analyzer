"""Configuration management for Git Repository Analyzer."""

import os
import toml
from typing import Optional, Dict, Any, List
from pathlib import Path
from dataclasses import dataclass, field
from pydantic import SecretStr, validator, ValidationError
from pydantic_settings import BaseSettings


@dataclass
class SecurityConfig:
    """Security-related configuration."""
    mask_secrets_in_logs: bool = True
    allowed_domains: List[str] = field(default_factory=lambda: [
        "github.com", "api.github.com",
        "dev.azure.com", "visualstudio.com",
        "bitbucket.org", "api.bitbucket.org"
    ])
    max_token_length: int = 1000
    require_https: bool = True


@dataclass
class AnalysisConfig:
    """Analysis-related configuration."""
    max_repo_size_mb: int = 500
    clone_timeout_seconds: int = 300
    analysis_timeout_seconds: int = 180
    max_files_to_analyze: int = 1000
    supported_languages: List[str] = field(default_factory=lambda: [
        "python", "javascript", "typescript", "java", "c#", "go", "rust", "php", "ruby"
    ])
    temp_dir: Path = field(default_factory=lambda: Path("temp_repos"))


@dataclass
class UIConfig:
    """UI-related configuration."""
    app_title: str = "Git Repository Analyzer"
    app_icon: str = "🔍"
    theme_color: str = "#FF6B35"
    page_title: str = "Git Repository Analyzer"
    layout: str = "wide"


class Settings(BaseSettings):
    """Comprehensive application settings with environment variable support and secrets integration."""

    # API Keys - these will be loaded from secrets or environment
    github_token: Optional[SecretStr] = None
    github_username: Optional[str] = None

    # OpenAI Configuration
    openai_api_key: Optional[SecretStr] = None
    openai_model: str = "gpt-4"
    openai_max_tokens: int = 4000
    openai_temperature: float = 0.1

    # Google AI Configuration
    google_ai_api_key: Optional[SecretStr] = None
    google_ai_model: str = "gemini-pro"

    # Azure OpenAI Configuration
    azure_openai_api_key: Optional[SecretStr] = None
    azure_openai_endpoint: Optional[str] = None
    azure_openai_model: str = "gpt-4"
    azure_openai_api_version: str = "2023-12-01-preview"

    # BitBucket Configuration
    bitbucket_username: Optional[str] = None
    bitbucket_app_password: Optional[SecretStr] = None

    # Azure DevOps Configuration
    azure_devops_organization: Optional[str] = None
    azure_devops_pat: Optional[SecretStr] = None

    # Configuration objects
    security: SecurityConfig = SecurityConfig()
    analysis: AnalysisConfig = AnalysisConfig()
    ui: UIConfig = UIConfig()

    # Environment settings
    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"

    @validator('github_token', 'openai_api_key', 'google_ai_api_key', 'azure_openai_api_key',
              'bitbucket_app_password', 'azure_devops_pat', pre=True)
    def validate_secret_length(cls, v):
        """Validate that secrets are not too long (security measure)."""
        if v and len(str(v)) > 1000:
            raise ValueError('Secret token is too long')
        return v

    @validator('azure_openai_endpoint')
    def validate_azure_endpoint(cls, v):
        """Validate Azure OpenAI endpoint format."""
        if v and not v.startswith('https://'):
            raise ValueError('Azure OpenAI endpoint must use HTTPS')
        return v

    @validator('openai_temperature')
    def validate_temperature(cls, v):
        """Validate OpenAI temperature is in valid range."""
        if not 0 <= v <= 2:
            raise ValueError('Temperature must be between 0 and 2')
        return v

    @validator('openai_max_tokens')
    def validate_max_tokens(cls, v):
        """Validate max tokens is reasonable."""
        if not 1 <= v <= 128000:
            raise ValueError('Max tokens must be between 1 and 128000')
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        validate_assignment = True

        @classmethod
        def customise_sources(
            cls,
            init_settings,
            env_settings,
            file_secret_settings,
        ):
            return (
                init_settings,
                StreamlitSecretsSource(),
                env_settings,
                file_secret_settings,
            )


class StreamlitSecretsSource:
    """Custom source for loading Streamlit secrets."""

    def __init__(self):
        self.secrets_path = Path(".streamlit/secrets.toml")

    def __call__(self, settings: BaseSettings) -> Dict[str, Any]:
        """Load secrets from Streamlit secrets.toml file."""
        if not self.secrets_path.exists():
            return {}

        try:
            with open(self.secrets_path, 'r', encoding='utf-8') as f:
                secrets = toml.load(f)

            # Flatten nested secrets if needed
            flattened = {}
            for key, value in secrets.items():
                if isinstance(value, dict):
                    # Handle nested sections like [ai] or [platforms]
                    for nested_key, nested_value in value.items():
                        flattened[f"{key}_{nested_key}"] = nested_value
                else:
                    flattened[key] = value

            return flattened

        except Exception as e:
            # Don't fail if secrets file is malformed, just log and continue
            print(f"Warning: Could not load secrets from {self.secrets_path}: {e}")
            return {}


def validate_configuration(settings: Settings) -> List[str]:
    """Validate the configuration and return list of validation errors."""
    errors = []

    # Check if at least one AI service is configured
    ai_services = [
        settings.openai_api_key,
        settings.google_ai_api_key,
        settings.azure_openai_api_key
    ]

    if not any(ai_services):
        errors.append("At least one AI service API key must be configured")

    # Validate Azure OpenAI configuration
    if settings.azure_openai_api_key and not settings.azure_openai_endpoint:
        errors.append("Azure OpenAI endpoint is required when Azure OpenAI API key is provided")

    # Validate file paths
    if not settings.analysis.temp_dir.parent.exists():
        errors.append(f"Parent directory for temp_dir does not exist: {settings.analysis.temp_dir.parent}")

    # Validate numeric ranges
    if settings.analysis.max_repo_size_mb <= 0:
        errors.append("max_repo_size_mb must be positive")

    if settings.analysis.clone_timeout_seconds <= 0:
        errors.append("clone_timeout_seconds must be positive")

    if settings.analysis.analysis_timeout_seconds <= 0:
        errors.append("analysis_timeout_seconds must be positive")

    return errors


def get_masked_config(settings: Settings) -> Dict[str, Any]:
    """Get configuration with sensitive values masked for logging/display."""
    config_dict = settings.dict()

    # Mask sensitive fields
    sensitive_fields = [
        'github_token', 'openai_api_key', 'google_ai_api_key',
        'azure_openai_api_key', 'bitbucket_app_password', 'azure_devops_pat'
    ]

    for field in sensitive_fields:
        if field in config_dict and config_dict[field] is not None:
            config_dict[field] = "***MASKED***"

    return config_dict


def load_settings() -> Settings:
    """Load and validate settings with proper error handling."""
    try:
        settings = Settings()

        # Validate configuration
        validation_errors = validate_configuration(settings)
        if validation_errors and settings.environment == "production":
            raise ValueError(f"Configuration validation failed: {', '.join(validation_errors)}")

        return settings

    except ValidationError as e:
        print(f"Configuration validation error: {e}")
        # In development, continue with default settings
        # In production, this should probably raise
        return Settings()
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return Settings()


# Global settings instance
settings = load_settings()


# Convenience functions for common operations
def get_github_config() -> tuple[Optional[str], Optional[str]]:
    """Get GitHub configuration (token, username)."""
    token = settings.github_token.get_secret_value() if settings.github_token else None
    return token, settings.github_username


def get_openai_config() -> tuple[Optional[str], str, int, float]:
    """Get OpenAI configuration (api_key, model, max_tokens, temperature)."""
    api_key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
    return api_key, settings.openai_model, settings.openai_max_tokens, settings.openai_temperature


def get_azure_openai_config() -> tuple[Optional[str], Optional[str], str, str]:
    """Get Azure OpenAI configuration (api_key, endpoint, model, api_version)."""
    api_key = settings.azure_openai_api_key.get_secret_value() if settings.azure_openai_api_key else None
    return api_key, settings.azure_openai_endpoint, settings.azure_openai_model, settings.azure_openai_api_version


def get_google_ai_config() -> tuple[Optional[str], str]:
    """Get Google AI configuration (api_key, model)."""
    api_key = settings.google_ai_api_key.get_secret_value() if settings.google_ai_api_key else None
    return api_key, settings.google_ai_model


def is_ai_service_configured(service: str) -> bool:
    """Check if a specific AI service is configured."""
    service_map = {
        'openai': settings.openai_api_key,
        'google': settings.google_ai_api_key,
        'azure': settings.azure_openai_api_key and settings.azure_openai_endpoint
    }
    return bool(service_map.get(service.lower()))


def get_available_ai_services() -> List[str]:
    """Get list of configured AI services."""
    services = []
    if is_ai_service_configured('openai'):
        services.append('OpenAI')
    if is_ai_service_configured('google'):
        services.append('Google AI')
    if is_ai_service_configured('azure'):
        services.append('Azure OpenAI')
    return services