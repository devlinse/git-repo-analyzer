"""Tests for configuration management."""

import os
import tempfile
import toml
from pathlib import Path
from unittest.mock import patch, mock_open
import pytest

from git_analyzer.config import (
    Settings,
    SecurityConfig,
    AnalysisConfig,
    UIConfig,
    StreamlitSecretsSource,
    validate_configuration,
    get_masked_config,
    load_settings,
    get_github_config,
    get_openai_config,
    is_ai_service_configured,
    get_available_ai_services
)


class TestSecurityConfig:
    """Test SecurityConfig dataclass."""

    def test_default_values(self):
        """Test default security configuration values."""
        config = SecurityConfig()

        assert config.mask_secrets_in_logs is True
        assert config.max_token_length == 1000
        assert config.require_https is True
        assert "github.com" in config.allowed_domains
        assert "dev.azure.com" in config.allowed_domains


class TestAnalysisConfig:
    """Test AnalysisConfig dataclass."""

    def test_default_values(self):
        """Test default analysis configuration values."""
        config = AnalysisConfig()

        assert config.max_repo_size_mb == 500
        assert config.clone_timeout_seconds == 300
        assert config.analysis_timeout_seconds == 180
        assert config.max_files_to_analyze == 1000
        assert "python" in config.supported_languages
        assert config.temp_dir == Path("temp_repos")


class TestUIConfig:
    """Test UIConfig dataclass."""

    def test_default_values(self):
        """Test default UI configuration values."""
        config = UIConfig()

        assert config.app_title == "Git Repository Analyzer"
        assert config.app_icon == "🔍"
        assert config.theme_color == "#FF6B35"
        assert config.layout == "wide"


class TestSettings:
    """Test Settings pydantic model."""

    def test_default_settings(self):
        """Test default settings initialization."""
        settings = Settings()

        # Check API keys are None by default
        assert settings.github_token is None
        assert settings.openai_api_key is None
        assert settings.google_ai_api_key is None

        # Check default values
        assert settings.openai_model == "gpt-4"
        assert settings.openai_max_tokens == 4000
        assert settings.openai_temperature == 0.1
        assert settings.environment == "development"
        assert settings.debug is False

    def test_environment_variable_loading(self):
        """Test loading configuration from environment variables."""
        with patch.dict(os.environ, {
            'GITHUB_TOKEN': 'test-token',
            'OPENAI_API_KEY': 'sk-test123',
            'OPENAI_MODEL': 'gpt-3.5-turbo',
            'DEBUG': 'true'
        }):
            settings = Settings()

            assert settings.github_token.get_secret_value() == 'test-token'
            assert settings.openai_api_key.get_secret_value() == 'sk-test123'
            assert settings.openai_model == 'gpt-3.5-turbo'
            assert settings.debug is True

    def test_secret_validation(self):
        """Test validation of secret fields."""
        # Test valid secret
        settings = Settings(github_token="valid-token")
        assert settings.github_token.get_secret_value() == "valid-token"

        # Test secret too long
        with pytest.raises(Exception):
            Settings(github_token="x" * 1001)

    def test_azure_endpoint_validation(self):
        """Test Azure OpenAI endpoint validation."""
        # Valid HTTPS endpoint
        settings = Settings(azure_openai_endpoint="https://test.openai.azure.com/")
        assert settings.azure_openai_endpoint == "https://test.openai.azure.com/"

        # Invalid HTTP endpoint
        with pytest.raises(Exception):
            Settings(azure_openai_endpoint="http://test.openai.azure.com/")

    def test_temperature_validation(self):
        """Test OpenAI temperature validation."""
        # Valid temperature
        settings = Settings(openai_temperature=0.5)
        assert settings.openai_temperature == 0.5

        # Invalid temperature (too high)
        with pytest.raises(Exception):
            Settings(openai_temperature=3.0)

        # Invalid temperature (negative)
        with pytest.raises(Exception):
            Settings(openai_temperature=-0.1)

    def test_max_tokens_validation(self):
        """Test max tokens validation."""
        # Valid max tokens
        settings = Settings(openai_max_tokens=2000)
        assert settings.openai_max_tokens == 2000

        # Invalid max tokens (too high)
        with pytest.raises(Exception):
            Settings(openai_max_tokens=200000)

        # Invalid max tokens (zero)
        with pytest.raises(Exception):
            Settings(openai_max_tokens=0)


class TestStreamlitSecretsSource:
    """Test StreamlitSecretsSource functionality."""

    def test_nonexistent_secrets_file(self):
        """Test behavior when secrets file doesn't exist."""
        source = StreamlitSecretsSource()
        source.secrets_path = Path("nonexistent/secrets.toml")

        result = source(Settings())
        assert result == {}

    def test_valid_secrets_file(self):
        """Test loading valid secrets file."""
        secrets_content = """
github_token = "test-github-token"
openai_api_key = "test-openai-key"

[azure]
openai_api_key = "test-azure-key"
openai_endpoint = "https://test.openai.azure.com/"
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(secrets_content)
            temp_path = f.name

        try:
            source = StreamlitSecretsSource()
            source.secrets_path = Path(temp_path)

            result = source(Settings())

            assert result['github_token'] == "test-github-token"
            assert result['openai_api_key'] == "test-openai-key"
            assert result['azure_openai_api_key'] == "test-azure-key"
            assert result['azure_openai_endpoint'] == "https://test.openai.azure.com/"

        finally:
            os.unlink(temp_path)

    def test_malformed_secrets_file(self):
        """Test handling of malformed secrets file."""
        malformed_content = """
invalid toml content [[[
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(malformed_content)
            temp_path = f.name

        try:
            source = StreamlitSecretsSource()
            source.secrets_path = Path(temp_path)

            # Should not raise exception, just return empty dict
            result = source(Settings())
            assert result == {}

        finally:
            os.unlink(temp_path)


class TestConfigurationValidation:
    """Test configuration validation functions."""

    def test_validate_configuration_no_ai_services(self):
        """Test validation when no AI services are configured."""
        settings = Settings()
        errors = validate_configuration(settings)

        assert "At least one AI service API key must be configured" in errors

    def test_validate_configuration_azure_missing_endpoint(self):
        """Test validation when Azure key is provided but endpoint is missing."""
        settings = Settings(azure_openai_api_key="test-key")
        errors = validate_configuration(settings)

        assert "Azure OpenAI endpoint is required when Azure OpenAI API key is provided" in errors

    def test_validate_configuration_negative_values(self):
        """Test validation of negative configuration values."""
        settings = Settings(
            openai_api_key="test-key"  # Provide at least one AI service
        )
        settings.analysis.max_repo_size_mb = -1
        settings.analysis.clone_timeout_seconds = -1
        settings.analysis.analysis_timeout_seconds = -1

        errors = validate_configuration(settings)

        assert "max_repo_size_mb must be positive" in errors
        assert "clone_timeout_seconds must be positive" in errors
        assert "analysis_timeout_seconds must be positive" in errors

    def test_validate_configuration_valid(self):
        """Test validation with valid configuration."""
        settings = Settings(openai_api_key="test-key")
        errors = validate_configuration(settings)

        # Should have no errors (except possibly temp_dir parent not existing)
        critical_errors = [e for e in errors if "temp_dir" not in e]
        assert len(critical_errors) == 0

    def test_get_masked_config(self):
        """Test masking of sensitive configuration values."""
        settings = Settings(
            github_token="secret-token",
            openai_api_key="secret-key",
            openai_model="gpt-4"
        )

        masked = get_masked_config(settings)

        assert masked['github_token'] == "***MASKED***"
        assert masked['openai_api_key'] == "***MASKED***"
        assert masked['openai_model'] == "gpt-4"  # Not sensitive, should remain


class TestConvenienceFunctions:
    """Test convenience functions for accessing configuration."""

    def test_get_github_config(self):
        """Test GitHub configuration getter."""
        settings = Settings(
            github_token="test-token",
            github_username="test-user"
        )

        # Mock the global settings
        with patch('git_analyzer.config.settings', settings):
            token, username = get_github_config()

            assert token == "test-token"
            assert username == "test-user"

    def test_get_openai_config(self):
        """Test OpenAI configuration getter."""
        settings = Settings(
            openai_api_key="test-key",
            openai_model="gpt-3.5-turbo",
            openai_max_tokens=2000,
            openai_temperature=0.7
        )

        with patch('git_analyzer.config.settings', settings):
            api_key, model, max_tokens, temperature = get_openai_config()

            assert api_key == "test-key"
            assert model == "gpt-3.5-turbo"
            assert max_tokens == 2000
            assert temperature == 0.7

    def test_is_ai_service_configured(self):
        """Test AI service configuration checker."""
        settings = Settings(
            openai_api_key="test-key",
            azure_openai_api_key="azure-key",
            azure_openai_endpoint="https://test.openai.azure.com/"
        )

        with patch('git_analyzer.config.settings', settings):
            assert is_ai_service_configured('openai') is True
            assert is_ai_service_configured('azure') is True
            assert is_ai_service_configured('google') is False

    def test_get_available_ai_services(self):
        """Test getting list of available AI services."""
        settings = Settings(
            openai_api_key="test-key",
            google_ai_api_key="google-key"
        )

        with patch('git_analyzer.config.settings', settings):
            services = get_available_ai_services()

            assert "OpenAI" in services
            assert "Google AI" in services
            assert "Azure OpenAI" not in services


class TestLoadSettings:
    """Test settings loading with error handling."""

    def test_load_settings_success(self):
        """Test successful settings loading."""
        with patch('git_analyzer.config.validate_configuration', return_value=[]):
            settings = load_settings()
            assert isinstance(settings, Settings)

    def test_load_settings_validation_errors_development(self):
        """Test handling validation errors in development environment."""
        with patch('git_analyzer.config.validate_configuration', return_value=["error"]):
            settings = load_settings()
            # Should still return settings in development
            assert isinstance(settings, Settings)

    @patch('git_analyzer.config.Settings')
    def test_load_settings_exception_handling(self, mock_settings):
        """Test exception handling during settings loading."""
        mock_settings.side_effect = Exception("Test error")

        settings = load_settings()
        # Should return default settings on exception
        assert isinstance(settings, Settings)


@pytest.fixture
def temp_secrets_file():
    """Create a temporary secrets file for testing."""
    secrets_content = """
github_token = "test-github-token"
openai_api_key = "sk-test123"

[azure]
openai_api_key = "azure-test-key"
openai_endpoint = "https://test.openai.azure.com/"
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(secrets_content)
        temp_path = f.name

    yield temp_path

    os.unlink(temp_path)


def test_integration_secrets_loading(temp_secrets_file):
    """Integration test for loading secrets from file."""
    # Mock the secrets path to use our temp file
    with patch.object(StreamlitSecretsSource, 'secrets_path', Path(temp_secrets_file)):
        settings = Settings()

        # The secrets should be loaded through the custom source
        assert settings.github_token.get_secret_value() == "test-github-token"
        assert settings.openai_api_key.get_secret_value() == "sk-test123"


if __name__ == "__main__":
    pytest.main([__file__])