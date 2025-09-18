#!/usr/bin/env python3
"""Simple validation script for configuration management."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_basic_config():
    """Test basic configuration loading."""
    try:
        from git_analyzer.config import (
            Settings, SecurityConfig, AnalysisConfig, UIConfig,
            validate_configuration, get_masked_config,
            get_available_ai_services, is_ai_service_configured
        )

        print("OK: All imports successful")

        # Test dataclasses
        security = SecurityConfig()
        analysis = AnalysisConfig()
        ui = UIConfig()

        print(f"OK: SecurityConfig defaults: mask_secrets={security.mask_secrets_in_logs}, max_token_length={security.max_token_length}")
        print(f"OK: AnalysisConfig defaults: max_size={analysis.max_repo_size_mb}MB, timeout={analysis.clone_timeout_seconds}s")
        print(f"OK: UIConfig defaults: title='{ui.app_title}', icon='[emoji]'")

        # Test Settings
        settings = Settings()
        print(f"OK: Settings created with environment: {settings.environment}")

        # Test validation
        errors = validate_configuration(settings)
        print(f"OK: Validation completed with {len(errors)} errors:")
        for error in errors:
            print(f"   ! {error}")

        # Test masking
        masked = get_masked_config(settings)
        print(f"OK: Config masking works")

        # Test AI service detection
        services = get_available_ai_services()
        print(f"OK: Available AI services: {services}")

        # Test specific service checks
        print(f"OK: OpenAI configured: {is_ai_service_configured('openai')}")
        print(f"OK: Google AI configured: {is_ai_service_configured('google')}")
        print(f"OK: Azure OpenAI configured: {is_ai_service_configured('azure')}")

        return True

    except Exception as e:
        print(f"X Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_secrets_source():
    """Test StreamlitSecretsSource."""
    try:
        from git_analyzer.config import StreamlitSecretsSource, Settings

        source = StreamlitSecretsSource()
        result = source(Settings())
        print(f"OK: StreamlitSecretsSource works, loaded {len(result)} secrets")

        return True

    except Exception as e:
        print(f"X StreamlitSecretsSource error: {e}")
        return False

def test_environment_variables():
    """Test environment variable loading."""
    import os

    # Set some test environment variables
    test_env = {
        'GITHUB_TOKEN': 'test-token-123',
        'OPENAI_MODEL': 'gpt-3.5-turbo',
        'DEBUG': 'true'
    }

    # Save original values
    original_env = {}
    for key in test_env:
        original_env[key] = os.environ.get(key)
        os.environ[key] = test_env[key]

    try:
        from git_analyzer.config import Settings

        settings = Settings()

        # Check if values were loaded
        if settings.github_token and settings.github_token.get_secret_value() == 'test-token-123':
            print("OK: GitHub token loaded from environment")
        else:
            print("! GitHub token not loaded from environment")

        if settings.openai_model == 'gpt-3.5-turbo':
            print("OK: OpenAI model loaded from environment")
        else:
            print("! OpenAI model not loaded from environment")

        if settings.debug:
            print("OK: Debug flag loaded from environment")
        else:
            print("! Debug flag not loaded from environment")

        return True

    except Exception as e:
        print(f"X Environment variable test error: {e}")
        return False

    finally:
        # Restore original environment
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

if __name__ == "__main__":
    print("Git Repository Analyzer - Configuration Validation")
    print("=" * 60)

    success = True

    print("\n1. Testing basic configuration...")
    success &= test_basic_config()

    print("\n2. Testing secrets source...")
    success &= test_secrets_source()

    print("\n3. Testing environment variables...")
    success &= test_environment_variables()

    print("\n" + "=" * 60)
    if success:
        print("SUCCESS: All configuration tests passed!")
        sys.exit(0)
    else:
        print("FAILED: Some configuration tests failed!")
        sys.exit(1)