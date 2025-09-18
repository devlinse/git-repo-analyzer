"""Tests for git platform service base class."""

import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from git_analyzer.services.git_platform import (
    AuthenticationType,
    AuthenticationInfo,
    GitPlatformService,
    GitPlatformError,
    AuthenticationError,
    RateLimitError,
    RepositoryNotFoundError,
    InvalidUrlError,
    RepositoryTooLargeError,
    Repository,
    RepositoryVisibility,
    PlatformInfo
)


class MockGitPlatformService(GitPlatformService):
    """Mock implementation for testing the base class."""

    def __init__(self, token=None, auth_type=AuthenticationType.PERSONAL_ACCESS_TOKEN):
        super().__init__(token, auth_type)
        self._mock_authenticated = False
        self._mock_repositories = []
        self._mock_organizations = []

    @property
    def platform_info(self) -> PlatformInfo:
        return PlatformInfo(
            name="MockPlatform",
            base_url="https://mock.example.com",
            api_url="https://api.mock.example.com",
            supported_auth_types=[AuthenticationType.PERSONAL_ACCESS_TOKEN],
            rate_limit_per_hour=5000,
            max_repo_size_mb=1000
        )

    async def authenticate(self) -> AuthenticationInfo:
        """Mock authentication."""
        if not self.token:
            raise AuthenticationError("No token provided")

        if self.token == "invalid_token":
            return AuthenticationInfo(
                valid=False,
                error_message="Invalid token"
            )

        if self.token == "rate_limited":
            raise RateLimitError("Rate limit exceeded", datetime.now())

        self._mock_authenticated = True
        return AuthenticationInfo(
            valid=True,
            username="testuser",
            scopes=["repo", "read:org"],
            rate_limit_remaining=4999,
            rate_limit_reset=datetime.now()
        )

    async def list_repositories(self,
                               owner=None,
                               include_forks=False,
                               include_archived=False,
                               visibility=None) -> list[Repository]:
        """Mock repository listing."""
        if not self._mock_authenticated:
            raise AuthenticationError("Not authenticated")

        return self._mock_repositories

    async def get_repository(self, owner: str, repo_name: str) -> Repository:
        """Mock get repository."""
        if not self._mock_authenticated:
            raise AuthenticationError("Not authenticated")

        if owner == "notfound" or repo_name == "notfound":
            raise RepositoryNotFoundError(f"Repository {owner}/{repo_name} not found")

        return Repository(
            name=repo_name,
            full_name=f"{owner}/{repo_name}",
            url=f"https://mock.example.com/{owner}/{repo_name}",
            clone_url=f"https://mock.example.com/{owner}/{repo_name}.git",
            ssh_url=f"git@mock.example.com:{owner}/{repo_name}.git",
            default_branch="main",
            visibility=RepositoryVisibility.PUBLIC,
            languages=["Python", "JavaScript"],
            size_bytes=1024000,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            description="Test repository"
        )

    async def list_organizations(self) -> list[str]:
        """Mock organization listing."""
        if not self._mock_authenticated:
            raise AuthenticationError("Not authenticated")

        return self._mock_organizations

    async def clone_repository(self,
                              repository: Repository,
                              target_dir: Path,
                              shallow: bool = True,
                              branch: str = None) -> Path:
        """Mock repository cloning."""
        if repository.size_bytes > self.platform_info.max_repo_size_mb * 1024 * 1024:
            raise RepositoryTooLargeError(
                "Repository too large",
                repository.size_bytes // (1024 * 1024),
                self.platform_info.max_repo_size_mb
            )

        clone_path = target_dir / repository.name
        clone_path.mkdir(parents=True, exist_ok=True)
        return clone_path

    def parse_repository_url(self, url: str) -> tuple[str, str]:
        """Mock URL parsing."""
        if "mock.example.com" not in url:
            raise InvalidUrlError(f"Invalid URL for MockPlatform: {url}")

        # Simple parsing for mock URLs
        if url.endswith(".git"):
            url = url[:-4]

        parts = url.rstrip('/').split('/')
        if len(parts) < 2:
            raise InvalidUrlError(f"Invalid URL format: {url}")

        return parts[-2], parts[-1]


class TestGitPlatformService:
    """Test cases for GitPlatformService base class."""

    def test_initialization(self):
        """Test service initialization."""
        service = MockGitPlatformService()
        assert service.token is None
        assert service.auth_type == AuthenticationType.PERSONAL_ACCESS_TOKEN
        assert not service._authenticated

        service_with_token = MockGitPlatformService("test_token")
        assert service_with_token.token == "test_token"

    def test_platform_info(self):
        """Test platform info property."""
        service = MockGitPlatformService()
        info = service.platform_info

        assert info.name == "MockPlatform"
        assert info.base_url == "https://mock.example.com"
        assert info.rate_limit_per_hour == 5000
        assert AuthenticationType.PERSONAL_ACCESS_TOKEN in info.supported_auth_types

    @pytest.mark.asyncio
    async def test_authentication_success(self):
        """Test successful authentication."""
        service = MockGitPlatformService("valid_token")
        auth_info = await service.authenticate()

        assert auth_info.valid
        assert auth_info.username == "testuser"
        assert "repo" in auth_info.scopes
        assert auth_info.rate_limit_remaining == 4999

    @pytest.mark.asyncio
    async def test_authentication_failure(self):
        """Test authentication failure."""
        service = MockGitPlatformService("invalid_token")
        auth_info = await service.authenticate()

        assert not auth_info.valid
        assert auth_info.error_message == "Invalid token"

    @pytest.mark.asyncio
    async def test_authentication_no_token(self):
        """Test authentication with no token."""
        service = MockGitPlatformService()

        with pytest.raises(AuthenticationError):
            await service.authenticate()

    @pytest.mark.asyncio
    async def test_authentication_rate_limit(self):
        """Test authentication rate limiting."""
        service = MockGitPlatformService("rate_limited")

        with pytest.raises(RateLimitError) as exc_info:
            await service.authenticate()

        assert "Rate limit exceeded" in str(exc_info.value)
        assert exc_info.value.reset_time is not None

    @pytest.mark.asyncio
    async def test_is_authenticated(self):
        """Test authentication status checking."""
        service = MockGitPlatformService()
        assert not await service.is_authenticated()

        service.token = "valid_token"
        assert await service.is_authenticated()
        assert service._authenticated

        # Should not call authenticate again
        service._mock_authenticated = False
        assert await service.is_authenticated()  # Should return True from cache

    @pytest.mark.asyncio
    async def test_is_authenticated_failure(self):
        """Test authentication status checking with invalid token."""
        service = MockGitPlatformService("invalid_token")
        assert not await service.is_authenticated()

    @pytest.mark.asyncio
    async def test_list_repositories_unauthenticated(self):
        """Test listing repositories without authentication."""
        service = MockGitPlatformService()

        with pytest.raises(AuthenticationError):
            await service.list_repositories()

    @pytest.mark.asyncio
    async def test_list_repositories_authenticated(self):
        """Test listing repositories with authentication."""
        service = MockGitPlatformService("valid_token")
        service._mock_authenticated = True
        service._mock_repositories = [
            Repository(
                name="test-repo",
                full_name="testuser/test-repo",
                url="https://mock.example.com/testuser/test-repo",
                clone_url="https://mock.example.com/testuser/test-repo.git",
                ssh_url="git@mock.example.com:testuser/test-repo.git",
                default_branch="main",
                visibility=RepositoryVisibility.PUBLIC,
                languages=["Python"],
                size_bytes=1024,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
        ]

        repos = await service.list_repositories()
        assert len(repos) == 1
        assert repos[0].name == "test-repo"

    @pytest.mark.asyncio
    async def test_get_repository_success(self):
        """Test getting repository information."""
        service = MockGitPlatformService("valid_token")
        service._mock_authenticated = True

        repo = await service.get_repository("testuser", "test-repo")

        assert repo.name == "test-repo"
        assert repo.full_name == "testuser/test-repo"
        assert repo.default_branch == "main"
        assert "Python" in repo.languages

    @pytest.mark.asyncio
    async def test_get_repository_not_found(self):
        """Test getting non-existent repository."""
        service = MockGitPlatformService("valid_token")
        service._mock_authenticated = True

        with pytest.raises(RepositoryNotFoundError):
            await service.get_repository("notfound", "notfound")

    @pytest.mark.asyncio
    async def test_get_repository_from_url(self):
        """Test getting repository from URL."""
        service = MockGitPlatformService("valid_token")
        service._mock_authenticated = True

        repo = await service.get_repository_from_url("https://mock.example.com/testuser/test-repo")

        assert repo.name == "test-repo"
        assert repo.full_name == "testuser/test-repo"

    @pytest.mark.asyncio
    async def test_get_repository_from_invalid_url(self):
        """Test getting repository from invalid URL."""
        service = MockGitPlatformService("valid_token")
        service._mock_authenticated = True

        with pytest.raises(InvalidUrlError):
            await service.get_repository_from_url("https://invalid.example.com/user/repo")

    @pytest.mark.asyncio
    async def test_list_organizations(self):
        """Test listing organizations."""
        service = MockGitPlatformService("valid_token")
        service._mock_authenticated = True
        service._mock_organizations = ["org1", "org2"]

        orgs = await service.list_organizations()
        assert orgs == ["org1", "org2"]

    @pytest.mark.asyncio
    async def test_clone_repository_success(self, tmp_path):
        """Test successful repository cloning."""
        service = MockGitPlatformService("valid_token")
        service._mock_authenticated = True

        repo = Repository(
            name="test-repo",
            full_name="testuser/test-repo",
            url="https://mock.example.com/testuser/test-repo",
            clone_url="https://mock.example.com/testuser/test-repo.git",
            ssh_url="git@mock.example.com:testuser/test-repo.git",
            default_branch="main",
            visibility=RepositoryVisibility.PUBLIC,
            languages=["Python"],
            size_bytes=1024,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        clone_path = await service.clone_repository(repo, tmp_path)
        assert clone_path.exists()
        assert clone_path.name == "test-repo"

    @pytest.mark.asyncio
    async def test_clone_repository_too_large(self, tmp_path):
        """Test cloning repository that exceeds size limits."""
        service = MockGitPlatformService("valid_token")
        service._mock_authenticated = True

        repo = Repository(
            name="large-repo",
            full_name="testuser/large-repo",
            url="https://mock.example.com/testuser/large-repo",
            clone_url="https://mock.example.com/testuser/large-repo.git",
            ssh_url="git@mock.example.com:testuser/large-repo.git",
            default_branch="main",
            visibility=RepositoryVisibility.PUBLIC,
            languages=["Python"],
            size_bytes=2000 * 1024 * 1024,  # 2GB
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        with pytest.raises(RepositoryTooLargeError) as exc_info:
            await service.clone_repository(repo, tmp_path)

        assert exc_info.value.size_mb == 2000
        assert exc_info.value.max_size_mb == 1000

    def test_parse_repository_url_success(self):
        """Test successful URL parsing."""
        service = MockGitPlatformService()

        owner, repo = service.parse_repository_url("https://mock.example.com/testuser/test-repo")
        assert owner == "testuser"
        assert repo == "test-repo"

        owner, repo = service.parse_repository_url("https://mock.example.com/testuser/test-repo.git")
        assert owner == "testuser"
        assert repo == "test-repo"

    def test_parse_repository_url_invalid(self):
        """Test URL parsing with invalid URLs."""
        service = MockGitPlatformService()

        with pytest.raises(InvalidUrlError):
            service.parse_repository_url("https://github.com/testuser/test-repo")

        with pytest.raises(InvalidUrlError):
            service.parse_repository_url("https://mock.example.com/invalid")

    @pytest.mark.asyncio
    async def test_get_rate_limit_status(self):
        """Test getting rate limit status."""
        service = MockGitPlatformService("valid_token")

        # Before authentication
        status = await service.get_rate_limit_status()
        assert status == {}

        # After authentication
        await service.authenticate()
        status = await service.get_rate_limit_status()
        assert 'remaining' in status
        assert 'reset_time' in status

    def test_cleanup_clone(self, tmp_path):
        """Test cleanup of cloned repository."""
        service = MockGitPlatformService()

        # Create a test directory
        test_dir = tmp_path / "test-repo"
        test_dir.mkdir()
        assert test_dir.exists()

        # Cleanup should remove the directory
        service.cleanup_clone(test_dir)
        assert not test_dir.exists()

    def test_cleanup_clone_nonexistent(self, tmp_path):
        """Test cleanup of non-existent directory."""
        service = MockGitPlatformService()

        # Should not raise exception
        nonexistent_path = tmp_path / "nonexistent"
        service.cleanup_clone(nonexistent_path)  # Should complete without error

    def test_validate_token_format(self):
        """Test token format validation."""
        service = MockGitPlatformService()

        assert service._validate_token_format("valid-token-123")
        assert not service._validate_token_format("")
        assert not service._validate_token_format("   ")
        assert not service._validate_token_format(None)

    def test_log_operation(self):
        """Test operation logging."""
        service = MockGitPlatformService("secret-token")

        with patch.object(service.logger, 'debug') as mock_debug:
            service._log_operation("test_operation", repo_name="test-repo")

            mock_debug.assert_called_once()
            call_args = mock_debug.call_args

            # Check that sensitive info is masked
            context = call_args[1]['extra']
            assert context['platform'] == "MockPlatform"
            assert context['operation'] == "test_operation"
            assert context['repo_name'] == "test-repo"


class TestDataClasses:
    """Test data classes used by git platform services."""

    def test_repository_creation(self):
        """Test Repository dataclass creation."""
        repo = Repository(
            name="test-repo",
            full_name="user/test-repo",
            url="https://example.com/user/test-repo",
            clone_url="https://example.com/user/test-repo.git",
            ssh_url="git@example.com:user/test-repo.git",
            default_branch="main",
            visibility=RepositoryVisibility.PUBLIC,
            languages=["Python", "JavaScript"],
            size_bytes=1024,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            description="Test repository"
        )

        assert repo.name == "test-repo"
        assert repo.topics == []  # Should be initialized by __post_init__
        assert not repo.fork
        assert not repo.archived

    def test_authentication_info_creation(self):
        """Test AuthenticationInfo dataclass creation."""
        auth_info = AuthenticationInfo(
            valid=True,
            username="testuser",
            rate_limit_remaining=4999
        )

        assert auth_info.valid
        assert auth_info.username == "testuser"
        assert auth_info.scopes == []  # Should be initialized by __post_init__
        assert auth_info.error_message is None

    def test_platform_info_creation(self):
        """Test PlatformInfo dataclass creation."""
        platform_info = PlatformInfo(
            name="TestPlatform",
            base_url="https://test.example.com",
            api_url="https://api.test.example.com",
            supported_auth_types=[AuthenticationType.PERSONAL_ACCESS_TOKEN],
            rate_limit_per_hour=5000,
            max_repo_size_mb=1000
        )

        assert platform_info.name == "TestPlatform"
        assert AuthenticationType.PERSONAL_ACCESS_TOKEN in platform_info.supported_auth_types


class TestExceptions:
    """Test custom exceptions."""

    def test_git_platform_error(self):
        """Test base GitPlatformError."""
        error = GitPlatformError("Test error")
        assert str(error) == "Test error"

    def test_authentication_error(self):
        """Test AuthenticationError."""
        error = AuthenticationError("Auth failed")
        assert str(error) == "Auth failed"
        assert isinstance(error, GitPlatformError)

    def test_rate_limit_error(self):
        """Test RateLimitError with reset time."""
        reset_time = datetime.now()
        error = RateLimitError("Rate limit exceeded", reset_time)

        assert str(error) == "Rate limit exceeded"
        assert error.reset_time == reset_time
        assert isinstance(error, GitPlatformError)

    def test_repository_not_found_error(self):
        """Test RepositoryNotFoundError."""
        error = RepositoryNotFoundError("Repo not found")
        assert str(error) == "Repo not found"
        assert isinstance(error, GitPlatformError)

    def test_invalid_url_error(self):
        """Test InvalidUrlError."""
        error = InvalidUrlError("Invalid URL")
        assert str(error) == "Invalid URL"
        assert isinstance(error, GitPlatformError)

    def test_repository_too_large_error(self):
        """Test RepositoryTooLargeError."""
        error = RepositoryTooLargeError("Too large", 2000, 1000)

        assert str(error) == "Too large"
        assert error.size_mb == 2000
        assert error.max_size_mb == 1000
        assert isinstance(error, GitPlatformError)


if __name__ == "__main__":
    pytest.main([__file__])