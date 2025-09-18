"""Tests for GitHub platform service implementation."""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch, call
from urllib.parse import urlparse
import subprocess

import pytest
import requests
from requests.exceptions import RequestException, Timeout

from git_analyzer.services.github import GitHubService
from git_analyzer.services.git_platform import (
    AuthenticationType,
    AuthenticationError,
    AuthenticationInfo,
    GitPlatformError,
    InvalidUrlError,
    PlatformInfo,
    RateLimitError,
    Repository,
    RepositoryNotFoundError,
    RepositoryTooLargeError,
    RepositoryVisibility,
)


class TestGitHubService:
    """Test cases for GitHubService."""

    def test_initialization_default(self):
        """Test default GitHub service initialization."""
        service = GitHubService()

        assert service.token is None
        assert service.base_url == "https://github.com"
        assert service.api_url == "https://api.github.com"
        assert service.auth_type == AuthenticationType.PERSONAL_ACCESS_TOKEN
        assert not service._authenticated

    def test_initialization_with_token(self):
        """Test GitHub service initialization with token."""
        token = "ghp_test_token_12345"
        service = GitHubService(token=token)

        assert service.token == token
        assert "Authorization" in service.session.headers
        assert service.session.headers["Authorization"] == f"token {token}"

    def test_initialization_enterprise(self):
        """Test GitHub Enterprise initialization."""
        base_url = "https://github.enterprise.com"
        service = GitHubService(base_url=base_url)

        assert service.base_url == base_url
        assert service.api_url == f"{base_url}/api/v3"

    def test_initialization_custom_api_url(self):
        """Test initialization with custom API URL."""
        base_url = "https://github.enterprise.com"
        api_url = "https://api.github.enterprise.com"
        service = GitHubService(base_url=base_url, api_url=api_url)

        assert service.base_url == base_url
        assert service.api_url == api_url

    def test_platform_info_github_com(self):
        """Test platform info for GitHub.com."""
        service = GitHubService()
        info = service.platform_info

        assert info.name == "GitHub"
        assert info.base_url == "https://github.com"
        assert info.api_url == "https://api.github.com"
        assert info.rate_limit_per_hour == 60  # Unauthenticated
        assert info.max_repo_size_mb == 500
        assert AuthenticationType.PERSONAL_ACCESS_TOKEN in info.supported_auth_types

    def test_platform_info_github_enterprise(self):
        """Test platform info for GitHub Enterprise."""
        service = GitHubService(base_url="https://github.enterprise.com")
        info = service.platform_info

        assert info.name == "GitHub Enterprise"
        assert info.base_url == "https://github.enterprise.com"

    def test_platform_info_authenticated(self):
        """Test platform info with authentication token."""
        service = GitHubService(token="ghp_test_token")
        info = service.platform_info

        assert info.rate_limit_per_hour == 5000  # Authenticated

    def test_validate_token_format(self):
        """Test token format validation."""
        service = GitHubService()

        # Valid tokens
        assert service._validate_token_format("ghp_1234567890abcdef1234567890abcdef12345678")
        assert service._validate_token_format("ghs_1234567890abcdef1234567890abcdef12345678")
        assert service._validate_token_format("github_pat_11ABCDEFG0123456789012345678901234567890123456789012345678901234567890")
        assert service._validate_token_format("v1.1f699f1069f60xxx")  # OAuth token

        # Invalid tokens
        assert not service._validate_token_format("")
        assert not service._validate_token_format("   ")
        assert not service._validate_token_format("short")
        assert not service._validate_token_format(None)

    @pytest.mark.asyncio
    async def test_authenticate_success(self):
        """Test successful authentication."""
        service = GitHubService(token="ghp_valid_token")

        # Mock responses
        user_response = Mock()
        user_response.json.return_value = {"login": "testuser", "id": 12345}
        user_response.headers = {"X-OAuth-Scopes": "repo, read:org"}

        rate_limit_response = Mock()
        rate_limit_response.json.return_value = {
            "resources": {
                "core": {
                    "remaining": 4999,
                    "reset": 1640995200
                }
            }
        }

        with patch.object(service, '_make_request') as mock_request:
            mock_request.side_effect = [user_response, rate_limit_response]

            auth_info = await service.authenticate()

            assert auth_info.valid
            assert auth_info.username == "testuser"
            assert auth_info.scopes == ["repo", "read:org"]
            assert auth_info.rate_limit_remaining == 4999
            assert isinstance(auth_info.rate_limit_reset, datetime)
            assert service._authenticated

    @pytest.mark.asyncio
    async def test_authenticate_no_token(self):
        """Test authentication without token."""
        service = GitHubService()

        auth_info = await service.authenticate()

        assert not auth_info.valid
        assert auth_info.error_message == "No authentication token provided"

    @pytest.mark.asyncio
    async def test_authenticate_invalid_token_format(self):
        """Test authentication with invalid token format."""
        service = GitHubService(token="invalid")

        auth_info = await service.authenticate()

        assert not auth_info.valid
        assert auth_info.error_message == "Invalid token format"

    @pytest.mark.asyncio
    async def test_authenticate_401_error(self):
        """Test authentication with 401 error."""
        service = GitHubService(token="ghp_invalid_token")

        mock_response = Mock()
        mock_response.status_code = 401

        error = RequestException("Unauthorized")
        error.response = mock_response

        with patch.object(service, '_make_request') as mock_request:
            mock_request.side_effect = error

            with pytest.raises(AuthenticationError) as exc_info:
                await service.authenticate()

            assert "Invalid or expired authentication token" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_authenticate_403_error(self):
        """Test authentication with 403 error."""
        service = GitHubService(token="ghp_limited_token")

        mock_response = Mock()
        mock_response.status_code = 403

        error = RequestException("Forbidden")
        error.response = mock_response

        with patch.object(service, '_make_request') as mock_request:
            mock_request.side_effect = error

            with pytest.raises(AuthenticationError) as exc_info:
                await service.authenticate()

            assert "Token does not have sufficient permissions" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_list_repositories_success(self):
        """Test successful repository listing."""
        service = GitHubService(token="ghp_valid_token")
        service._authenticated = True

        # Mock repository data
        repos_data = [
            {
                "name": "repo1",
                "full_name": "testuser/repo1",
                "html_url": "https://github.com/testuser/repo1",
                "clone_url": "https://github.com/testuser/repo1.git",
                "ssh_url": "git@github.com:testuser/repo1.git",
                "default_branch": "main",
                "private": False,
                "size": 1024,
                "created_at": "2023-01-01T00:00:00Z",
                "updated_at": "2023-01-02T00:00:00Z",
                "description": "Test repository 1",
                "topics": ["python", "test"],
                "fork": False,
                "archived": False
            }
        ]

        mock_response = Mock()
        mock_response.json.return_value = repos_data

        with patch.object(service, 'is_authenticated', return_value=True):
            with patch.object(service, '_make_request', return_value=mock_response):
                repositories = await service.list_repositories()

        assert len(repositories) == 1
        repo = repositories[0]
        assert repo.name == "repo1"
        assert repo.full_name == "testuser/repo1"
        assert repo.visibility == RepositoryVisibility.PUBLIC
        assert not repo.fork
        assert not repo.archived

    @pytest.mark.asyncio
    async def test_list_repositories_with_filters(self):
        """Test repository listing with filters."""
        service = GitHubService(token="ghp_valid_token")

        repos_data = [
            {
                "name": "repo1",
                "full_name": "testuser/repo1",
                "html_url": "https://github.com/testuser/repo1",
                "clone_url": "https://github.com/testuser/repo1.git",
                "ssh_url": "git@github.com:testuser/repo1.git",
                "default_branch": "main",
                "private": False,
                "size": 1024,
                "created_at": "2023-01-01T00:00:00Z",
                "updated_at": "2023-01-02T00:00:00Z",
                "fork": True,
                "archived": False
            },
            {
                "name": "repo2",
                "full_name": "testuser/repo2",
                "html_url": "https://github.com/testuser/repo2",
                "clone_url": "https://github.com/testuser/repo2.git",
                "ssh_url": "git@github.com:testuser/repo2.git",
                "default_branch": "main",
                "private": False,
                "size": 1024,
                "created_at": "2023-01-01T00:00:00Z",
                "updated_at": "2023-01-02T00:00:00Z",
                "fork": False,
                "archived": True
            }
        ]

        mock_response = Mock()
        mock_response.json.return_value = repos_data

        with patch.object(service, 'is_authenticated', return_value=True):
            with patch.object(service, '_make_request', return_value=mock_response):
                # Test excluding forks and archived
                repositories = await service.list_repositories(
                    include_forks=False,
                    include_archived=False
                )

        assert len(repositories) == 0  # Both repos should be filtered out

    @pytest.mark.asyncio
    async def test_list_repositories_pagination(self):
        """Test repository listing with pagination."""
        service = GitHubService(token="ghp_valid_token")

        # First page
        page1_data = [{"name": f"repo{i}", "full_name": f"user/repo{i}",
                       "html_url": f"https://github.com/user/repo{i}",
                       "clone_url": f"https://github.com/user/repo{i}.git",
                       "ssh_url": f"git@github.com:user/repo{i}.git",
                       "default_branch": "main", "private": False, "size": 1024,
                       "created_at": "2023-01-01T00:00:00Z",
                       "updated_at": "2023-01-02T00:00:00Z",
                       "fork": False, "archived": False} for i in range(100)]

        # Second page (partial)
        page2_data = [{"name": "repo100", "full_name": "user/repo100",
                       "html_url": "https://github.com/user/repo100",
                       "clone_url": "https://github.com/user/repo100.git",
                       "ssh_url": "git@github.com:user/repo100.git",
                       "default_branch": "main", "private": False, "size": 1024,
                       "created_at": "2023-01-01T00:00:00Z",
                       "updated_at": "2023-01-02T00:00:00Z",
                       "fork": False, "archived": False}]

        responses = [Mock(), Mock()]
        responses[0].json.return_value = page1_data
        responses[1].json.return_value = page2_data

        with patch.object(service, 'is_authenticated', return_value=True):
            with patch.object(service, '_make_request', side_effect=responses):
                repositories = await service.list_repositories()

        assert len(repositories) == 101

    @pytest.mark.asyncio
    async def test_list_repositories_unauthenticated(self):
        """Test repository listing without authentication."""
        service = GitHubService()

        with patch.object(service, 'is_authenticated', return_value=False):
            with pytest.raises(AuthenticationError):
                await service.list_repositories()

    @pytest.mark.asyncio
    async def test_get_repository_success(self):
        """Test successful get repository."""
        service = GitHubService(token="ghp_valid_token")

        repo_data = {
            "name": "test-repo",
            "full_name": "testuser/test-repo",
            "html_url": "https://github.com/testuser/test-repo",
            "clone_url": "https://github.com/testuser/test-repo.git",
            "ssh_url": "git@github.com:testuser/test-repo.git",
            "default_branch": "main",
            "private": False,
            "size": 1024,
            "created_at": "2023-01-01T00:00:00Z",
            "updated_at": "2023-01-02T00:00:00Z",
            "description": "Test repository",
            "topics": ["python"],
            "fork": False,
            "archived": False
        }

        languages_data = {"Python": 50000, "JavaScript": 25000}

        repo_response = Mock()
        repo_response.json.return_value = repo_data

        languages_response = Mock()
        languages_response.json.return_value = languages_data

        with patch.object(service, '_make_request', side_effect=[repo_response, languages_response]):
            repository = await service.get_repository("testuser", "test-repo")

        assert repository.name == "test-repo"
        assert repository.full_name == "testuser/test-repo"
        assert "Python" in repository.languages
        assert "JavaScript" in repository.languages
        assert repository.description == "Test repository"

    @pytest.mark.asyncio
    async def test_get_repository_not_found(self):
        """Test get repository with 404 error."""
        service = GitHubService(token="ghp_valid_token")

        mock_response = Mock()
        mock_response.status_code = 404

        error = RequestException("Not Found")
        error.response = mock_response

        with patch.object(service, '_make_request', side_effect=error):
            with pytest.raises(RepositoryNotFoundError) as exc_info:
                await service.get_repository("testuser", "nonexistent")

            assert "Repository testuser/nonexistent not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_list_organizations_success(self):
        """Test successful organization listing."""
        service = GitHubService(token="ghp_valid_token")

        orgs_data = [
            {"login": "org1", "id": 1},
            {"login": "org2", "id": 2}
        ]

        mock_response = Mock()
        mock_response.json.return_value = orgs_data

        with patch.object(service, 'is_authenticated', return_value=True):
            with patch.object(service, '_make_request', return_value=mock_response):
                organizations = await service.list_organizations()

        assert organizations == ["org1", "org2"]

    @pytest.mark.asyncio
    async def test_clone_repository_success(self, tmp_path):
        """Test successful repository cloning."""
        service = GitHubService(token="ghp_valid_token")

        repository = Repository(
            name="test-repo",
            full_name="testuser/test-repo",
            url="https://github.com/testuser/test-repo",
            clone_url="https://github.com/testuser/test-repo.git",
            ssh_url="git@github.com:testuser/test-repo.git",
            default_branch="main",
            visibility=RepositoryVisibility.PUBLIC,
            languages=["Python"],
            size_bytes=50 * 1024 * 1024,  # 50MB
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Cloning into 'test-repo'..."
        mock_result.stderr = ""

        with patch('subprocess.run', return_value=mock_result) as mock_run:
            clone_path = await service.clone_repository(repository, tmp_path)

        assert clone_path == tmp_path / "test-repo"
        mock_run.assert_called_once()

        # Verify git command
        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "git"
        assert call_args[1] == "clone"
        assert "--depth" in call_args
        assert "1" in call_args
        assert "--branch" in call_args
        assert "main" in call_args

    @pytest.mark.asyncio
    async def test_clone_repository_too_large(self, tmp_path):
        """Test cloning repository that exceeds size limit."""
        service = GitHubService(token="ghp_valid_token")

        repository = Repository(
            name="large-repo",
            full_name="testuser/large-repo",
            url="https://github.com/testuser/large-repo",
            clone_url="https://github.com/testuser/large-repo.git",
            ssh_url="git@github.com:testuser/large-repo.git",
            default_branch="main",
            visibility=RepositoryVisibility.PUBLIC,
            languages=["Python"],
            size_bytes=600 * 1024 * 1024,  # 600MB (exceeds 500MB limit)
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        with pytest.raises(RepositoryTooLargeError) as exc_info:
            await service.clone_repository(repository, tmp_path)

        assert exc_info.value.size_mb == 600.0
        assert exc_info.value.max_size_mb == 500

    @pytest.mark.asyncio
    async def test_clone_repository_private_with_token(self, tmp_path):
        """Test cloning private repository with authentication."""
        service = GitHubService(token="ghp_valid_token")

        repository = Repository(
            name="private-repo",
            full_name="testuser/private-repo",
            url="https://github.com/testuser/private-repo",
            clone_url="https://github.com/testuser/private-repo.git",
            ssh_url="git@github.com:testuser/private-repo.git",
            default_branch="main",
            visibility=RepositoryVisibility.PRIVATE,
            languages=["Python"],
            size_bytes=50 * 1024 * 1024,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        mock_result = Mock()
        mock_result.returncode = 0

        with patch('subprocess.run', return_value=mock_result) as mock_run:
            await service.clone_repository(repository, tmp_path)

        # Verify authenticated URL is used
        call_args = mock_run.call_args[0][0]
        clone_url = call_args[-2]  # Second to last argument should be the URL
        assert "ghp_valid_token@github.com" in clone_url

    @pytest.mark.asyncio
    async def test_clone_repository_git_error(self, tmp_path):
        """Test cloning repository with git command error."""
        service = GitHubService(token="ghp_valid_token")

        repository = Repository(
            name="error-repo",
            full_name="testuser/error-repo",
            url="https://github.com/testuser/error-repo",
            clone_url="https://github.com/testuser/error-repo.git",
            ssh_url="git@github.com:testuser/error-repo.git",
            default_branch="main",
            visibility=RepositoryVisibility.PUBLIC,
            languages=["Python"],
            size_bytes=50 * 1024 * 1024,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        error = subprocess.CalledProcessError(1, ["git", "clone"], stderr="fatal: repository not found")

        with patch('subprocess.run', side_effect=error):
            with pytest.raises(GitPlatformError) as exc_info:
                await service.clone_repository(repository, tmp_path)

            assert "Failed to clone repository" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_clone_repository_timeout(self, tmp_path):
        """Test cloning repository with timeout."""
        service = GitHubService(token="ghp_valid_token")

        repository = Repository(
            name="slow-repo",
            full_name="testuser/slow-repo",
            url="https://github.com/testuser/slow-repo",
            clone_url="https://github.com/testuser/slow-repo.git",
            ssh_url="git@github.com:testuser/slow-repo.git",
            default_branch="main",
            visibility=RepositoryVisibility.PUBLIC,
            languages=["Python"],
            size_bytes=50 * 1024 * 1024,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        with patch('subprocess.run', side_effect=subprocess.TimeoutExpired(["git", "clone"], 300)):
            with pytest.raises(GitPlatformError) as exc_info:
                await service.clone_repository(repository, tmp_path)

            assert "Repository clone timed out" in str(exc_info.value)

    def test_parse_repository_url_https(self):
        """Test parsing HTTPS repository URLs."""
        service = GitHubService()

        # Standard GitHub URLs
        owner, repo = service.parse_repository_url("https://github.com/user/repo")
        assert owner == "user" and repo == "repo"

        owner, repo = service.parse_repository_url("https://github.com/user/repo.git")
        assert owner == "user" and repo == "repo"

        owner, repo = service.parse_repository_url("https://github.com/user/repo/")
        assert owner == "user" and repo == "repo"

    def test_parse_repository_url_ssh(self):
        """Test parsing SSH repository URLs."""
        service = GitHubService()

        owner, repo = service.parse_repository_url("git@github.com:user/repo.git")
        assert owner == "user" and repo == "repo"

        owner, repo = service.parse_repository_url("git@github.com:user/repo")
        assert owner == "user" and repo == "repo"

    def test_parse_repository_url_enterprise(self):
        """Test parsing GitHub Enterprise URLs."""
        service = GitHubService()

        # Enterprise HTTPS
        owner, repo = service.parse_repository_url("https://github.enterprise.com/user/repo")
        assert owner == "user" and repo == "repo"

        # Enterprise SSH
        owner, repo = service.parse_repository_url("git@github.enterprise.com:user/repo.git")
        assert owner == "user" and repo == "repo"

    def test_parse_repository_url_invalid(self):
        """Test parsing invalid repository URLs."""
        service = GitHubService()

        with pytest.raises(InvalidUrlError):
            service.parse_repository_url("invalid-url")

        with pytest.raises(InvalidUrlError):
            service.parse_repository_url("https://notgithub.com/user/repo")

        with pytest.raises(InvalidUrlError):
            service.parse_repository_url("https://github.com/user")

    @pytest.mark.asyncio
    async def test_make_request_success(self):
        """Test successful HTTP request."""
        service = GitHubService(token="ghp_valid_token")

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"test": "data"}
        mock_response.headers = {}

        with patch.object(service.session, 'request', return_value=mock_response) as mock_request:
            response = await service._make_request("GET", "/test")

        assert response == mock_response
        mock_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_make_request_rate_limit(self):
        """Test HTTP request with rate limit error."""
        service = GitHubService(token="ghp_valid_token")

        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.headers = {"X-RateLimit-Reset": "1640995200"}

        with patch.object(service.session, 'request', return_value=mock_response):
            with pytest.raises(RateLimitError) as exc_info:
                await service._make_request("GET", "/test")

            assert "GitHub API rate limit exceeded" in str(exc_info.value)
            assert exc_info.value.reset_time is not None

    @pytest.mark.asyncio
    async def test_make_request_timeout(self):
        """Test HTTP request timeout."""
        service = GitHubService(token="ghp_valid_token")

        with patch.object(service.session, 'request', side_effect=Timeout("Request timeout")):
            with pytest.raises(GitPlatformError) as exc_info:
                await service._make_request("GET", "/test")

            assert "timed out" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_make_request_401_error(self):
        """Test HTTP request with 401 error."""
        service = GitHubService(token="ghp_invalid_token")

        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = RequestException("Unauthorized")

        error = RequestException("Unauthorized")
        error.response = mock_response

        with patch.object(service.session, 'request', side_effect=error):
            with pytest.raises(AuthenticationError) as exc_info:
                await service._make_request("GET", "/test")

            assert "Invalid or expired token" in str(exc_info.value)

    def test_parse_repository_data(self):
        """Test parsing repository data from GitHub API response."""
        service = GitHubService()

        repo_data = {
            "name": "test-repo",
            "full_name": "user/test-repo",
            "html_url": "https://github.com/user/test-repo",
            "clone_url": "https://github.com/user/test-repo.git",
            "ssh_url": "git@github.com:user/test-repo.git",
            "default_branch": "main",
            "private": True,
            "size": 1024,  # KB
            "created_at": "2023-01-01T00:00:00Z",
            "updated_at": "2023-01-02T00:00:00Z",
            "description": "Test repository",
            "topics": ["python", "test"],
            "fork": False,
            "archived": False
        }

        languages_data = {"Python": 50000, "JavaScript": 25000}

        repo = service._parse_repository(repo_data, languages_data)

        assert repo.name == "test-repo"
        assert repo.full_name == "user/test-repo"
        assert repo.visibility == RepositoryVisibility.PRIVATE
        assert repo.size_bytes == 1024 * 1024  # Converted from KB to bytes
        assert repo.languages == ["Python", "JavaScript"]
        assert repo.topics == ["python", "test"]
        assert not repo.fork
        assert not repo.archived

    def test_extract_scopes_from_headers(self):
        """Test extracting OAuth scopes from response headers."""
        service = GitHubService()

        # With scopes
        headers = {"X-OAuth-Scopes": "repo, read:org, write:packages"}
        scopes = service._extract_scopes_from_headers(headers)
        assert scopes == ["repo", "read:org", "write:packages"]

        # Without scopes header
        headers = {}
        scopes = service._extract_scopes_from_headers(headers)
        assert scopes == []

        # Empty scopes
        headers = {"X-OAuth-Scopes": ""}
        scopes = service._extract_scopes_from_headers(headers)
        assert scopes == []


@pytest.mark.integration
class TestGitHubServiceIntegration:
    """Integration tests for GitHubService requiring real API access."""

    @pytest.mark.skipif(
        not pytest.importorskip("os").getenv("GITHUB_TOKEN"),
        reason="No GitHub token available for integration testing"
    )
    @pytest.mark.asyncio
    async def test_real_authentication(self):
        """Test real GitHub authentication."""
        import os
        token = os.getenv("GITHUB_TOKEN")
        service = GitHubService(token=token)

        auth_info = await service.authenticate()

        assert auth_info.valid
        assert auth_info.username is not None
        assert isinstance(auth_info.scopes, list)

    @pytest.mark.skipif(
        not pytest.importorskip("os").getenv("GITHUB_TOKEN"),
        reason="No GitHub token available for integration testing"
    )
    @pytest.mark.asyncio
    async def test_real_repository_listing(self):
        """Test real repository listing."""
        import os
        token = os.getenv("GITHUB_TOKEN")
        service = GitHubService(token=token)

        repositories = await service.list_repositories()

        assert isinstance(repositories, list)
        if repositories:
            repo = repositories[0]
            assert isinstance(repo, Repository)
            assert repo.name
            assert repo.full_name
            assert repo.url


if __name__ == "__main__":
    pytest.main([__file__])