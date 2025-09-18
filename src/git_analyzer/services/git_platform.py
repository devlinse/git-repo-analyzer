"""Abstract base class for git platform integration services."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class AuthenticationType(Enum):
    """Authentication types supported by git platforms."""
    PERSONAL_ACCESS_TOKEN = "pat"
    APP_PASSWORD = "app_password"
    OAUTH_TOKEN = "oauth"
    SSH_KEY = "ssh_key"


class RepositoryVisibility(Enum):
    """Repository visibility levels."""
    PUBLIC = "public"
    PRIVATE = "private"
    INTERNAL = "internal"


@dataclass
class Repository:
    """Repository information from git platforms."""
    name: str
    full_name: str  # owner/repo
    url: str
    clone_url: str
    ssh_url: str
    default_branch: str
    visibility: RepositoryVisibility
    languages: List[str]
    size_bytes: int
    created_at: datetime
    updated_at: datetime
    description: Optional[str] = None
    topics: List[str] = None
    fork: bool = False
    archived: bool = False

    def __post_init__(self):
        """Initialize optional fields."""
        if self.topics is None:
            self.topics = []


@dataclass
class AuthenticationInfo:
    """Authentication information and validation result."""
    valid: bool
    username: Optional[str] = None
    scopes: List[str] = None
    rate_limit_remaining: Optional[int] = None
    rate_limit_reset: Optional[datetime] = None
    error_message: Optional[str] = None

    def __post_init__(self):
        """Initialize optional fields."""
        if self.scopes is None:
            self.scopes = []


@dataclass
class PlatformInfo:
    """Platform-specific information."""
    name: str
    base_url: str
    api_url: str
    supported_auth_types: List[AuthenticationType]
    rate_limit_per_hour: int
    max_repo_size_mb: int


class GitPlatformError(Exception):
    """Base exception for git platform operations."""
    pass


class AuthenticationError(GitPlatformError):
    """Authentication failed."""
    pass


class RateLimitError(GitPlatformError):
    """Rate limit exceeded."""

    def __init__(self, message: str, reset_time: Optional[datetime] = None):
        super().__init__(message)
        self.reset_time = reset_time


class RepositoryNotFoundError(GitPlatformError):
    """Repository not found or not accessible."""
    pass


class InvalidUrlError(GitPlatformError):
    """Invalid repository URL format."""
    pass


class RepositoryTooLargeError(GitPlatformError):
    """Repository exceeds size limits."""

    def __init__(self, message: str, size_mb: int, max_size_mb: int):
        super().__init__(message)
        self.size_mb = size_mb
        self.max_size_mb = max_size_mb


class GitPlatformService(ABC):
    """Abstract base class for git platform integration services."""

    def __init__(self, token: Optional[str] = None, auth_type: AuthenticationType = AuthenticationType.PERSONAL_ACCESS_TOKEN):
        """Initialize the git platform service.

        Args:
            token: Authentication token (PAT, app password, etc.)
            auth_type: Type of authentication being used
        """
        self.token = token
        self.auth_type = auth_type
        self._authenticated = False
        self._auth_info: Optional[AuthenticationInfo] = None

        # Set up logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @property
    @abstractmethod
    def platform_info(self) -> PlatformInfo:
        """Get platform-specific information."""
        pass

    @abstractmethod
    async def authenticate(self) -> AuthenticationInfo:
        """Validate authentication credentials.

        Returns:
            AuthenticationInfo with validation results and user info

        Raises:
            AuthenticationError: If authentication fails
        """
        pass

    @abstractmethod
    async def list_repositories(self,
                               owner: Optional[str] = None,
                               include_forks: bool = False,
                               include_archived: bool = False,
                               visibility: Optional[RepositoryVisibility] = None) -> List[Repository]:
        """List repositories for authenticated user or specified owner.

        Args:
            owner: Repository owner (username/organization). If None, lists current user's repos
            include_forks: Whether to include forked repositories
            include_archived: Whether to include archived repositories
            visibility: Filter by repository visibility

        Returns:
            List of Repository objects

        Raises:
            AuthenticationError: If not authenticated
            RateLimitError: If rate limit exceeded
            GitPlatformError: For other API errors
        """
        pass

    @abstractmethod
    async def get_repository(self, owner: str, repo_name: str) -> Repository:
        """Get detailed information about a specific repository.

        Args:
            owner: Repository owner
            repo_name: Repository name

        Returns:
            Repository object with detailed information

        Raises:
            RepositoryNotFoundError: If repository doesn't exist or not accessible
            AuthenticationError: If not authenticated and repo is private
            RateLimitError: If rate limit exceeded
        """
        pass

    @abstractmethod
    async def list_organizations(self) -> List[str]:
        """List organizations/teams the authenticated user has access to.

        Returns:
            List of organization names

        Raises:
            AuthenticationError: If not authenticated
            RateLimitError: If rate limit exceeded
        """
        pass

    @abstractmethod
    async def clone_repository(self,
                              repository: Repository,
                              target_dir: Path,
                              shallow: bool = True,
                              branch: Optional[str] = None) -> Path:
        """Clone a repository to local filesystem.

        Args:
            repository: Repository to clone
            target_dir: Directory to clone into
            shallow: Whether to perform shallow clone (--depth=1)
            branch: Specific branch to clone

        Returns:
            Path to cloned repository

        Raises:
            RepositoryTooLargeError: If repository exceeds size limits
            GitPlatformError: For git operation failures
        """
        pass

    @abstractmethod
    def parse_repository_url(self, url: str) -> tuple[str, str]:
        """Parse repository URL to extract owner and repository name.

        Args:
            url: Repository URL (HTTPS or SSH)

        Returns:
            Tuple of (owner, repository_name)

        Raises:
            InvalidUrlError: If URL format is invalid for this platform
        """
        pass

    async def is_authenticated(self) -> bool:
        """Check if service is properly authenticated.

        Returns:
            True if authenticated, False otherwise
        """
        if not self._authenticated and self.token:
            try:
                auth_info = await self.authenticate()
                self._authenticated = auth_info.valid
                self._auth_info = auth_info
            except Exception as e:
                self.logger.error(f"Authentication check failed: {e}")
                self._authenticated = False

        return self._authenticated

    async def get_repository_from_url(self, url: str) -> Repository:
        """Get repository information from URL.

        Args:
            url: Repository URL

        Returns:
            Repository object

        Raises:
            InvalidUrlError: If URL format is invalid
            RepositoryNotFoundError: If repository doesn't exist
        """
        try:
            owner, repo_name = self.parse_repository_url(url)
            return await self.get_repository(owner, repo_name)
        except Exception as e:
            self.logger.error(f"Failed to get repository from URL {url}: {e}")
            raise

    def _validate_token_format(self, token: str) -> bool:
        """Validate token format (to be overridden by platform implementations).

        Args:
            token: Authentication token

        Returns:
            True if format is valid
        """
        return token and len(token.strip()) > 0

    def _log_operation(self, operation: str, **kwargs):
        """Log git platform operation for debugging.

        Args:
            operation: Operation name
            **kwargs: Additional context
        """
        context = {
            'platform': self.platform_info.name,
            'operation': operation,
            'authenticated': self._authenticated,
            **kwargs
        }

        # Remove sensitive information
        if 'token' in context:
            context['token'] = '***MASKED***'

        self.logger.debug(f"Git platform operation", extra=context)

    async def get_rate_limit_status(self) -> Dict[str, Union[int, datetime]]:
        """Get current rate limit status.

        Returns:
            Dictionary with rate limit information
        """
        if self._auth_info:
            return {
                'remaining': self._auth_info.rate_limit_remaining,
                'reset_time': self._auth_info.rate_limit_reset
            }
        return {}

    def cleanup_clone(self, clone_path: Path) -> None:
        """Clean up cloned repository directory.

        Args:
            clone_path: Path to cloned repository
        """
        try:
            if clone_path.exists() and clone_path.is_dir():
                import shutil
                shutil.rmtree(clone_path)
                self.logger.info(f"Cleaned up clone directory: {clone_path}")
        except Exception as e:
            self.logger.error(f"Failed to cleanup clone directory {clone_path}: {e}")