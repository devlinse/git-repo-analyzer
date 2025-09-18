"""Service layer for Git Repository Analyzer."""

from .git_platform import (
    GitPlatformService,
    Repository,
    RepositoryVisibility,
    AuthenticationType,
    AuthenticationInfo,
    PlatformInfo,
    GitPlatformError,
    AuthenticationError,
    RateLimitError,
    RepositoryNotFoundError,
    InvalidUrlError,
    RepositoryTooLargeError,
)
from .github import GitHubService
from .repository_manager import (
    RepositoryManager,
    TempDirectoryManager,
    CloneOptions,
    RepositoryMetadata,
    CloneResult,
    ProgressCallback,
)

__all__ = [
    "GitPlatformService",
    "GitHubService",
    "RepositoryManager",
    "TempDirectoryManager",
    "CloneOptions",
    "RepositoryMetadata",
    "CloneResult",
    "ProgressCallback",
    "Repository",
    "RepositoryVisibility",
    "AuthenticationType",
    "AuthenticationInfo",
    "PlatformInfo",
    "GitPlatformError",
    "AuthenticationError",
    "RateLimitError",
    "RepositoryNotFoundError",
    "InvalidUrlError",
    "RepositoryTooLargeError",
]