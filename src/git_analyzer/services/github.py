"""GitHub platform service implementation."""

import asyncio
import json
import logging
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import RequestException, Timeout
from urllib3.util.retry import Retry

from .git_platform import (
    AuthenticationType,
    AuthenticationError,
    AuthenticationInfo,
    GitPlatformError,
    GitPlatformService,
    InvalidUrlError,
    PlatformInfo,
    RateLimitError,
    Repository,
    RepositoryNotFoundError,
    RepositoryTooLargeError,
    RepositoryVisibility,
)

logger = logging.getLogger(__name__)


class GitHubService(GitPlatformService):
    """GitHub platform service implementation."""

    def __init__(
        self,
        token: Optional[str] = None,
        base_url: str = "https://github.com",
        api_url: Optional[str] = None,
        auth_type: AuthenticationType = AuthenticationType.PERSONAL_ACCESS_TOKEN,
    ):
        """Initialize GitHub service.

        Args:
            token: GitHub personal access token or app token
            base_url: Base URL for GitHub (for GitHub Enterprise)
            api_url: API URL (auto-detected if not provided)
            auth_type: Authentication type
        """
        super().__init__(token, auth_type)
        self.base_url = base_url.rstrip("/")

        # Auto-detect API URL based on base URL
        if api_url is None:
            if self.base_url == "https://github.com":
                self.api_url = "https://api.github.com"
            else:
                # GitHub Enterprise Server
                self.api_url = f"{self.base_url}/api/v3"
        else:
            self.api_url = api_url.rstrip("/")

        # Set up HTTP session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Set default headers
        self.session.headers.update({
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "git-repo-analyzer/1.0"
        })

        if self.token:
            self.session.headers.update({
                "Authorization": f"token {self.token}"
            })

    @property
    def platform_info(self) -> PlatformInfo:
        """Get GitHub platform information."""
        is_enterprise = self.base_url != "https://github.com"

        return PlatformInfo(
            name="GitHub Enterprise" if is_enterprise else "GitHub",
            base_url=self.base_url,
            api_url=self.api_url,
            supported_auth_types=[
                AuthenticationType.PERSONAL_ACCESS_TOKEN,
                AuthenticationType.OAUTH_TOKEN,
            ],
            rate_limit_per_hour=5000 if self.token else 60,
            max_repo_size_mb=500,
        )

    async def authenticate(self) -> AuthenticationInfo:
        """Validate GitHub authentication."""
        self._log_operation("authenticate")

        if not self.token:
            return AuthenticationInfo(
                valid=False,
                error_message="No authentication token provided"
            )

        if not self._validate_token_format(self.token):
            return AuthenticationInfo(
                valid=False,
                error_message="Invalid token format"
            )

        try:
            # Get authenticated user info
            response = await self._make_request("GET", "/user")
            user_data = response.json()

            # Get rate limit info
            rate_limit_response = await self._make_request("GET", "/rate_limit")
            rate_limit_data = rate_limit_response.json()

            core_limit = rate_limit_data.get("resources", {}).get("core", {})

            auth_info = AuthenticationInfo(
                valid=True,
                username=user_data.get("login"),
                scopes=self._extract_scopes_from_headers(response.headers),
                rate_limit_remaining=core_limit.get("remaining"),
                rate_limit_reset=datetime.fromtimestamp(
                    core_limit.get("reset", 0), tz=timezone.utc
                ) if core_limit.get("reset") else None
            )

            self._auth_info = auth_info
            self._authenticated = True

            self.logger.info(f"Successfully authenticated as {auth_info.username}")
            return auth_info

        except requests.exceptions.RequestException as e:
            error_msg = f"Authentication failed: {str(e)}"
            if hasattr(e, "response") and e.response is not None:
                if e.response.status_code == 401:
                    error_msg = "Invalid or expired authentication token"
                elif e.response.status_code == 403:
                    error_msg = "Token does not have sufficient permissions"

            self.logger.error(error_msg)
            raise AuthenticationError(error_msg) from e

    async def list_repositories(
        self,
        owner: Optional[str] = None,
        include_forks: bool = False,
        include_archived: bool = False,
        visibility: Optional[RepositoryVisibility] = None,
    ) -> List[Repository]:
        """List repositories from GitHub."""
        self._log_operation("list_repositories", owner=owner)

        if not await self.is_authenticated():
            raise AuthenticationError("Not authenticated")

        repositories = []
        page = 1
        per_page = 100

        while True:
            try:
                if owner:
                    # List repositories for specific owner/organization
                    url = f"/users/{owner}/repos"
                else:
                    # List repositories for authenticated user
                    url = "/user/repos"

                params = {
                    "page": page,
                    "per_page": per_page,
                    "sort": "updated",
                    "direction": "desc"
                }

                # Add visibility filter if specified
                if visibility:
                    if visibility == RepositoryVisibility.PUBLIC:
                        params["visibility"] = "public"
                    elif visibility == RepositoryVisibility.PRIVATE:
                        params["visibility"] = "private"

                response = await self._make_request("GET", url, params=params)
                repos_data = response.json()

                if not repos_data:
                    break

                for repo_data in repos_data:
                    repo = self._parse_repository(repo_data)

                    # Apply filters
                    if not include_forks and repo.fork:
                        continue
                    if not include_archived and repo.archived:
                        continue

                    repositories.append(repo)

                # Check if there are more pages
                if len(repos_data) < per_page:
                    break

                page += 1

            except RateLimitError:
                # Re-raise rate limit errors
                raise
            except Exception as e:
                self.logger.error(f"Failed to list repositories: {e}")
                raise GitPlatformError(f"Failed to list repositories: {str(e)}") from e

        self.logger.info(f"Listed {len(repositories)} repositories")
        return repositories

    async def get_repository(self, owner: str, repo_name: str) -> Repository:
        """Get specific repository information."""
        self._log_operation("get_repository", owner=owner, repo=repo_name)

        try:
            url = f"/repos/{owner}/{repo_name}"
            response = await self._make_request("GET", url)
            repo_data = response.json()

            # Get languages
            languages_response = await self._make_request("GET", f"{url}/languages")
            languages_data = languages_response.json()

            repo = self._parse_repository(repo_data, languages_data)
            self.logger.info(f"Retrieved repository {repo.full_name}")
            return repo

        except requests.exceptions.RequestException as e:
            if hasattr(e, "response") and e.response is not None:
                if e.response.status_code == 404:
                    raise RepositoryNotFoundError(f"Repository {owner}/{repo_name} not found") from e
                elif e.response.status_code == 403:
                    raise AuthenticationError("Insufficient permissions to access repository") from e

            raise GitPlatformError(f"Failed to get repository {owner}/{repo_name}: {str(e)}") from e

    async def list_organizations(self) -> List[str]:
        """List organizations for authenticated user."""
        self._log_operation("list_organizations")

        if not await self.is_authenticated():
            raise AuthenticationError("Not authenticated")

        try:
            response = await self._make_request("GET", "/user/orgs")
            orgs_data = response.json()

            organizations = [org["login"] for org in orgs_data]
            self.logger.info(f"Listed {len(organizations)} organizations")
            return organizations

        except Exception as e:
            self.logger.error(f"Failed to list organizations: {e}")
            raise GitPlatformError(f"Failed to list organizations: {str(e)}") from e

    async def clone_repository(
        self,
        repository: Repository,
        target_dir: Path,
        shallow: bool = True,
        branch: Optional[str] = None,
    ) -> Path:
        """Clone repository using git command."""
        self._log_operation("clone_repository", repo=repository.full_name)

        # Check repository size
        size_mb = repository.size_bytes / (1024 * 1024)
        if size_mb > self.platform_info.max_repo_size_mb:
            raise RepositoryTooLargeError(
                f"Repository {repository.full_name} is {size_mb:.1f}MB, "
                f"exceeds limit of {self.platform_info.max_repo_size_mb}MB",
                size_mb=size_mb,
                max_size_mb=self.platform_info.max_repo_size_mb
            )

        # Determine clone URL
        clone_url = repository.clone_url
        if self.token and repository.visibility != RepositoryVisibility.PUBLIC:
            # Use authenticated HTTPS URL for private repos
            parsed_url = urlparse(clone_url)
            clone_url = f"{parsed_url.scheme}://{self.token}@{parsed_url.netloc}{parsed_url.path}"

        # Prepare target directory
        repo_dir = target_dir / repository.name
        if repo_dir.exists():
            self.logger.warning(f"Directory {repo_dir} already exists, removing it")
            import shutil
            shutil.rmtree(repo_dir)

        target_dir.mkdir(parents=True, exist_ok=True)

        # Build git clone command
        cmd = ["git", "clone"]

        if shallow:
            cmd.extend(["--depth", "1"])

        if branch:
            cmd.extend(["--branch", branch])
        else:
            cmd.extend(["--branch", repository.default_branch])

        cmd.extend([clone_url, str(repo_dir)])

        try:
            self.logger.info(f"Cloning repository {repository.full_name} to {repo_dir}")

            # Run git clone command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                check=True
            )

            self.logger.info(f"Successfully cloned {repository.full_name}")
            return repo_dir

        except subprocess.TimeoutExpired:
            self.cleanup_clone(repo_dir)
            raise GitPlatformError(f"Repository clone timed out after 5 minutes")
        except subprocess.CalledProcessError as e:
            self.cleanup_clone(repo_dir)
            error_msg = e.stderr.strip() if e.stderr else str(e)
            raise GitPlatformError(f"Failed to clone repository: {error_msg}") from e
        except Exception as e:
            self.cleanup_clone(repo_dir)
            raise GitPlatformError(f"Unexpected error during clone: {str(e)}") from e

    def parse_repository_url(self, url: str) -> tuple[str, str]:
        """Parse GitHub repository URL."""
        # GitHub URL patterns
        patterns = [
            # HTTPS URLs
            r"https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$",
            # SSH URLs
            r"git@github\.com:([^/]+)/([^/]+?)(?:\.git)?/?$",
            # GitHub Enterprise URLs
            r"https?://([^/]+)/([^/]+)/([^/]+?)(?:\.git)?/?$",
            r"git@([^:]+):([^/]+)/([^/]+?)(?:\.git)?/?$",
        ]

        for pattern in patterns:
            match = re.match(pattern, url.strip())
            if match:
                groups = match.groups()
                if len(groups) == 2:
                    # Standard GitHub
                    return groups[0], groups[1]
                elif len(groups) == 3:
                    # GitHub Enterprise - return owner and repo, ignore domain
                    return groups[1], groups[2]

        raise InvalidUrlError(f"Invalid GitHub repository URL format: {url}")

    def _validate_token_format(self, token: str) -> bool:
        """Validate GitHub token format."""
        if not token or not token.strip():
            return False

        token = token.strip()

        # GitHub personal access tokens (classic) start with 'ghp_'
        # GitHub App installation tokens start with 'ghs_'
        # OAuth tokens can have various formats
        # Fine-grained personal access tokens start with 'github_pat_'

        valid_prefixes = ['ghp_', 'ghs_', 'github_pat_']

        # Check for known prefixes or assume it's an OAuth token
        return any(token.startswith(prefix) for prefix in valid_prefixes) or len(token) >= 20

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        timeout: int = 30
    ) -> requests.Response:
        """Make HTTP request to GitHub API with rate limiting and error handling."""
        url = f"{self.api_url}{endpoint}"

        try:
            # Make the request in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.session.request(
                    method, url, params=params, json=data, timeout=timeout
                )
            )

            # Handle rate limiting
            if response.status_code == 429:
                reset_time = None
                if "X-RateLimit-Reset" in response.headers:
                    reset_timestamp = int(response.headers["X-RateLimit-Reset"])
                    reset_time = datetime.fromtimestamp(reset_timestamp, tz=timezone.utc)

                raise RateLimitError("GitHub API rate limit exceeded", reset_time)

            # Handle other HTTP errors
            response.raise_for_status()

            return response

        except Timeout:
            raise GitPlatformError(f"Request to {url} timed out")
        except RequestException as e:
            if hasattr(e, 'response') and e.response is not None:
                if e.response.status_code == 401:
                    raise AuthenticationError("Invalid or expired token")
                elif e.response.status_code == 403:
                    # Check if it's a rate limit or permission issue
                    if "rate limit" in e.response.text.lower():
                        raise RateLimitError("GitHub API rate limit exceeded")
                    else:
                        raise AuthenticationError("Insufficient permissions")
                elif e.response.status_code == 404:
                    raise RepositoryNotFoundError(f"Resource not found: {url}")

            raise GitPlatformError(f"GitHub API request failed: {str(e)}") from e

    def _parse_repository(self, repo_data: Dict, languages_data: Optional[Dict] = None) -> Repository:
        """Parse repository data from GitHub API response."""
        # Extract languages
        languages = []
        if languages_data:
            languages = list(languages_data.keys())
        elif "languages_url" in repo_data:
            # Languages not provided, extract from repo data if available
            languages = []

        # Parse visibility
        if repo_data.get("private"):
            visibility = RepositoryVisibility.PRIVATE
        else:
            visibility = RepositoryVisibility.PUBLIC

        # Parse dates
        created_at = datetime.fromisoformat(repo_data["created_at"].replace("Z", "+00:00"))
        updated_at = datetime.fromisoformat(repo_data["updated_at"].replace("Z", "+00:00"))

        return Repository(
            name=repo_data["name"],
            full_name=repo_data["full_name"],
            url=repo_data["html_url"],
            clone_url=repo_data["clone_url"],
            ssh_url=repo_data["ssh_url"],
            default_branch=repo_data.get("default_branch", "main"),
            visibility=visibility,
            languages=languages,
            size_bytes=repo_data.get("size", 0) * 1024,  # GitHub returns size in KB
            created_at=created_at,
            updated_at=updated_at,
            description=repo_data.get("description"),
            topics=repo_data.get("topics", []),
            fork=repo_data.get("fork", False),
            archived=repo_data.get("archived", False),
        )

    def _extract_scopes_from_headers(self, headers: Dict[str, str]) -> List[str]:
        """Extract OAuth scopes from response headers."""
        scopes_header = headers.get("X-OAuth-Scopes", "")
        if scopes_header:
            return [scope.strip() for scope in scopes_header.split(",")]
        return []