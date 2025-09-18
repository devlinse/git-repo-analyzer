"""Repository management service for Git Repository Analyzer."""

import asyncio
import logging
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Tuple, Union, Callable
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
import weakref

from .git_platform import Repository, RepositoryTooLargeError, GitPlatformError

logger = logging.getLogger(__name__)


class ProgressCallback(Protocol):
    """Protocol for progress tracking callbacks."""

    def __call__(self, current: int, total: int, message: str = "") -> None:
        """Update progress.

        Args:
            current: Current progress value
            total: Total progress value
            message: Optional progress message
        """
        ...


@dataclass
class CloneOptions:
    """Configuration options for repository cloning."""
    shallow: bool = True
    branch: Optional[str] = None
    depth: int = 1
    single_branch: bool = True
    recurse_submodules: bool = False
    timeout_seconds: int = 300
    max_size_mb: int = 500


@dataclass
class RepositoryMetadata:
    """Metadata extracted from a repository."""
    languages: Dict[str, int] = field(default_factory=dict)
    total_files: int = 0
    total_size_bytes: int = 0
    file_extensions: Dict[str, int] = field(default_factory=dict)
    directory_structure: Dict[str, int] = field(default_factory=dict)
    git_info: Dict[str, Union[str, int]] = field(default_factory=dict)
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    config_files: List[str] = field(default_factory=list)
    documentation_files: List[str] = field(default_factory=list)
    test_directories: List[str] = field(default_factory=list)


@dataclass
class CloneResult:
    """Result of a repository clone operation."""
    success: bool
    clone_path: Optional[Path] = None
    metadata: Optional[RepositoryMetadata] = None
    error_message: Optional[str] = None
    clone_time_seconds: float = 0.0
    size_mb: float = 0.0


class TempDirectoryManager:
    """Manages temporary directories with automatic cleanup."""

    def __init__(self, base_dir: Optional[Path] = None, max_age_hours: int = 24):
        """Initialize temporary directory manager.

        Args:
            base_dir: Base directory for temporary files
            max_age_hours: Maximum age for temporary directories before cleanup
        """
        self.base_dir = base_dir or Path(tempfile.gettempdir()) / "git-repo-analyzer"
        self.max_age_hours = max_age_hours
        self._active_dirs: weakref.WeakSet = weakref.WeakSet()

        # Ensure base directory exists
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Clean up old directories on initialization
        self._cleanup_old_directories()

    def create_temp_dir(self, prefix: str = "repo_") -> Path:
        """Create a new temporary directory.

        Args:
            prefix: Prefix for directory name

        Returns:
            Path to created directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        dir_name = f"{prefix}{timestamp}"
        temp_dir = self.base_dir / dir_name
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Track directory for cleanup
        self._active_dirs.add(temp_dir)

        logger.debug(f"Created temporary directory: {temp_dir}")
        return temp_dir

    def cleanup_directory(self, directory: Path) -> None:
        """Clean up a specific directory.

        Args:
            directory: Directory to clean up
        """
        try:
            if directory.exists() and directory.is_dir():
                # Make files writable before deletion (handles Git readonly files)
                self._make_writable(directory)
                shutil.rmtree(directory)
                logger.debug(f"Cleaned up directory: {directory}")
        except Exception as e:
            logger.error(f"Failed to cleanup directory {directory}: {e}")

    def _make_writable(self, path: Path) -> None:
        """Make path and all contents writable.

        Args:
            path: Path to make writable
        """
        try:
            if path.is_file():
                path.chmod(0o666)
            elif path.is_dir():
                path.chmod(0o777)
                for item in path.rglob("*"):
                    if item.is_file():
                        item.chmod(0o666)
                    elif item.is_dir():
                        item.chmod(0o777)
        except Exception as e:
            logger.warning(f"Failed to make path writable {path}: {e}")

    def _cleanup_old_directories(self) -> None:
        """Clean up old temporary directories."""
        try:
            if not self.base_dir.exists():
                return

            cutoff_time = datetime.now() - timedelta(hours=self.max_age_hours)

            for item in self.base_dir.iterdir():
                if item.is_dir():
                    # Check modification time
                    mod_time = datetime.fromtimestamp(item.stat().st_mtime)
                    if mod_time < cutoff_time:
                        self.cleanup_directory(item)

        except Exception as e:
            logger.error(f"Failed to cleanup old directories: {e}")

    def cleanup_all(self) -> None:
        """Clean up all tracked directories."""
        # Convert to list to avoid modifying set during iteration
        dirs_to_clean = list(self._active_dirs)
        for directory in dirs_to_clean:
            self.cleanup_directory(directory)


class RepositoryManager:
    """Manages git repository operations including cloning, metadata extraction, and cleanup."""

    def __init__(
        self,
        base_temp_dir: Optional[Path] = None,
        max_concurrent_clones: int = 3,
        default_timeout: int = 300
    ):
        """Initialize repository manager.

        Args:
            base_temp_dir: Base directory for temporary repositories
            max_concurrent_clones: Maximum number of concurrent clone operations
            default_timeout: Default timeout for operations in seconds
        """
        self.temp_manager = TempDirectoryManager(base_temp_dir)
        self.max_concurrent_clones = max_concurrent_clones
        self.default_timeout = default_timeout
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Thread pool for parallel operations
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent_clones)

        # Semaphore to limit concurrent clones
        self._clone_semaphore = asyncio.Semaphore(max_concurrent_clones)

        # Track active operations for cleanup
        self._active_clones: Dict[str, Path] = {}

    async def clone_repository(
        self,
        repository: Repository,
        options: Optional[CloneOptions] = None,
        progress_callback: Optional[ProgressCallback] = None
    ) -> CloneResult:
        """Clone a repository with configurable options.

        Args:
            repository: Repository to clone
            options: Clone configuration options
            progress_callback: Optional progress tracking callback

        Returns:
            CloneResult with operation details
        """
        async with self._clone_semaphore:
            return await self._clone_repository_impl(repository, options, progress_callback)

    async def _clone_repository_impl(
        self,
        repository: Repository,
        options: Optional[CloneOptions],
        progress_callback: Optional[ProgressCallback]
    ) -> CloneResult:
        """Internal implementation of repository cloning."""
        start_time = time.time()
        clone_options = options or CloneOptions()

        # Check repository size limits
        size_mb = repository.size_bytes / (1024 * 1024)
        if size_mb > clone_options.max_size_mb:
            return CloneResult(
                success=False,
                error_message=f"Repository size ({size_mb:.1f}MB) exceeds limit ({clone_options.max_size_mb}MB)",
                clone_time_seconds=time.time() - start_time
            )

        # Create temporary directory
        clone_dir = self.temp_manager.create_temp_dir(f"clone_{repository.name}_")
        clone_path = clone_dir / repository.name

        try:
            # Track active clone
            self._active_clones[repository.full_name] = clone_path

            if progress_callback:
                progress_callback(0, 100, "Initializing clone...")

            # Build git clone command
            cmd = self._build_clone_command(repository, clone_path, clone_options)

            if progress_callback:
                progress_callback(10, 100, "Starting git clone...")

            # Execute clone with timeout
            result = await asyncio.wait_for(
                self._run_git_command(cmd, progress_callback),
                timeout=clone_options.timeout_seconds
            )

            if not result.success:
                return CloneResult(
                    success=False,
                    error_message=result.error_message,
                    clone_time_seconds=time.time() - start_time
                )

            if progress_callback:
                progress_callback(70, 100, "Extracting metadata...")

            # Extract repository metadata
            metadata = await self._extract_metadata(clone_path, progress_callback)

            if progress_callback:
                progress_callback(100, 100, "Clone completed successfully")

            clone_time = time.time() - start_time
            actual_size_mb = self._calculate_directory_size(clone_path) / (1024 * 1024)

            return CloneResult(
                success=True,
                clone_path=clone_path,
                metadata=metadata,
                clone_time_seconds=clone_time,
                size_mb=actual_size_mb
            )

        except asyncio.TimeoutError:
            self.cleanup_clone(clone_path)
            return CloneResult(
                success=False,
                error_message=f"Clone operation timed out after {clone_options.timeout_seconds} seconds",
                clone_time_seconds=time.time() - start_time
            )
        except Exception as e:
            self.cleanup_clone(clone_path)
            return CloneResult(
                success=False,
                error_message=f"Clone failed: {str(e)}",
                clone_time_seconds=time.time() - start_time
            )
        finally:
            # Remove from active clones
            self._active_clones.pop(repository.full_name, None)

    def _build_clone_command(
        self,
        repository: Repository,
        clone_path: Path,
        options: CloneOptions
    ) -> List[str]:
        """Build git clone command with options.

        Args:
            repository: Repository to clone
            clone_path: Target clone path
            options: Clone options

        Returns:
            Git command as list of arguments
        """
        cmd = ["git", "clone"]

        if options.shallow:
            cmd.extend(["--depth", str(options.depth)])

        if options.single_branch:
            cmd.append("--single-branch")

        if options.branch:
            cmd.extend(["--branch", options.branch])

        if options.recurse_submodules:
            cmd.append("--recurse-submodules")

        cmd.extend([repository.clone_url, str(clone_path)])

        return cmd

    async def _run_git_command(
        self,
        cmd: List[str],
        progress_callback: Optional[ProgressCallback] = None
    ) -> CloneResult:
        """Run git command asynchronously.

        Args:
            cmd: Git command to run
            progress_callback: Optional progress callback

        Returns:
            CloneResult with command execution details
        """
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=dict(os.environ, GIT_TERMINAL_PROMPT="0")  # Disable password prompts
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                return CloneResult(success=True)
            else:
                error_msg = stderr.decode('utf-8', errors='ignore') if stderr else "Unknown git error"
                return CloneResult(
                    success=False,
                    error_message=f"Git command failed: {error_msg}"
                )

        except Exception as e:
            return CloneResult(
                success=False,
                error_message=f"Failed to execute git command: {str(e)}"
            )

    async def _extract_metadata(
        self,
        repo_path: Path,
        progress_callback: Optional[ProgressCallback] = None
    ) -> RepositoryMetadata:
        """Extract metadata from cloned repository.

        Args:
            repo_path: Path to cloned repository
            progress_callback: Optional progress callback

        Returns:
            RepositoryMetadata with extracted information
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._extract_metadata_sync,
            repo_path,
            progress_callback
        )

    def _extract_metadata_sync(
        self,
        repo_path: Path,
        progress_callback: Optional[ProgressCallback] = None
    ) -> RepositoryMetadata:
        """Synchronous metadata extraction implementation."""
        metadata = RepositoryMetadata()

        try:
            # File analysis
            total_files = 0
            file_extensions = {}
            languages = {}
            config_files = []
            documentation_files = []
            test_directories = []

            # Language detection patterns
            language_patterns = {
                '.py': 'Python',
                '.js': 'JavaScript',
                '.ts': 'TypeScript',
                '.java': 'Java',
                '.cs': 'C#',
                '.go': 'Go',
                '.rs': 'Rust',
                '.php': 'PHP',
                '.rb': 'Ruby',
                '.cpp': 'C++',
                '.c': 'C',
                '.html': 'HTML',
                '.css': 'CSS',
                '.sql': 'SQL',
                '.sh': 'Shell',
                '.yml': 'YAML',
                '.yaml': 'YAML',
                '.json': 'JSON',
                '.xml': 'XML'
            }

            # Config file patterns
            config_patterns = [
                'package.json', 'requirements.txt', 'Cargo.toml', 'pom.xml',
                'Gemfile', 'composer.json', 'setup.py', 'pyproject.toml',
                '.gitignore', 'Dockerfile', 'docker-compose.yml'
            ]

            # Documentation patterns
            doc_patterns = [
                'README', 'readme', 'CHANGELOG', 'changelog',
                'LICENSE', 'license', 'CONTRIBUTING', 'contributing'
            ]

            # Walk through repository files
            for root, dirs, files in os.walk(repo_path):
                root_path = Path(root)

                # Skip .git directory
                dirs[:] = [d for d in dirs if d != '.git']

                # Check for test directories
                if any(test_name in root_path.name.lower() for test_name in ['test', 'tests', 'spec']):
                    test_directories.append(str(root_path.relative_to(repo_path)))

                for file in files:
                    file_path = root_path / file
                    total_files += 1

                    # Track file extensions
                    suffix = file_path.suffix.lower()
                    file_extensions[suffix] = file_extensions.get(suffix, 0) + 1

                    # Detect languages
                    if suffix in language_patterns:
                        lang = language_patterns[suffix]
                        languages[lang] = languages.get(lang, 0) + 1

                    # Identify config files
                    if file.lower() in [p.lower() for p in config_patterns]:
                        config_files.append(str(file_path.relative_to(repo_path)))

                    # Identify documentation files
                    if any(doc in file.lower() for doc in doc_patterns):
                        documentation_files.append(str(file_path.relative_to(repo_path)))

            # Calculate directory structure
            directory_structure = {}
            for root, dirs, files in os.walk(repo_path):
                level = len(Path(root).relative_to(repo_path).parts)
                directory_structure[f"level_{level}"] = directory_structure.get(f"level_{level}", 0) + len(dirs)

            # Get git information
            git_info = self._extract_git_info(repo_path)

            # Calculate total size
            total_size = self._calculate_directory_size(repo_path)

            metadata.languages = languages
            metadata.total_files = total_files
            metadata.total_size_bytes = total_size
            metadata.file_extensions = file_extensions
            metadata.directory_structure = directory_structure
            metadata.git_info = git_info
            metadata.config_files = config_files
            metadata.documentation_files = documentation_files
            metadata.test_directories = test_directories

        except Exception as e:
            self.logger.error(f"Failed to extract metadata from {repo_path}: {e}")

        return metadata

    def _extract_git_info(self, repo_path: Path) -> Dict[str, Union[str, int]]:
        """Extract git repository information.

        Args:
            repo_path: Path to repository

        Returns:
            Dictionary with git information
        """
        git_info = {}

        try:
            # Get commit count
            result = subprocess.run(
                ["git", "rev-list", "--count", "HEAD"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                git_info["commit_count"] = int(result.stdout.strip())

            # Get latest commit info
            result = subprocess.run(
                ["git", "log", "-1", "--format=%H|%an|%ae|%ad|%s"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split('|', 4)
                if len(parts) >= 5:
                    git_info.update({
                        "latest_commit_hash": parts[0],
                        "latest_commit_author": parts[1],
                        "latest_commit_email": parts[2],
                        "latest_commit_date": parts[3],
                        "latest_commit_message": parts[4]
                    })

            # Get branch information
            result = subprocess.run(
                ["git", "branch", "-r"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                branches = [line.strip().replace('origin/', '') for line in result.stdout.split('\n') if line.strip()]
                git_info["remote_branches"] = len(branches)

        except Exception as e:
            self.logger.warning(f"Failed to extract git info: {e}")

        return git_info

    def _calculate_directory_size(self, directory: Path) -> int:
        """Calculate total size of directory in bytes.

        Args:
            directory: Directory to calculate size for

        Returns:
            Size in bytes
        """
        total_size = 0
        try:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    file_path = Path(root) / file
                    try:
                        total_size += file_path.stat().st_size
                    except (OSError, FileNotFoundError):
                        continue
        except Exception as e:
            self.logger.warning(f"Failed to calculate directory size: {e}")

        return total_size

    def cleanup_clone(self, clone_path: Path) -> None:
        """Clean up a cloned repository.

        Args:
            clone_path: Path to cloned repository
        """
        try:
            self.temp_manager.cleanup_directory(clone_path.parent)
        except Exception as e:
            self.logger.error(f"Failed to cleanup clone {clone_path}: {e}")

    @asynccontextmanager
    async def temporary_clone(
        self,
        repository: Repository,
        options: Optional[CloneOptions] = None,
        progress_callback: Optional[ProgressCallback] = None
    ):
        """Context manager for temporary repository cloning with automatic cleanup.

        Args:
            repository: Repository to clone
            options: Clone options
            progress_callback: Progress tracking callback

        Yields:
            CloneResult with clone information
        """
        result = await self.clone_repository(repository, options, progress_callback)
        try:
            yield result
        finally:
            if result.success and result.clone_path:
                self.cleanup_clone(result.clone_path)

    async def analyze_repository_structure(
        self,
        repository: Repository,
        progress_callback: Optional[ProgressCallback] = None
    ) -> Optional[RepositoryMetadata]:
        """Analyze repository structure without keeping the clone.

        Args:
            repository: Repository to analyze
            progress_callback: Progress tracking callback

        Returns:
            RepositoryMetadata if successful, None otherwise
        """
        async with self.temporary_clone(repository, progress_callback=progress_callback) as clone_result:
            if clone_result.success:
                return clone_result.metadata
            else:
                self.logger.error(f"Failed to analyze repository structure: {clone_result.error_message}")
                return None

    def cleanup_all(self) -> None:
        """Clean up all temporary directories and resources."""
        # Cleanup all active clones
        for repo_name, clone_path in list(self._active_clones.items()):
            self.cleanup_clone(clone_path)

        # Cleanup temp manager
        self.temp_manager.cleanup_all()

        # Shutdown executor
        self._executor.shutdown(wait=True)

    async def get_clone_status(self, repository_name: str) -> Optional[Dict[str, Union[str, bool]]]:
        """Get status of an active clone operation.

        Args:
            repository_name: Full name of repository (owner/repo)

        Returns:
            Status dictionary if clone is active, None otherwise
        """
        if repository_name in self._active_clones:
            clone_path = self._active_clones[repository_name]
            return {
                "active": True,
                "clone_path": str(clone_path),
                "started_at": "unknown"  # Could be enhanced to track start times
            }
        return None

    async def cancel_clone(self, repository_name: str) -> bool:
        """Cancel an active clone operation.

        Args:
            repository_name: Full name of repository (owner/repo)

        Returns:
            True if clone was cancelled, False if not found
        """
        if repository_name in self._active_clones:
            clone_path = self._active_clones[repository_name]
            self.cleanup_clone(clone_path)
            return True
        return False

    def __del__(self):
        """Cleanup on object destruction."""
        try:
            self.cleanup_all()
        except:
            pass  # Ignore errors during cleanup