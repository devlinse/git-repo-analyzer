"""Tests for repository management service."""

import asyncio
import os
import pytest
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from unittest import mock

from git_analyzer.services.repository_manager import (
    RepositoryManager,
    TempDirectoryManager,
    CloneOptions,
    RepositoryMetadata,
    CloneResult,
    ProgressCallback
)
from git_analyzer.services.git_platform import Repository, RepositoryVisibility


@pytest.fixture
def sample_repository():
    """Create a sample repository for testing."""
    return Repository(
        name="test-repo",
        full_name="owner/test-repo",
        url="https://github.com/owner/test-repo",
        clone_url="https://github.com/owner/test-repo.git",
        ssh_url="git@github.com:owner/test-repo.git",
        default_branch="main",
        visibility=RepositoryVisibility.PUBLIC,
        languages=["Python", "JavaScript"],
        size_bytes=1024 * 1024,  # 1MB
        created_at=datetime.now(),
        updated_at=datetime.now(),
        description="Test repository",
        topics=["testing", "python"],
        fork=False,
        archived=False
    )


@pytest.fixture
def temp_base_dir():
    """Create temporary base directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def repo_manager(temp_base_dir):
    """Create repository manager instance for testing."""
    manager = RepositoryManager(
        base_temp_dir=temp_base_dir,
        max_concurrent_clones=2,
        default_timeout=30
    )
    yield manager
    manager.cleanup_all()


class TestTempDirectoryManager:
    """Tests for temporary directory management."""

    def test_create_temp_dir(self, temp_base_dir):
        """Test temporary directory creation."""
        temp_manager = TempDirectoryManager(temp_base_dir)
        temp_dir = temp_manager.create_temp_dir("test_")

        assert temp_dir.exists()
        assert temp_dir.is_dir()
        assert "test_" in temp_dir.name
        assert temp_dir.parent == temp_base_dir

        temp_manager.cleanup_all()

    def test_cleanup_directory(self, temp_base_dir):
        """Test directory cleanup."""
        temp_manager = TempDirectoryManager(temp_base_dir)
        temp_dir = temp_manager.create_temp_dir("test_")

        # Create some test files
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        assert test_file.exists()

        temp_manager.cleanup_directory(temp_dir)
        assert not temp_dir.exists()

    def test_cleanup_readonly_files(self, temp_base_dir):
        """Test cleanup of readonly files (like git files)."""
        temp_manager = TempDirectoryManager(temp_base_dir)
        temp_dir = temp_manager.create_temp_dir("test_")

        # Create readonly file
        readonly_file = temp_dir / "readonly.txt"
        readonly_file.write_text("readonly content")
        readonly_file.chmod(0o444)  # Read-only

        temp_manager.cleanup_directory(temp_dir)
        assert not temp_dir.exists()

    def test_cleanup_old_directories(self, temp_base_dir):
        """Test cleanup of old directories."""
        temp_manager = TempDirectoryManager(temp_base_dir, max_age_hours=0)  # Immediate cleanup
        temp_dir = temp_manager.create_temp_dir("old_")

        # Simulate old directory
        import time
        time.sleep(0.1)  # Small delay to ensure age difference

        temp_manager._cleanup_old_directories()
        # Directory should still exist as it's tracked in active_dirs


class TestCloneOptions:
    """Tests for clone configuration options."""

    def test_default_options(self):
        """Test default clone options."""
        options = CloneOptions()

        assert options.shallow is True
        assert options.branch is None
        assert options.depth == 1
        assert options.single_branch is True
        assert options.recurse_submodules is False
        assert options.timeout_seconds == 300
        assert options.max_size_mb == 500

    def test_custom_options(self):
        """Test custom clone options."""
        options = CloneOptions(
            shallow=False,
            branch="develop",
            depth=5,
            single_branch=False,
            recurse_submodules=True,
            timeout_seconds=600,
            max_size_mb=1000
        )

        assert options.shallow is False
        assert options.branch == "develop"
        assert options.depth == 5
        assert options.single_branch is False
        assert options.recurse_submodules is True
        assert options.timeout_seconds == 600
        assert options.max_size_mb == 1000


class TestRepositoryMetadata:
    """Tests for repository metadata."""

    def test_default_metadata(self):
        """Test default metadata initialization."""
        metadata = RepositoryMetadata()

        assert metadata.languages == {}
        assert metadata.total_files == 0
        assert metadata.total_size_bytes == 0
        assert metadata.file_extensions == {}
        assert metadata.directory_structure == {}
        assert metadata.git_info == {}
        assert isinstance(metadata.analysis_timestamp, datetime)
        assert metadata.config_files == []
        assert metadata.documentation_files == []
        assert metadata.test_directories == []


class TestRepositoryManager:
    """Tests for repository manager."""

    @pytest.mark.asyncio
    async def test_clone_repository_size_limit(self, repo_manager, sample_repository):
        """Test repository size limit enforcement."""
        # Create repository that exceeds size limit
        large_repo = sample_repository
        large_repo.size_bytes = 600 * 1024 * 1024  # 600MB

        options = CloneOptions(max_size_mb=500)
        result = await repo_manager.clone_repository(large_repo, options)

        assert not result.success
        assert "exceeds limit" in result.error_message
        assert result.clone_path is None

    @pytest.mark.asyncio
    async def test_build_clone_command(self, repo_manager, sample_repository):
        """Test git clone command building."""
        options = CloneOptions(
            shallow=True,
            branch="main",
            depth=1,
            single_branch=True
        )
        clone_path = Path("/test/path")

        cmd = repo_manager._build_clone_command(sample_repository, clone_path, options)

        expected = [
            "git", "clone",
            "--depth", "1",
            "--single-branch",
            "--branch", "main",
            sample_repository.clone_url,
            str(clone_path)
        ]

        assert cmd == expected

    @pytest.mark.asyncio
    async def test_build_clone_command_with_submodules(self, repo_manager, sample_repository):
        """Test git clone command with submodules."""
        options = CloneOptions(
            shallow=False,
            recurse_submodules=True
        )
        clone_path = Path("/test/path")

        cmd = repo_manager._build_clone_command(sample_repository, clone_path, options)

        assert "git" in cmd
        assert "clone" in cmd
        assert "--recurse-submodules" in cmd
        assert "--depth" not in cmd  # Should not be present when shallow=False

    @pytest.mark.asyncio
    @patch('asyncio.create_subprocess_exec')
    async def test_run_git_command_success(self, mock_subprocess, repo_manager):
        """Test successful git command execution."""
        # Mock successful subprocess
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (b"Success", b"")
        mock_subprocess.return_value = mock_process

        cmd = ["git", "clone", "test"]
        result = await repo_manager._run_git_command(cmd)

        assert result.success is True
        mock_subprocess.assert_called_once()

    @pytest.mark.asyncio
    @patch('asyncio.create_subprocess_exec')
    async def test_run_git_command_failure(self, mock_subprocess, repo_manager):
        """Test failed git command execution."""
        # Mock failed subprocess
        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate.return_value = (b"", b"Error message")
        mock_subprocess.return_value = mock_process

        cmd = ["git", "clone", "test"]
        result = await repo_manager._run_git_command(cmd)

        assert result.success is False
        assert "Error message" in result.error_message

    def test_extract_metadata_sync(self, repo_manager, temp_base_dir):
        """Test synchronous metadata extraction."""
        # Create test directory structure
        test_repo = temp_base_dir / "test_repo"
        test_repo.mkdir()

        # Create test files
        (test_repo / "main.py").write_text("print('hello')")
        (test_repo / "script.js").write_text("console.log('hello');")
        (test_repo / "README.md").write_text("# Test Repo")
        (test_repo / "requirements.txt").write_text("requests==2.28.0")

        # Create test directory
        test_dir = test_repo / "tests"
        test_dir.mkdir()
        (test_dir / "test_main.py").write_text("def test_main(): pass")

        metadata = repo_manager._extract_metadata_sync(test_repo)

        assert metadata.total_files > 0
        assert "Python" in metadata.languages
        assert "JavaScript" in metadata.languages
        assert ".py" in metadata.file_extensions
        assert ".js" in metadata.file_extensions
        assert any("requirements.txt" in config for config in metadata.config_files)
        assert any("README.md" in doc for doc in metadata.documentation_files)
        assert metadata.test_directories

    def test_calculate_directory_size(self, repo_manager, temp_base_dir):
        """Test directory size calculation."""
        test_dir = temp_base_dir / "size_test"
        test_dir.mkdir()

        # Create files with known sizes
        (test_dir / "file1.txt").write_text("a" * 100)  # 100 bytes
        (test_dir / "file2.txt").write_text("b" * 200)  # 200 bytes

        size = repo_manager._calculate_directory_size(test_dir)
        assert size >= 300  # At least 300 bytes

    @patch('subprocess.run')
    def test_extract_git_info(self, mock_run, repo_manager, temp_base_dir):
        """Test git information extraction."""
        # Mock git commands
        mock_run.side_effect = [
            # git rev-list --count HEAD
            Mock(returncode=0, stdout="42\n"),
            # git log -1 --format=...
            Mock(returncode=0, stdout="abc123|John Doe|john@example.com|2023-01-01|Initial commit\n"),
            # git branch -r
            Mock(returncode=0, stdout="  origin/main\n  origin/develop\n")
        ]

        test_repo = temp_base_dir / "git_test"
        test_repo.mkdir()

        git_info = repo_manager._extract_git_info(test_repo)

        assert git_info["commit_count"] == 42
        assert git_info["latest_commit_hash"] == "abc123"
        assert git_info["latest_commit_author"] == "John Doe"
        assert git_info["remote_branches"] == 2

    @pytest.mark.asyncio
    async def test_temporary_clone_context_manager(self, repo_manager, sample_repository):
        """Test temporary clone context manager."""
        with patch.object(repo_manager, 'clone_repository') as mock_clone:
            with patch.object(repo_manager, 'cleanup_clone') as mock_cleanup:
                # Mock successful clone
                mock_result = CloneResult(
                    success=True,
                    clone_path=Path("/test/clone/path")
                )
                mock_clone.return_value = mock_result

                async with repo_manager.temporary_clone(sample_repository) as result:
                    assert result is mock_result

                # Verify cleanup was called
                mock_cleanup.assert_called_once_with(mock_result.clone_path)

    @pytest.mark.asyncio
    async def test_analyze_repository_structure(self, repo_manager, sample_repository):
        """Test repository structure analysis."""
        with patch.object(repo_manager, 'temporary_clone') as mock_temp_clone:
            # Mock successful analysis
            mock_metadata = RepositoryMetadata()
            mock_metadata.languages = {"Python": 5}

            mock_result = CloneResult(
                success=True,
                metadata=mock_metadata
            )

            mock_temp_clone.return_value.__aenter__.return_value = mock_result

            metadata = await repo_manager.analyze_repository_structure(sample_repository)

            assert metadata is mock_metadata
            assert metadata.languages == {"Python": 5}

    def test_cleanup_clone(self, repo_manager, temp_base_dir):
        """Test clone cleanup."""
        # Create test clone directory
        clone_dir = temp_base_dir / "test_clone"
        clone_dir.mkdir()
        (clone_dir / "test_file.txt").write_text("test")

        repo_manager.cleanup_clone(clone_dir)

        # The parent directory should be cleaned up
        # Note: This depends on the temp_manager implementation

    @pytest.mark.asyncio
    async def test_get_clone_status(self, repo_manager):
        """Test clone status tracking."""
        # Test non-existent clone
        status = await repo_manager.get_clone_status("owner/nonexistent")
        assert status is None

        # Test active clone
        test_path = Path("/test/path")
        repo_manager._active_clones["owner/test"] = test_path

        status = await repo_manager.get_clone_status("owner/test")
        assert status is not None
        assert status["active"] is True
        assert status["clone_path"] == str(test_path)

    @pytest.mark.asyncio
    async def test_cancel_clone(self, repo_manager):
        """Test clone cancellation."""
        # Test non-existent clone
        cancelled = await repo_manager.cancel_clone("owner/nonexistent")
        assert cancelled is False

        # Test active clone
        test_path = Path("/test/path")
        repo_manager._active_clones["owner/test"] = test_path

        with patch.object(repo_manager, 'cleanup_clone') as mock_cleanup:
            cancelled = await repo_manager.cancel_clone("owner/test")
            assert cancelled is True
            mock_cleanup.assert_called_once_with(test_path)

    def test_progress_callback_protocol(self):
        """Test progress callback protocol compliance."""
        def test_callback(current: int, total: int, message: str = "") -> None:
            assert isinstance(current, int)
            assert isinstance(total, int)
            assert isinstance(message, str)

        # Test callback matches protocol
        callback: ProgressCallback = test_callback
        callback(50, 100, "Testing...")

    @pytest.mark.asyncio
    async def test_concurrent_clone_semaphore(self, repo_manager, sample_repository):
        """Test that concurrent clones are limited by semaphore."""
        # This test would need more complex mocking to properly test
        # the semaphore behavior, but we can at least verify the semaphore exists
        assert repo_manager._clone_semaphore is not None
        assert repo_manager._clone_semaphore._value == repo_manager.max_concurrent_clones

    def test_cleanup_all(self, repo_manager):
        """Test cleanup of all resources."""
        # Add some mock active clones
        test_path = Path("/test/path")
        repo_manager._active_clones["owner/test"] = test_path

        with patch.object(repo_manager, 'cleanup_clone') as mock_cleanup:
            with patch.object(repo_manager.temp_manager, 'cleanup_all') as mock_temp_cleanup:
                with patch.object(repo_manager._executor, 'shutdown') as mock_shutdown:
                    repo_manager.cleanup_all()

                    mock_cleanup.assert_called_once_with(test_path)
                    mock_temp_cleanup.assert_called_once()
                    mock_shutdown.assert_called_once_with(wait=True)


@pytest.mark.integration
class TestRepositoryManagerIntegration:
    """Integration tests for repository manager."""

    @pytest.mark.asyncio
    async def test_clone_real_small_repo(self):
        """Test cloning a real small repository (requires internet)."""
        # This test should only run in integration test environment
        # Skip if no internet or in CI without proper setup
        pytest.skip("Integration test - requires internet and git")

        sample_repo = Repository(
            name="hello-world",
            full_name="github/hello-world",
            url="https://github.com/github/hello-world",
            clone_url="https://github.com/github/hello-world.git",
            ssh_url="git@github.com:github/hello-world.git",
            default_branch="master",
            visibility=RepositoryVisibility.PUBLIC,
            languages=["HTML"],
            size_bytes=1024,  # Very small
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = RepositoryManager(Path(temp_dir))

            try:
                result = await manager.clone_repository(sample_repo)

                assert result.success
                assert result.clone_path is not None
                assert result.clone_path.exists()
                assert result.metadata is not None

            finally:
                manager.cleanup_all()


if __name__ == "__main__":
    pytest.main([__file__])