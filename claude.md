# Git Repository Analyzer - Development Instructions

## Project Context
We're building a Streamlit application that analyzes git repositories across multiple platforms (GitHub, Azure DevOps, BitBucket) and provides AI-powered insights into code architecture, dependencies, and technology stacks.

## Development Environment

### Python Package Management
- **Use standard Python venv and pip** for dependency management
- Always work within a virtual environment: `python -m venv venv`
- Activate environment before installing:
  - Linux/macOS: `source venv/bin/activate`
  - Windows: `.\venv\Scripts\Activate.ps1`
- Install from requirements: `pip install -r requirements.txt`
- Add new dependencies: `pip install <package>` then update requirements.txt
- Use `requirements.txt` for production and `requirements-dev.txt` for development
- Pin versions for reproducible builds

### Project Structure
```
git-repo-analyzer/
├── pyproject.toml          # uv configuration and dependencies
├── src/git_analyzer/       # Main package
│   ├── __init__.py
│   ├── config.py          # Configuration management
│   ├── services/          # Business logic services
│   ├── ui/               # Streamlit UI components
│   └── utils/            # Utility functions
├── tests/                # Test suite
├── .streamlit/           # Streamlit configuration
└── app.py               # Main Streamlit application
```

### Code Quality Standards

#### Testing
- Use `pytest` for all testing
- Aim for >90% test coverage
- Write tests BEFORE implementing features (TDD approach)
- Use fixtures for common test data and mocked services
- Test file naming: `test_<module_name>.py`

#### Code Style
- Use `black` for code formatting (line length: 88)
- Use `isort` for import sorting
- Use `flake8` for linting
- Use type hints for all function parameters and return values
- Use dataclasses or Pydantic models for structured data
- **Path Handling**: Always use `pathlib.Path` for cross-platform compatibility
  ```python
  from pathlib import Path
  
  # Good - cross-platform
  repo_path = Path("repositories") / repo_name
  config_file = Path.home() / ".gitconfig"
  
  # Avoid - platform-specific
  repo_path = "repositories/" + repo_name  # Unix-only
  repo_path = f"repositories\\{repo_name}"  # Windows-only
  ```

#### Documentation
- Add docstrings to all classes and public methods
- Use Google-style docstrings
- Include type information in docstrings
- Add inline comments for complex business logic

## Architectural Patterns

### Service Layer Pattern
All business logic lives in service classes. UI components only handle presentation.

Example service structure:
```python
from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class Repository:
    name: str
    url: str
    default_branch: str
    languages: List[str]

class GitPlatformService(ABC):
    @abstractmethod
    async def authenticate(self, token: str) -> bool:
        """Validate authentication token."""
        pass
    
    @abstractmethod
    async def list_repositories(self) -> List[Repository]:
        """List all accessible repositories."""
        pass
```

### Configuration Management
Use Pydantic for configuration with environment variable support:

```python
from pydantic import BaseSettings, SecretStr

class Settings(BaseSettings):
    github_token: Optional[SecretStr] = None
    openai_api_key: Optional[SecretStr] = None
    
    class Config:
        env_file = ".env"
        secrets_dir = ".streamlit/secrets.toml"
```

### Error Handling
Create custom exceptions for different error types:
```python
class GitAnalyzerError(Exception):
    """Base exception for git analyzer."""
    pass

class AuthenticationError(GitAnalyzerError):
    """Authentication failed."""
    pass

class RepositoryNotFoundError(GitAnalyzerError):
    """Repository not found."""
    pass
```

## Security Requirements

### Credential Handling
- Store all secrets in `.streamlit/secrets.toml` (never commit)
- Use `SecretStr` type for sensitive values
- Implement credential validation before use
- Add token rotation support for future enhancement

### Data Safety
- Auto-cleanup cloned repositories after analysis
- Implement size limits for repository clones (max 500MB)
- Sanitize all user inputs
- Log security events (auth failures, oversized requests)

## AI Integration Guidelines

### Prompt Engineering
Structure AI prompts with clear sections:
```python
ARCHITECTURE_PROMPT = """
Analyze the provided repository structure and code samples.

## Repository Structure:
{repo_structure}

## Key Files Content:
{key_files}

## Analysis Required:
1. Overall architecture pattern (MVC, microservices, monolith, etc.)
2. Major technology components and frameworks
3. Data flow patterns and storage mechanisms
4. API design and integration points

## Output Format:
Provide response as structured JSON with sections for each analysis type.
"""
```

### Service Integration
Always implement fallback strategies:
1. Primary AI service (user preference)
2. Secondary service if primary fails
3. Basic static analysis if all AI services fail

## Testing Guidelines

### Unit Tests
Mock external dependencies:
```python
@pytest.fixture
def mock_github_api():
    with patch('git_analyzer.services.github.GithubAPI') as mock:
        yield mock

def test_repository_listing(mock_github_api):
    # Test implementation
    pass
```

### Integration Tests
Test real API connections with test repositories:
```python
@pytest.mark.integration
@pytest.mark.skipif(not os.getenv('GITHUB_TOKEN'), reason="No GitHub token")
def test_github_integration():
    # Real API test
    pass
```

### Performance Tests
Add benchmarks for key operations:
```python
@pytest.mark.benchmark
def test_large_repo_analysis_performance(benchmark):
    result = benchmark(analyze_repository, large_repo_path)
    assert result.execution_time < 300  # 5 minutes max
```

## Streamlit Best Practices

### State Management
Use session state for persistent data:
```python
if 'repositories' not in st.session_state:
    st.session_state.repositories = []

# Access with st.session_state.repositories
```

### UI Components
Create reusable components:
```python
def render_repository_card(repo: Repository) -> None:
    with st.container():
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**{repo.name}**")
            st.write(f"Languages: {', '.join(repo.languages)}")
        with col2:
            if st.button("Analyze", key=f"analyze_{repo.name}"):
                st.session_state.selected_repo = repo
```

### Error Handling in UI
Always handle errors gracefully:
```python
try:
    repositories = git_service.list_repositories()
    for repo in repositories:
        render_repository_card(repo)
except AuthenticationError:
    st.error("Authentication failed. Please check your token.")
except Exception as e:
    st.error(f"An unexpected error occurred: {str(e)}")
    logger.exception("Repository listing failed")
```

## Commit Guidelines

### Message Format
Use conventional commits:
- `feat: add GitHub repository listing`
- `fix: handle rate limiting in OpenAI service`
- `test: add integration tests for BitBucket service`
- `docs: update API documentation`

### Commit Frequency
- Commit after each feature implementation
- Commit after adding tests
- Commit after fixing bugs
- Always ensure tests pass before committing

## Development Workflow

1. **Ensure virtual environment is activated**:
   - Linux/macOS: `source venv/bin/activate`
   - Windows: `.\venv\Scripts\Activate.ps1`
2. Read the current prompt from `@prompt_plan.md`
3. Write tests first (TDD)
4. Implement the feature
5. Ensure all tests pass: `pytest`
6. Run linting and formatting: `black src/ && isort src/ && flake8 src/`
7. Commit with descriptive message
8. Mark prompt as completed in prompt_plan.md

## Cross-Platform Commands

### Running Tests
**Linux/macOS:**
```bash
pytest --cov=git_analyzer --cov-report=term-missing
```

**Windows (PowerShell):**  
```powershell
pytest --cov=git_analyzer --cov-report=term-missing
```

### Code Quality
**Linux/macOS:**
```bash
black src/ tests/
isort src/ tests/
flake8 src/ tests/
mypy src/
```

**Windows (PowerShell):**
```powershell
black src\ tests\
isort src\ tests\
flake8 src\ tests\
mypy src\
```

### Installing New Dependencies
**Both platforms:**
```bash
pip install <package_name>
# Then update requirements.txt
pip freeze > requirements.txt
```

## Platform-Specific Notes

### Windows Considerations
- Use raw strings for file paths: `r"C:\path\to\file"`
- Handle path separators with `os.path.join()` or `pathlib.Path`
- Be aware of case-insensitive filesystem
- PowerShell execution policy may require adjustment

### Linux/macOS Considerations  
- Use forward slashes in paths
- Case-sensitive filesystem
- Shell scripts need execute permissions: `chmod +x script.sh`

## Pre-commit Hooks
Ensure these checks run before every commit:
- `black` formatting
- `isort` import sorting
- `flake8` linting
- `pytest` test execution
- Type checking with `mypy`

## Performance Requirements
- Repository cloning: <2 minutes for repos <100MB
- AI analysis: <3 minutes per repository
- UI responsiveness: <1 second for all user interactions
- Memory usage: <2GB peak during analysis

If performance issues arise, implement these optimizations:
1. Shallow git clones (`--depth=1`)
2. File sampling for large repositories
3. Async operations for I/O bound tasks
4. Caching for repeated analyses

## Debugging and Logging
Use structured logging throughout:
```python
import logging

logger = logging.getLogger(__name__)

def analyze_repository(repo_path: str) -> AnalysisResult:
    logger.info(f"Starting analysis of {repo_path}")
    try:
        # Analysis logic
        logger.info(f"Analysis completed successfully for {repo_path}")
        return result
    except Exception as e:
        logger.error(f"Analysis failed for {repo_path}: {str(e)}")
        raise
```

Remember: Focus on one prompt at a time, implement with tests, and ensure everything works before moving to the next prompt.