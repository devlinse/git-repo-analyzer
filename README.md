# Git Repository Analyzer

AI-powered git repository analysis tool with multi-platform support for GitHub, Azure DevOps, and BitBucket.

## Features

- 🔍 **Multi-platform Support**: Analyze repositories from GitHub, Azure DevOps, and BitBucket
- 🤖 **AI-Powered Analysis**: Leverage OpenAI GPT-4, Google Gemini Pro, or Azure OpenAI for insights
- 🏗️ **Architecture Detection**: Identify design patterns, frameworks, and architectural decisions
- 📦 **Dependency Analysis**: Understand technology stacks and dependency relationships
- 🔄 **Batch Processing**: Analyze multiple repositories simultaneously
- 🌐 **Web Interface**: User-friendly Streamlit web application
- 🔒 **Secure**: Proper handling of API tokens and credentials

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Git installed on your system

### Installation

#### Linux/macOS

```bash
# Clone the repository
git clone https://github.com/username/git-repo-analyzer.git
cd git-repo-analyzer

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

#### Windows

```powershell
# Clone the repository
git clone https://github.com/username/git-repo-analyzer.git
cd git-repo-analyzer

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Development Setup

#### Linux/macOS

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Run linting and formatting
black src/ tests/
isort src/ tests/
flake8 src/ tests/
mypy src/
```

#### Windows

```powershell
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Run linting and formatting
black src\ tests\
isort src\ tests\
flake8 src\ tests\
mypy src\
```

## Configuration

### API Keys

Create a `.streamlit/secrets.toml` file with your API keys:

```toml
# GitHub (optional - for private repos and higher rate limits)
GITHUB_TOKEN = "your_github_token_here"

# AI Services (choose one or more)
OPENAI_API_KEY = "your_openai_key_here"
GOOGLE_AI_API_KEY = "your_google_ai_key_here"

# Azure OpenAI (if using Azure)
AZURE_OPENAI_API_KEY = "your_azure_key_here"
AZURE_OPENAI_ENDPOINT = "https://your-resource.openai.azure.com/"
```

### Application Settings

You can modify settings in `src/git_analyzer/config.py` or through the web interface:

- `max_repo_size_mb`: Maximum repository size to analyze (default: 500MB)
- `clone_timeout_seconds`: Timeout for repository cloning (default: 300s)
- `analysis_timeout_seconds`: Timeout for AI analysis (default: 180s)

## Usage

1. **Start the Application**:
   ```bash
   streamlit run app.py
   ```

2. **Configure API Keys**:
   - Navigate to the Settings page
   - Enter your API tokens for the platforms and AI services you want to use

3. **Analyze a Repository**:
   - Go to "Repository Analysis"
   - Enter a repository URL or local path
   - Select analysis options
   - Choose an AI model
   - Click "Start Analysis"

4. **Batch Analysis**:
   - Use "Batch Analysis" for multiple repositories
   - Provide a list of repository URLs
   - Configure batch settings
   - Monitor progress and download results

## Supported Platforms

### Git Hosting Platforms
- **GitHub**: Public and private repositories (with token)
- **Azure DevOps**: Git repositories in Azure DevOps Services
- **BitBucket**: Git repositories (coming soon)

### AI Services
- **OpenAI GPT-4**: Most comprehensive analysis
- **Google Gemini Pro**: Fast and reliable analysis
- **Azure OpenAI**: Enterprise-grade OpenAI models

## Architecture

```
git-repo-analyzer/
├── app.py                      # Main Streamlit application
├── src/git_analyzer/           # Core package
│   ├── config.py              # Configuration management
│   ├── services/              # Business logic services
│   │   ├── github.py          # GitHub API integration
│   │   ├── azure_devops.py    # Azure DevOps integration
│   │   ├── ai_analyzer.py     # AI analysis services
│   │   └── repository.py      # Repository operations
│   ├── ui/                    # UI components
│   └── utils/                 # Utility functions
├── tests/                     # Test suite
├── requirements.txt           # Production dependencies
├── requirements-dev.txt       # Development dependencies
└── .streamlit/               # Streamlit configuration
```

## Development

### Code Quality Standards

This project follows strict code quality standards:

- **Formatting**: Black (88 character line length)
- **Import Sorting**: isort
- **Linting**: Flake8
- **Type Checking**: MyPy
- **Testing**: Pytest with >90% coverage target

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=git_analyzer --cov-report=term-missing

# Run only unit tests
pytest -m "not integration"

# Run integration tests (requires API tokens)
pytest -m integration
```

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes following the code quality standards
4. Run tests and ensure they pass
5. Submit a pull request

## Security

- API tokens are stored securely in `.streamlit/secrets.toml`
- Temporary repository clones are automatically cleaned up
- Size limits prevent excessive resource usage
- Input validation protects against malicious repository URLs

## Troubleshooting

### Common Issues

**Virtual Environment Not Activating (Windows)**:
```powershell
# If PowerShell execution policy prevents activation
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Import Errors**:
```bash
# Ensure you're in the project directory and virtual environment is activated
pip install -r requirements.txt
```

**Streamlit Not Starting**:
```bash
# Check if port 8501 is available or specify a different port
streamlit run app.py --server.port 8502
```

### Performance Optimization

For large repositories:
- Use shallow clones (automatic for repositories >100MB)
- Enable file sampling for analysis
- Adjust timeout settings in configuration

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/) for the web interface
- Uses [GitPython](https://gitpython.readthedocs.io/) for Git operations
- AI analysis powered by OpenAI, Google AI, and Azure OpenAI
- Repository platforms: GitHub, Azure DevOps, BitBucket