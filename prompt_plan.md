# Git Repository Analyzer - Prompt Execution Plan

## Phase 1: Project Foundation
**Status**: ⏳ Pending

### Prompt 1.1: Project Setup and Structure
```
Set up the basic Streamlit project structure with proper Python packaging:
1. Create pyproject.toml with dependencies (streamlit, gitpython, openai, google-generativeai, azure-openai)
2. Set up src/git_analyzer/ package structure
3. Create basic Streamlit app.py with placeholder pages
4. Add .gitignore for Python projects
5. Create .streamlit/config.toml for app configuration
6. Add requirements.txt for deployment compatibility
7. Create basic README.md with setup instructions
```

### Prompt 1.2: Configuration Management
```
Implement configuration and secrets management:
1. Create src/git_analyzer/config.py with settings dataclass
2. Add support for .streamlit/secrets.toml integration
3. Implement environment variable fallbacks
4. Create configuration validation functions
5. Add tests for configuration loading
6. Ensure secure handling of API keys and PATs
```

## Phase 2: Git Platform Integration
**Status**: ⏳ Pending

### Prompt 2.1: Git Service Abstraction
```
Create an abstract base class for git platform integration:
1. Design GitPlatformService interface with required methods
2. Implement repository discovery, authentication validation
3. Add support for PAT-based authentication
4. Create error handling for common git operations
5. Write comprehensive tests for the base service
6. Add logging for debugging git operations
```

### Prompt 2.2: GitHub Integration
```
Implement GitHub platform service:
1. Create GitHubService extending GitPlatformService
2. Implement repository listing, cloning, metadata extraction
3. Handle GitHub API rate limiting and pagination
4. Support both GitHub.com and GitHub Enterprise
5. Add authentication validation and error handling
6. Write tests covering happy path and error scenarios
```

### Prompt 2.3: Azure DevOps Integration
```
Implement Azure DevOps platform service:
1. Create AzureDevOpsService extending GitPlatformService
2. Handle Azure DevOps REST API authentication
3. Implement repository discovery across organizations/projects
4. Add support for Azure DevOps git operations
5. Handle Azure-specific error responses
6. Write comprehensive test suite
```

### Prompt 2.4: BitBucket Integration
```
Implement BitBucket platform service:
1. Create BitBucketService extending GitPlatformService
2. Support both Bitbucket Cloud and Server APIs
3. Implement workspace/project repository listing
4. Handle BitBucket-specific authentication patterns
5. Add error handling for API limitations
6. Write tests for both Cloud and Server variants
```

## Phase 3: Repository Management
**Status**: ⏳ Pending

### Prompt 3.1: Repository Operations
```
Create repository management service:
1. Implement RepositoryManager with clone, cleanup operations
2. Add shallow cloning for efficiency
3. Create temporary storage management with auto-cleanup
4. Implement repository metadata extraction (languages, size, structure)
5. Add progress tracking for long operations
6. Write tests for all repository operations
```

### Prompt 3.2: File Analysis Engine
```
Build code analysis foundation:
1. Create FileAnalyzer for detecting languages, frameworks
2. Implement dependency file parsers (package.json, requirements.txt, etc.)
3. Add architecture pattern detection (MVC, microservices, etc.)
4. Create data flow analysis for API endpoints and databases
5. Add caching for repeated analysis
6. Write comprehensive tests for analysis accuracy
```

## Phase 4: AI Integration
**Status**: ⏳ Pending

### Prompt 4.1: AI Service Abstraction
```
Create AI service integration layer:
1. Design AIAnalysisService interface
2. Implement prompt templates for different analysis types
3. Add response parsing and validation
4. Create error handling and retry logic
5. Implement rate limiting and token management
6. Write tests for AI service interactions
```

### Prompt 4.2: OpenAI Integration
```
Implement OpenAI analysis service:
1. Create OpenAIService with ChatGPT and Azure OpenAI support
2. Add structured prompts for architecture analysis
3. Implement token counting and cost estimation
4. Handle API errors and rate limits gracefully
5. Add response validation and parsing
6. Write tests with mocked API responses
```

### Prompt 4.3: Gemini Integration
```
Implement Google Gemini analysis service:
1. Create GeminiService with proper authentication
2. Adapt prompts for Gemini's capabilities
3. Handle Gemini-specific response formats
4. Add safety settings and content filtering
5. Implement fallback strategies for failed requests
6. Write comprehensive test suite
```

## Phase 5: Streamlit UI
**Status**: ⏳ Pending

### Prompt 5.1: Main Interface
```
Build the primary Streamlit interface:
1. Create main app layout with sidebar navigation
2. Implement platform selection and authentication forms
3. Add repository selection with search and filtering
4. Create analysis configuration options (AI service, depth)
5. Add progress indicators and status updates
6. Style with Streamlit components for professional appearance
```

### Prompt 5.2: Results Presentation
```
Create analysis results interface:
1. Design tabbed layout for different analysis types
2. Implement architecture visualization with diagrams
3. Add dependency tree and version analysis display
4. Create data flow visualization components
5. Add export functionality (JSON, PDF, Markdown)
6. Implement analysis history and comparison features
```

## Phase 6: Testing and Polish
**Status**: ⏳ Pending

### Prompt 6.1: Integration Testing
```
Create end-to-end testing suite:
1. Set up pytest configuration with fixtures
2. Create integration tests for each git platform
3. Add AI service integration tests with mocked responses
4. Test complete analysis workflows
5. Add performance benchmarking tests
6. Create test data repositories for consistent testing
```

### Prompt 6.2: Documentation and Deployment
```
Finalize project documentation and deployment:
1. Create comprehensive README with setup instructions
2. Add API documentation for all services
3. Create deployment guide for various platforms
4. Add troubleshooting documentation
5. Create example configurations and use cases
6. Add CHANGELOG and contribution guidelines
```

## Completion Checklist
- [ ] All prompts marked as completed
- [ ] All tests passing
- [ ] Documentation complete
- [ ] Example configurations provided
- [ ] Security review completed
- [ ] Performance benchmarks met