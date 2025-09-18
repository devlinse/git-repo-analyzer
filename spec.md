# Git Repository Analyzer - Project Specification

## Overview
A Streamlit application that analyzes git repositories across multiple platforms, providing AI-powered insights into code architecture, dependencies, and data flows.

## Core Features

### 1. Git Platform Integration
- **Supported Platforms**: GitHub, GitHub Enterprise, Azure DevOps, BitBucket
- **Authentication**: PAT-based (primary), SSH keys (future enhancement)
- **Operations**: Repository discovery, cloning, branch listing

### 2. Repository Management
- **Clone Strategy**: Shallow clones for analysis efficiency
- **Storage**: Temporary local storage with cleanup
- **Metadata Extraction**: Repository size, languages, structure

### 3. AI Analysis Integration
- **Supported Services**: 
  - OpenAI (ChatGPT)
  - Azure OpenAI
  - Google Gemini
- **Fallback Strategy**: Primary → Secondary → Tertiary service
- **Rate Limiting**: Built-in handling and retry logic

### 4. Code Analysis Outputs
- **Architecture Summary**: High-level system design, patterns used
- **Dependency Analysis**: Package managers, major dependencies, versions
- **Technology Stack**: Languages, frameworks, databases, services
- **Data Flow Mapping**: Input/output patterns, API interactions, data persistence

## Technical Requirements

### Performance
- **Analysis Time**: <5 minutes for repositories <100MB
- **Concurrent Analysis**: Support for 3 parallel repository analyses
- **Memory Usage**: <2GB peak memory usage

### Security
- **Credential Storage**: Encrypted PAT storage using streamlit secrets
- **Data Retention**: Auto-cleanup of cloned repositories after 24 hours
- **API Keys**: Secure storage and rotation support

### User Experience
- **Progress Tracking**: Real-time analysis progress indicators
- **Export Options**: JSON, PDF, and Markdown reports
- **History**: Analysis history with re-run capability

## Architecture Decisions
- **Framework**: Streamlit for rapid prototyping and deployment
- **Git Operations**: GitPython library for repository management
- **AI Integration**: Service-agnostic wrapper with adapter pattern
- **State Management**: Streamlit session state with persistent configuration
- **File Structure**: Modular design with separate services layer

## Success Criteria
1. Successfully connect to all four git platforms
2. Clone and analyze repositories up to 500MB
3. Generate accurate architecture summaries in <3 minutes
4. Handle authentication failures gracefully
5. Provide actionable insights for technical decision-making