"""Main Streamlit application for Git Repository Analyzer."""

import streamlit as st
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from git_analyzer.config import settings


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title=settings.app_title,
        page_icon=settings.app_icon,
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title(f"{settings.app_icon} {settings.app_title}")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Home", "Repository Analysis", "Batch Analysis", "Settings"]
    )

    if page == "Home":
        show_home_page()
    elif page == "Repository Analysis":
        show_analysis_page()
    elif page == "Batch Analysis":
        show_batch_analysis_page()
    elif page == "Settings":
        show_settings_page()


def show_home_page():
    """Display the home page."""
    st.header("Welcome to Git Repository Analyzer")

    st.markdown("""
    This application analyzes Git repositories across multiple platforms and provides
    AI-powered insights into code architecture, dependencies, and technology stacks.

    ### Features:
    - **Multi-platform Support**: GitHub, Azure DevOps, BitBucket
    - **AI-Powered Analysis**: OpenAI, Google AI, Azure OpenAI integration
    - **Architecture Insights**: Identify patterns, frameworks, and design decisions
    - **Dependency Analysis**: Understand technology stack and dependencies
    - **Batch Processing**: Analyze multiple repositories at once

    ### Getting Started:
    1. Configure your API tokens in the Settings page
    2. Navigate to Repository Analysis to analyze a single repository
    3. Use Batch Analysis to process multiple repositories
    """)

    # Quick stats placeholder
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Repositories Analyzed", "0")
    with col2:
        st.metric("Platforms Supported", "3")
    with col3:
        st.metric("AI Models", "3")


def show_analysis_page():
    """Display the repository analysis page."""
    st.header("Repository Analysis")

    st.markdown("Analyze a single Git repository for architecture and technology insights.")

    # Repository input
    repo_input_method = st.radio(
        "How would you like to specify the repository?",
        ["URL", "Local Path"]
    )

    if repo_input_method == "URL":
        repo_url = st.text_input(
            "Repository URL",
            placeholder="https://github.com/username/repository"
        )

        # Platform detection
        if repo_url:
            if "github.com" in repo_url:
                st.info("= Detected GitHub repository")
            elif "dev.azure.com" in repo_url or "visualstudio.com" in repo_url:
                st.info("=7 Detected Azure DevOps repository")
            elif "bitbucket.org" in repo_url:
                st.info(">� Detected Bitbucket repository")
    else:
        repo_path = st.text_input(
            "Local Repository Path",
            placeholder="/path/to/local/repository"
        )

    # Analysis options
    st.subheader("Analysis Options")

    col1, col2 = st.columns(2)
    with col1:
        analyze_architecture = st.checkbox("Architecture Analysis", value=True)
        analyze_dependencies = st.checkbox("Dependency Analysis", value=True)
    with col2:
        analyze_code_quality = st.checkbox("Code Quality Metrics", value=False)
        analyze_security = st.checkbox("Security Analysis", value=False)

    # AI model selection
    ai_model = st.selectbox(
        "Select AI Model",
        ["OpenAI GPT-4", "Google Gemini Pro", "Azure OpenAI"],
        help="Choose the AI model for analysis"
    )

    if st.button("Start Analysis", type="primary"):
        st.info("🚀 Analysis feature coming soon! This will integrate with the repository services and AI models.")


def show_batch_analysis_page():
    """Display the batch analysis page."""
    st.header("Batch Analysis")

    st.markdown("Analyze multiple repositories in batch for comparative insights.")

    # Repository list input
    st.subheader("Repository List")

    input_method = st.radio(
        "How would you like to provide repositories?",
        ["Manual Entry", "CSV Upload", "Organization/User"]
    )

    if input_method == "Manual Entry":
        repo_list = st.text_area(
            "Repository URLs (one per line)",
            placeholder="https://github.com/user/repo1\nhttps://github.com/user/repo2"
        )
    elif input_method == "CSV Upload":
        uploaded_file = st.file_uploader("Upload CSV with repository URLs", type="csv")
        if uploaded_file:
            st.info("📁 CSV processing feature coming soon!")
    else:
        platform = st.selectbox("Platform", ["GitHub", "Azure DevOps", "Bitbucket"])
        org_user = st.text_input(f"{platform} Organization/User")
        if st.button("Fetch Repositories"):
            st.info("= Organization repository fetching coming soon!")

    # Analysis settings
    st.subheader("Batch Analysis Settings")

    col1, col2 = st.columns(2)
    with col1:
        max_repos = st.number_input("Maximum repositories to analyze", min_value=1, max_value=50, value=10)
        parallel_jobs = st.number_input("Parallel analysis jobs", min_value=1, max_value=5, value=2)
    with col2:
        output_format = st.selectbox("Output Format", ["JSON", "CSV", "Excel"])
        include_summary = st.checkbox("Include summary report", value=True)

    if st.button("Start Batch Analysis", type="primary"):
        st.info("=� Batch analysis feature coming soon!")


def show_settings_page():
    """Display the settings page."""
    st.header("Settings")

    st.markdown("Configure API tokens and application settings.")

    # API Configuration
    st.subheader("API Configuration")

    with st.expander("GitHub Settings", expanded=False):
        github_token = st.text_input(
            "GitHub Personal Access Token",
            type="password",
            help="Required for accessing private repositories and higher rate limits"
        )

    with st.expander("OpenAI Settings", expanded=False):
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Required for GPT-4 analysis"
        )

    with st.expander("Google AI Settings", expanded=False):
        google_key = st.text_input(
            "Google AI API Key",
            type="password",
            help="Required for Gemini Pro analysis"
        )

    with st.expander("Azure OpenAI Settings", expanded=False):
        azure_key = st.text_input(
            "Azure OpenAI API Key",
            type="password"
        )
        azure_endpoint = st.text_input(
            "Azure OpenAI Endpoint",
            placeholder="https://your-resource.openai.azure.com/"
        )

    # Application Settings
    st.subheader("Application Settings")

    col1, col2 = st.columns(2)
    with col1:
        max_repo_size = st.number_input(
            "Maximum repository size (MB)",
            min_value=1,
            max_value=2000,
            value=settings.max_repo_size_mb
        )
        clone_timeout = st.number_input(
            "Clone timeout (seconds)",
            min_value=30,
            max_value=600,
            value=settings.clone_timeout_seconds
        )
    with col2:
        analysis_timeout = st.number_input(
            "Analysis timeout (seconds)",
            min_value=60,
            max_value=1800,
            value=settings.analysis_timeout_seconds
        )

    if st.button("Save Settings"):
        st.success("� Settings saving functionality coming soon! Settings will be saved to .streamlit/secrets.toml")

    # System Info
    st.subheader("System Information")
    st.info(f"Application Version: {settings.__dict__.get('version', '0.1.0')}")


if __name__ == "__main__":
    main()