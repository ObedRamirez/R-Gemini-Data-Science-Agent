# System Architecture

## Overview

Dynamic R Analyst is a multi-agent system that orchestrates specialized AI agents to automate R-based data science workflows.

## Agent Architecture

### 1. Master Agent (Orchestrator)
**File**: `dynamic_r_analyst_v2.py`

**Responsibilities**:
- Manages Streamlit UI
- Coordinates agent workflow
- Handles file I/O
- Manages Docker execution

**Technology Stack**:
- Streamlit for UI
- LangChain AgentExecutor
- Subprocess for Docker communication

### 2. Prompt Translation Agent
**File**: `data_science_prompt_agent_v2.py`

**Responsibilities**:
- Translates natural language to R specifications
- Identifies analysis type (EDA, modeling, etc.)
- Extracts variable requirements
- Generates detailed coding instructions

**Input**: User's natural language request
**Output**: Structured prompt with analysis specifications

**LLM**: Google Gemini (via langchain-google-genai)

### 3. R Coding Agent
**File**: `data_science_coding_agent_v2.py`

**Responsibilities**:
- Generates complete RMarkdown documents
- Includes data loading, analysis, and visualization
- Follows R best practices
- Creates reproducible code

**Input**: Detailed specifications from Prompt Agent
**Output**: Complete RMarkdown (.Rmd) file

**Features**:
- Comprehensive error handling
- Automatic package loading
- YAML frontmatter generation
- Structured code sections

### 4. Audit Agent
**File**: `data_science_audit_agent_v2.py`

**Responsibilities**:
- Syntax validation
- Error detection and correction
- Code quality improvement
- Best practices enforcement

**Input**: Generated R/RMarkdown code
**Output**: Corrected, production-ready code

**Validation Checks**:
- Syntax errors
- Missing packages
- Variable naming
- Code structure

## Workflow Diagram

User Request
↓
[Streamlit UI]
↓
[Prompt Agent] → Detailed Specifications
↓
[Coding Agent] → RMarkdown Code
↓
[Audit Agent] → Validated Code
↓
[Docker Container] → Execute R Code
↓
HTML Report ← Results

text

## Docker Architecture

### R Environment Container
**Image**: `r_data_science:v2.0`
**Base**: `rocker/tidyverse:4.4.1`

**Installed Packages** (65 explicit + 400 dependencies):
- Data Processing: data.table, dplyr, tidyr, arrow
- Visualization: ggplot2, plotly, gganimate
- ML: caret, xgboost, randomForest, glmnet
- Statistics: Statistical tests, correlation analysis
- Development: devtools, testthat

**Execution Model**:
1. Code written to container filesystem
2. Container executes RMarkdown rendering
3. Results copied back to host
4. Container automatically removed

## Data Flow

1. **Input**: CSV file uploaded via Streamlit
2. **Storage**: Saved as `data.csv` in working directory
3. **Transfer**: Mounted into Docker container at `/app`
4. **Processing**: R code reads from `/app/data.csv`
5. **Output**: HTML report generated in `/app`
6. **Retrieval**: Copied back to host filesystem
7. **Display**: Available for download in Streamlit

## Technology Stack

### Backend
- **Python 3.8+**: Core language
- **LangChain**: Agent framework
- **Google Gemini**: LLM provider
- **Pydantic**: Data validation

### Frontend
- **Streamlit**: Web UI
- **Pandas**: Data preview

### Execution Environment
- **Docker**: Container orchestration
- **R 4.4.1**: Statistical computing
- **RMarkdown**: Literate programming

## Security Considerations

1. **API Key Management**: Environment variables or Streamlit secrets
2. **Container Isolation**: Each execution in fresh container
3. **File Access**: Limited to mounted volumes
4. **Code Validation**: Audit agent checks before execution
5. **Resource Limits**: Docker memory constraints

## Scalability

**Current**: Single-user, local execution
**Future Enhancements**:
- Multi-user support with queue system
- Cloud deployment (AWS, GCP, Azure)
- Kubernetes orchestration
- Persistent storage solutions
- Load balancing for concurrent requests

## Error Handling

Each agent includes:
- Try-catch blocks for API failures
- Timeout handling (30s default)
- Graceful degradation
- User-friendly error messages
- Logging for debugging

