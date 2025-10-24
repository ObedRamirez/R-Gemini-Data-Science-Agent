# Dynamic R Analyst - Multi-Agent Data Science System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)](https://www.docker.com/)

A sophisticated multi-agent system that leverages LangChain and Google's Generative AI to orchestrate automated R data science workflows. The system translates natural language requests into production-ready R code, audits it for quality, and executes analyses in isolated Docker containers.

## 🌟 Features

- **Natural Language Interface**: Describe your analysis needs in plain English
- **Multi-Agent Architecture**: Specialized agents for prompt translation, code generation, and quality auditing
- **R Code Generation**: Produces comprehensive RMarkdown documents with statistical analysis and visualizations
- **Automated Quality Control**: Built-in code auditor catches errors and suggests improvements
- **Docker Integration**: Executes R code in isolated containers with 65+ pre-installed packages
- **Interactive UI**: Streamlit-based web interface for easy interaction
- **Comprehensive ML Support**: Pre-configured with caret, xgboost, randomForest, and statistical packages

## 🏗️ System Architecture

The system consists of four specialized agents:

1. **Prompt Agent** (`data_science_prompt_agent_v2.py`): Translates user requests into detailed R programming specifications
2. **Coding Agent** (`data_science_coding_agent_v2.py`): Generates complete RMarkdown documents with analysis code
3. **Audit Agent** (`data_science_audit_agent_v2.py`): Reviews and corrects R code for syntax errors and best practices
4. **Master Agent** (`dynamic_r_analyst_v2.py`): Orchestrates the workflow and provides the Streamlit interface

## 📋 Prerequisites

- Python 3.8 or higher
- Docker Desktop installed and running
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))
- 4GB+ RAM available for Docker containers
- Git (for cloning the repository)

## 🚀 Quick Start

### 1. Clone the Repository

git clone https://github.com/YOUR_USERNAME/dynamic-r-analyst.git
cd dynamic-r-analyst

text

### 2. Install Python Dependencies

pip install -r requirements.txt

text

### 3. Build the R Docker Image

cd docker
docker build -f Dockerfile.consolidated -t r_data_science:v2.0 .
cd ..

text

This will take 15-20 minutes to build a comprehensive R environment with 65+ packages.

### 4. Set Up API Key

Create a `.env` file in the root directory:

echo "GOOGLE_API_KEY=your_api_key_here" > .env

text

Or set it as an environment variable:

export GOOGLE_API_KEY="your_api_key_here"

text

### 5. Run the Application

streamlit run src/dynamic_r_analyst_v2.py

text

The app will open in your browser at `http://localhost:8501`

## 💡 Usage Examples

### Example 1: Exploratory Data Analysis

Upload your CSV file and enter:
"Perform exploratory data analysis on this dataset.
Include summary statistics, distribution plots, and correlation analysis."

text

### Example 2: Predictive Modeling

"Build a random forest model to predict the target variable.
Include feature importance analysis and model evaluation metrics."

text

### Example 3: Time Series Analysis

"Analyze the time series trends in the data.
Include decomposition, stationarity tests, and forecasting."

text

## 📁 Project Structure

dynamic-r-analyst/

├── README.md # This file

├── LICENSE # MIT License

├── .gitignore # Git ignore patterns

├── requirements.txt # Python dependencies

│
├── src/ # Main source code

│ ├── init.py

│ ├── data_science_audit_agent_v2.py # Code auditor

│ ├── data_science_prompt_agent_v2.py # Prompt translator

│ ├── data_science_coding_agent_v2.py # R code generator

│ └── dynamic_r_analyst_v2.py # Streamlit app

│

├── docker/ # Docker configuration

│ ├── Dockerfile.consolidated # R environment (recommended)

│ ├── Dockerfile # Python/Streamlit environment

│ ├── .dockerignore # Docker ignore patterns

│ ├── r_requirements.txt # Full R package list

│ └── README.md # Docker setup guide

│

├── docs/ # Documentation

│ ├── INSTALLATION.md # Detailed setup instructions

│ ├── USAGE.md # Usage guide and examples

│ └── ARCHITECTURE.md # System architecture details

│
├── examples/ # Example files

│ └── sample_analysis.csv

│
└── scripts/ # Utility scripts

├── build_docker.sh # Docker build automation
└── run_app.sh # App launcher script



## 🔧 Configuration

### Docker Configuration

The R Docker image includes:
- **Base Image**: rocker/tidyverse:4.4.1
- **ML Packages**: caret, xgboost, randomForest, glmnet (version-pinned)
- **Visualization**: ggplot2, plotly, gganimate, patchwork
- **Data Processing**: data.table, arrow, dplyr, tidyr
- **Development Tools**: devtools, testthat, profvis
- **Total**: 65 explicitly installed packages + 400+ dependencies

See `docker/README.md` for detailed package information.

### Python Configuration

The system uses:
- **LangChain**: For agent orchestration
- **Google Generative AI**: Gemini models for code generation
- **Streamlit**: Web interface
- **Pandas**: Data handling
- **Pydantic**: Data validation

## 🐛 Troubleshooting

### Docker Build Issues

If the Docker build fails:

Use a specific CRAN mirror

docker build --build-arg CRAN_MIRROR=https://cloud.r-project.org -f docker/Dockerfile.consolidated -t r_data_science:v2.0 .

text

### API Key Issues

Verify your API key is set:

python -c "import os; print('API Key:', os.getenv('GOOGLE_API_KEY'))"

text

### Memory Issues

Increase Docker memory allocation:
- Docker Desktop → Settings → Resources → Memory → 8GB+

### R Execution Errors

Check Docker container logs:

docker logs <container_id>

text

## 📚 Documentation

- [Installation Guide](docs/INSTALLATION.md) - Detailed setup instructions
- [Usage Guide](docs/USAGE.md) - Comprehensive usage examples
- [Architecture Guide](docs/ARCHITECTURE.md) - System design and components
- [Docker Setup](docker/README.md) - Docker configuration details

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [LangChain](https://www.langchain.com/)
- Powered by [Google Gemini](https://deepmind.google/technologies/gemini/)
- R environment based on [Rocker Project](https://rocker-project.org/)
- UI built with [Streamlit](https://streamlit.io/)

## 📧 Contact

Dr. Obed Ramirez - obed_eo@hotmail.com

Project Link: [https://github.com/YOUR_USERNAME/dynamic-r-analyst](https://github.com/YOUR_USERNAME/dynamic-r-analyst)

## 🔗 Related Projects

- [LangChain](https://github.com/langchain-ai/langchain)
- [Rocker](https://github.com/rocker-org/rocker)
- [Streamlit](https://github.com/streamlit/streamlit)

---

⭐ Star this repo if you find it helpful!
