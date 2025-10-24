# Installation Guide

## Prerequisites

- Python 3.8 or higher
- Docker Desktop (with 8GB+ RAM allocated)
- Git
- Google Gemini API key

## Step-by-Step Installation

### 1. Clone Repository

git clone https://github.com/YOUR_USERNAME/dynamic-r-analyst.git
cd dynamic-r-analyst

text

### 2. Create Virtual Environment

python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

text

### 3. Install Python Dependencies

pip install -r requirements.txt

text

### 4. Build Docker Image

cd docker
docker build -f Dockerfile.consolidated -t r_data_science:v2.0 .
cd ..

text

Build time: 15-20 minutes. This creates an R environment with 65+ packages.

### 5. Configure API Key

**Option A: Environment Variable**

export GOOGLE_API_KEY="your_key_here"

text

**Option B: .env File**

echo "GOOGLE_API_KEY=your_key_here" > .env

text

**Option C: Streamlit Secrets** (Recommended for deployment)

mkdir -p .streamlit
echo '[google]' > .streamlit/secrets.toml
echo 'api_key = "your_key_here"' >> .streamlit/secrets.toml

text

### 6. Verify Installation

Test Python imports

python -c "from src.dynamic_r_analyst_v2 import *; print('Success!')"
Check Docker image

docker images | grep r_data_science
Run application

streamlit run src/dynamic_r_analyst_v2.py

text

## Troubleshooting

### Docker Build Fails

- Increase Docker memory: Docker Desktop → Settings → Resources → 8GB
- Use alternative CRAN mirror: Add `--build-arg CRAN_MIRROR=https://cran.rstudio.com`

### Import Errors

pip install --upgrade -r requirements.txt

text

### API Key Not Found

Check environment variable:

echo $GOOGLE_API_KEY # Linux/Mac
echo %GOOGLE_API_KEY% # Windows
