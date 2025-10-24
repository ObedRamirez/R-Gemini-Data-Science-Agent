# Docker Build Instructions for R Data Science Image

## Overview
This consolidated Dockerfile combines all incremental builds (V1.0 - V1.4) into a single, optimized image for AI agents with R statistics and ML capabilities.

## Build Commands

### 1. Basic Build
```bash
docker build -f Dockerfile.consolidated -t r_data_science:v2.0 .
```

### 2. Build with Build Cache Disabled (for clean build)
```bash
docker build --no-cache -f Dockerfile.consolidated -t r_data_science:v2.0 .
```

### 3. Build with Progress Output
```bash
docker build --progress=plain -f Dockerfile.consolidated -t r_data_science:v2.0 .
```

## Run Commands

### Start Interactive R Session
```bash
docker run -it --rm r_data_science:v2.0
```

### Mount Local Directory
```bash
docker run -it --rm -v $(pwd):/app r_data_science:v2.0
```

### Run Specific R Script
```bash
docker run --rm -v $(pwd):/app r_data_science:v2.0 Rscript /app/your_script.R
```

### Run RMarkdown Report
```bash
docker run --rm -v $(pwd):/app r_data_science:v2.0 Rscript -e "rmarkdown::render('/app/report.Rmd')"
```

## Package Information

### Key ML Packages (Pinned Versions)
- caret @ 6.0-94
- randomForest @ 4.7-1.2
- xgboost @ 1.7.8.1
- glmnet @ 4.1-8

### Total Packages Explicitly Installed
65 top-level packages (dependencies handled automatically by R)

### Full Package List with Versions
See r_requirements.txt for complete list of all 463 installed packages including dependencies.

## Build Optimizations

1. **Layered Installation**: Packages grouped by category for better caching
2. **Cleanup**: Each RUN command removes downloaded packages to reduce image size
3. **Pinned ML Versions**: Critical ML packages locked to current versions
4. **System Dependencies**: All necessary libraries pre-installed
5. **.dockerignore**: Excludes unnecessary files from build context

## Troubleshooting

### Build Fails on Package Installation
- Check internet connection
- Try building with `--network=host` flag
- Use a CRAN mirror: Add `options(repos = c(CRAN = "https://cloud.r-project.org"))` before install commands

### Out of Memory During Build
- Increase Docker memory allocation (Docker Desktop -> Settings -> Resources)
- Build packages in smaller groups

### Package Version Conflicts
- The consolidated build should resolve all conflicts automatically
- If issues persist, check r_requirements.txt for working versions

