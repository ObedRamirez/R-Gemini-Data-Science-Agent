# Docker Configuration

## Overview

The R data science Docker image provides a complete statistical computing environment with 65 explicitly installed packages and 400+ dependencies.

## Image Details

**Name**: `r_data_science:v2.0`
**Base**: `rocker/tidyverse:4.4.1`
**Size**: ~2.5GB
**Build Time**: 15-20 minutes

## Package Categories

### Machine Learning (Pinned Versions)
- caret @ 6.0-94
- randomForest @ 4.7-1.2
- xgboost @ 1.7.8.1
- glmnet @ 4.1-8
- e1071
- factoextra

### Data Processing
- data.table, dplyr, tidyr
- arrow, fst, qs
- lubridate, stringr

### Visualization
- ggplot2, plotly, gganimate
- ggthemes, ggrepel, patchwork
- corrplot, pheatmap, viridis

### Development Tools
- devtools, usethis, testthat
- roxygen2, profvis

## Build Instructions

### Standard Build

docker build -f Dockerfile.consolidated -t r_data_science:v2.0 .

text

### With Progress Output

docker build --progress=plain -f Dockerfile.consolidated -t r_data_science:v2.0 .

text

### Clean Build (No Cache)

docker build --no-cache -f Dockerfile.consolidated -t r_data_science:v2.0 .

text

## Usage

### Interactive R Session

docker run -it --rm r_data_science:v2.0

text

### Execute Script

docker run --rm -v $(pwd):/app r_data_science:v2.0 Rscript /app/script.R

text

### Render RMarkdown

docker run --rm -v $(pwd):/app r_data_science:v2.0 Rscript -e "rmarkdown::render('/app/report.Rmd')"

text

## Package Reference

See `r_requirements.txt` for complete list with versions.

## Maintenance

### Update Base Image
Change `FROM rocker/tidyverse:4.4.1` to newer version

### Add Package
Add to appropriate RUN command in Dockerfile.consolidated

### Rebuild After Changes

docker build --no-cache -f Dockerfile.consolidated -t r_data_science:v2.0 .
