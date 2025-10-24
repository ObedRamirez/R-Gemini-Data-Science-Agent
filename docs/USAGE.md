# Usage Guide

## Starting the Application

streamlit run src/dynamic_r_analyst_v2.py

text

Open browser at `http://localhost:8501`

## Basic Workflow

### 1. Upload Data

- Click "Browse files" 
- Select CSV file
- Data is automatically saved as `data.csv` in the container

### 2. Enter Analysis Request

Examples:
- "Perform exploratory data analysis with summary statistics and visualizations"
- "Build a predictive model using random forest to predict [target_variable]"
- "Analyze correlations and create a heatmap"

### 3. Review Generated Code

The system generates:
- Prompt specifications (what to analyze)
- Complete RMarkdown code
- Audited, error-free code

### 4. Execute Analysis

Click "Run Analysis" to execute in Docker container.

### 5. Download Results

Download the HTML report with:
- Data summaries
- Visualizations
- Statistical analysis
- Model results (if applicable)

## Advanced Usage

### Custom Analysis Types

**Exploratory Data Analysis**

Analyze the distribution of variables, identify outliers,
create correlation matrix, and provide summary statistics
grouped by categorical variables.

text

**Predictive Modeling**

Build and compare multiple models (random forest, xgboost, glmnet)
to predict [target]. Include feature importance, cross-validation,
and evaluation metrics (RMSE, R², confusion matrix).

text

**Time Series Analysis**

Decompose the time series, test for stationarity,
create ACF/PACF plots, and fit ARIMA model with forecasts.

text

**Clustering Analysis**

Perform k-means and hierarchical clustering.
Determine optimal clusters using elbow method and silhouette analysis.
Visualize clusters with PCA.

text

### Working with Results

Results are saved as HTML reports containing:
- Embedded visualizations
- Statistical tables
- Model summaries
- Reproducible R code

### Tips for Best Results

1. **Be specific**: Include variable names and analysis details
2. **Mention data types**: Specify if variables are categorical/continuous
3. **State objectives**: Classification, regression, exploration, etc.
4. **Request specific plots**: Histograms, scatter plots, heatmaps
5. **Ask for metrics**: Accuracy, RMSE, AUC, p-values

## Command-Line Usage

### Run Agents Independently

**Prompt Agent**

python src/data_science_prompt_agent_v2.py --request "analyze customer churn"

text

**Coding Agent**

python src/data_science_coding_agent_v2.py --request "build regression model" --type modeling

text

**Audit Agent**

python src/data_science_audit_agent_v2.py --file script.R

text

## Docker Management

**List containers**

docker ps -a

text

**Clean up stopped containers**

docker container prune

text

**Remove old images**

docker image prune -a
