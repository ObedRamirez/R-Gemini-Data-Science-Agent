#!/usr/bin/env python3
"""
R Data Science Coding Agent
============================

An independent expert R data science agent that generates comprehensive RMarkdown documents
for statistical analysis, machine learning, and data visualization tasks.

Usage:
    As a module:
        from r_coding_agent import ExpertRDataScientist, create_r_agent
        
        agent = create_r_agent()
        result = agent.analyze("Analyze customer behavior patterns")
        
    As a standalone script:
        python r_coding_agent.py
        
    With command line arguments:
        python r_coding_agent.py --request "Build a regression model" --type modeling --save

Author: Your Name
Version: 1.0.0
"""

import re
import os
import sys
import argparse
from typing import Dict, List, Optional, Union
from datetime import datetime
import json
import logging

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.prompts import ChatPromptTemplate
    from langchain.chains import LLMChain
    from langchain.schema import BaseOutputParser
    from langchain.output_parsers import PydanticOutputParser
    from pydantic import BaseModel, Field
except ImportError as e:
    print(f"Missing required dependencies: {e}")
    print("Install with: pip install langchain langchain-google-genai pydantic")
    sys.exit(1)

# Configuration
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_TEMPERATURE = 0.2

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RDataScienceResult(BaseModel):
    """Model for R Data Science analysis results"""
    rmarkdown_code: str = Field(description="Complete RMarkdown document with analysis")
    analysis_type: str = Field(description="Type of analysis performed")
    packages_used: List[str] = Field(description="R packages required for the analysis")
    key_insights: List[str] = Field(description="Key insights from the analysis approach")
    complexity_level: str = Field(description="Analysis complexity: beginner, intermediate, advanced")
    estimated_runtime: str = Field(description="Estimated runtime for the analysis")


class ExpertRDataScientist:
    """
    Expert R Data Science Agent using LangChain and Gemini
    Specializes in creating comprehensive RMarkdown documents for data science analysis
    """
    
    def __init__(self, model_name: str = DEFAULT_MODEL, temperature: float = DEFAULT_TEMPERATURE, google_api_key: str = None):
        """
        Initialize the Expert R Data Science Agent
        
        Args:
            model_name: Gemini model to use
            temperature: Temperature for code generation (slightly higher for creativity)
            google_api_key: Google API key
        """
        if google_api_key:
            os.environ["GOOGLE_API_KEY"] = google_api_key
        
        # Validate API key
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("Google API key must be provided or set in GOOGLE_API_KEY environment variable")
        
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            convert_system_message_to_human=True
        )
        self.output_parser = PydanticOutputParser(pydantic_object=RDataScienceResult)
        self._setup_prompts()
        
        # R Data Science knowledge base
        self.r_packages = {
            "data_manipulation": ["dplyr", "tidyr", "data.table", "purrr"],
            "visualization": ["ggplot2", "plotly", "lattice", "corrplot", "pheatmap"],
            "statistics": ["stats", "broom", "car", "lmtest", "nortest"],
            "machine_learning": ["caret", "randomForest", "e1071", "glmnet", "xgboost"],
            "time_series": ["forecast", "tseries", "zoo", "xts", "lubridate"],
            "text_analysis": ["tm", "tidytext", "wordcloud", "quanteda"],
            "reporting": ["knitr", "rmarkdown", "DT", "formattable", "kableExtra"],
            "database": ["DBI", "RSQLite", "RMySQL", "RPostgreSQL"],
            "web_scraping": ["rvest", "httr", "jsonlite"],
            "spatial": ["sf", "sp", "leaflet", "tmap"]
        }
        
        logger.info(f"R Data Science Agent initialized with model: {model_name}")
        
    def _setup_prompts(self):
        """Setup specialized prompts for R data science analysis"""
        
        self.main_prompt = ChatPromptTemplate.from_messages([
            ("human", """You are an EXPERT R DATA SCIENTIST with 15+ years of experience in statistical analysis, machine learning, and data visualization. You are known for creating comprehensive, publication-ready RMarkdown documents that follow best practices.

CORE EXPERTISE:
- Advanced statistical modeling and hypothesis testing
- Machine learning algorithms and model validation
- Data visualization and storytelling with data
- Reproducible research and literate programming
- Performance optimization and efficient R coding
- Modern R ecosystem (tidyverse, tidymodels, etc.)

R CODING PRINCIPLES YOU FOLLOW:
1. Always use tidyverse approach when possible
2. Write clean, readable, and well-documented code
3. Include comprehensive error handling and data validation
4. Use modern R packages and functions
5. Follow R style guides (snake_case, proper spacing)
6. Include informative plots with professional aesthetics
7. Provide statistical interpretations and business insights
8. Use efficient data manipulation techniques
9. Include reproducibility elements (set.seed, sessionInfo)
10. Structure code in logical, easy-to-follow sections

RMARKDOWN DOCUMENT STRUCTURE YOU CREATE:
1. **YAML Header**: Complete with title, author, date, output options
2. **Setup Chunk**: Library loading, global options, custom functions
3. **Data Import & Exploration**: Data loading, structure, summary statistics
4. **Data Cleaning & Preprocessing**: Missing values, outliers, transformations
5. **Exploratory Data Analysis**: Visualizations, correlations, patterns
6. **Statistical Analysis/Modeling**: Appropriate methods with validation
7. **Results & Interpretation**: Clear explanations of findings
8. **Conclusions & Recommendations**: Actionable insights
9. **Appendix**: Session info, additional details

ANALYSIS REQUEST: {analysis_request}

CREATE A COMPLETE RMARKDOWN DOCUMENT that addresses this request. Include:
- Professional YAML header
- Comprehensive analysis workflow
- Modern R code with best practices
- Informative visualizations
- Statistical rigor and proper interpretations
- Clear markdown explanations between code chunks
- Reproducible and well-documented approach

Focus on creating a document that could be used in a professional data science environment.

Return your response in the following JSON format:
{format_instructions}""")
        ])
        
        # Specialized prompts for different analysis types
        self.eda_prompt = ChatPromptTemplate.from_messages([
            ("human", """You are creating an EXPLORATORY DATA ANALYSIS (EDA) RMarkdown document. Focus on:

SPECIFIC EDA REQUIREMENTS:
- Comprehensive data profiling and quality assessment
- Distribution analysis for all variables
- Correlation analysis and multicollinearity detection
- Outlier detection and treatment strategies
- Missing data patterns and imputation recommendations
- Feature engineering opportunities
- Visual storytelling with professional plots
- Statistical summaries and insights

VISUALIZATION REQUIREMENTS:
- Use ggplot2 with modern aesthetics
- Include interactive plots where appropriate (plotly)
- Professional color schemes and themes
- Multiple visualization types (histograms, boxplots, scatter plots, etc.)
- Correlation matrices and heatmaps
- Statistical diagnostic plots

Analysis Request: {analysis_request}

Create a comprehensive EDA RMarkdown document.

{format_instructions}""")
        ])
        
        self.modeling_prompt = ChatPromptTemplate.from_messages([
            ("human", """You are creating a STATISTICAL MODELING / MACHINE LEARNING RMarkdown document. Focus on:

MODELING REQUIREMENTS:
- Proper train/validation/test splits
- Feature selection and engineering
- Model selection with cross-validation
- Hyperparameter tuning
- Model performance evaluation
- Statistical assumptions checking
- Residual analysis and diagnostics
- Model interpretability and explanations
- Business impact and recommendations

STATISTICAL RIGOR:
- Appropriate statistical tests
- Effect size calculations
- Confidence intervals
- Multiple comparison corrections where needed
- Assumption testing and validation
- Robust statistical practices

Analysis Request: {analysis_request}

Create a rigorous modeling RMarkdown document with statistical best practices.

{format_instructions}""")
        ])
        
        self.visualization_prompt = ChatPromptTemplate.from_messages([
            ("human", """You are creating a DATA VISUALIZATION focused RMarkdown document. Focus on:

VISUALIZATION EXCELLENCE:
- Publication-quality plots with ggplot2
- Interactive visualizations with plotly/shiny
- Dashboard-style layouts
- Color theory and accessibility
- Statistical graphics and charts
- Animated plots where appropriate
- Custom themes and styling
- Storytelling through visualization

ADVANCED TECHNIQUES:
- Multi-panel plots with faceting
- Complex statistical visualizations
- Geographic and spatial plots
- Network and relationship visualizations
- Time series and temporal plots
- Custom visualization functions

Analysis Request: {analysis_request}

Create a visualization-rich RMarkdown document with stunning, informative plots.

{format_instructions}""")
        ])
    
    def _detect_analysis_type(self, request: str) -> str:
        """Detect the type of analysis requested"""
        request_lower = request.lower()
        
        # Keywords for different analysis types
        modeling_keywords = ["model", "predict", "machine learning", "regression", "classification", "clustering", "forecast"]
        viz_keywords = ["visualiz", "plot", "chart", "graph", "dashboard", "interactive"]
        eda_keywords = ["explore", "eda", "exploratory", "summary", "describe", "profile"]
        
        modeling_score = sum(1 for keyword in modeling_keywords if keyword in request_lower)
        viz_score = sum(1 for keyword in viz_keywords if keyword in request_lower)
        eda_score = sum(1 for keyword in eda_keywords if keyword in request_lower)
        
        if modeling_score > max(viz_score, eda_score):
            return "modeling"
        elif viz_score > max(modeling_score, eda_score):
            return "visualization"
        elif eda_score > 0:
            return "eda"
        else:
            return "general"
    
    def _recommend_packages(self, request: str, analysis_type: str) -> List[str]:
        """Recommend R packages based on analysis request"""
        recommended = ["dplyr", "ggplot2", "tidyr", "knitr", "rmarkdown"]  # Core packages
        
        request_lower = request.lower()
        
        # Add packages based on keywords in request
        if any(word in request_lower for word in ["time", "date", "temporal", "forecast"]):
            recommended.extend(self.r_packages["time_series"])
        
        if any(word in request_lower for word in ["text", "sentiment", "nlp", "word"]):
            recommended.extend(self.r_packages["text_analysis"])
        
        if any(word in request_lower for word in ["machine learning", "ml", "predict", "model"]):
            recommended.extend(self.r_packages["machine_learning"])
        
        if any(word in request_lower for word in ["map", "spatial", "geographic", "location"]):
            recommended.extend(self.r_packages["spatial"])
        
        if analysis_type == "modeling":
            recommended.extend(self.r_packages["statistics"] + self.r_packages["machine_learning"][:3])
        elif analysis_type == "visualization":
            recommended.extend(self.r_packages["visualization"])
        elif analysis_type == "eda":
            recommended.extend(self.r_packages["data_manipulation"] + self.r_packages["visualization"][:3])
        
        # Remove duplicates and return
        return list(set(recommended))
    
    def analyze(self, analysis_request: str, analysis_type: str = "auto") -> RDataScienceResult:
        """
        Main method to generate R data science analysis
        
        Args:
            analysis_request: Description of the analysis needed
            analysis_type: Type of analysis ("auto", "eda", "modeling", "visualization", "general")
            
        Returns:
            RDataScienceResult with complete RMarkdown document
        """
        try:
            logger.info(f"Starting analysis: {analysis_request[:100]}...")
            
            # Detect analysis type if auto
            if analysis_type == "auto":
                analysis_type = self._detect_analysis_type(analysis_request)
                logger.info(f"Detected analysis type: {analysis_type}")
            
            # Choose appropriate prompt
            if analysis_type == "eda":
                chain = LLMChain(llm=self.llm, prompt=self.eda_prompt)
                result = chain.run(
                    analysis_request=analysis_request,
                    format_instructions=self.output_parser.get_format_instructions()
                )
            elif analysis_type == "modeling":
                chain = LLMChain(llm=self.llm, prompt=self.modeling_prompt)
                result = chain.run(
                    analysis_request=analysis_request,
                    format_instructions=self.output_parser.get_format_instructions()
                )
            elif analysis_type == "visualization":
                chain = LLMChain(llm=self.llm, prompt=self.visualization_prompt)
                result = chain.run(
                    analysis_request=analysis_request,
                    format_instructions=self.output_parser.get_format_instructions()
                )
            else:  # general
                chain = LLMChain(llm=self.llm, prompt=self.main_prompt)
                result = chain.run(
                    analysis_request=analysis_request,
                    format_instructions=self.output_parser.get_format_instructions()
                )
            
            # Parse the result
            try:
                ds_result = self.output_parser.parse(result)
            except Exception as parse_error:
                logger.warning(f"Failed to parse structured output: {parse_error}")
                # Fallback: create basic result
                ds_result = RDataScienceResult(
                    rmarkdown_code=self._extract_rmarkdown_from_response(result),
                    analysis_type=analysis_type,
                    packages_used=self._recommend_packages(analysis_request, analysis_type),
                    key_insights=["Generated RMarkdown document for data science analysis"],
                    complexity_level="intermediate",
                    estimated_runtime="5-15 minutes"
                )
            
            # Enhance the result
            ds_result.analysis_type = analysis_type
            if not ds_result.packages_used:
                ds_result.packages_used = self._recommend_packages(analysis_request, analysis_type)
            
            logger.info(f"Analysis generated successfully. Type: {ds_result.analysis_type}, "
                       f"Complexity: {ds_result.complexity_level}")
            
            return ds_result
            
        except Exception as e:
            logger.error(f"Error during analysis generation: {str(e)}")
            # Return minimal result in case of error
            return RDataScienceResult(
                rmarkdown_code=self._create_error_rmarkdown(analysis_request, str(e)),
                analysis_type="error",
                packages_used=["base"],
                key_insights=[f"Error occurred: {str(e)}"],
                complexity_level="unknown",
                estimated_runtime="unknown"
            )
    
    def _extract_rmarkdown_from_response(self, response: str) -> str:
        """Extract RMarkdown content from response if JSON parsing fails"""
        # Try to find RMarkdown code blocks
        patterns = [
            r'```{?rmarkdown}?\n(.*?)```',
            r'```{?r}?\n(.*?)```',
            r'```\n(.*?)```',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                return matches[0].strip()
        
        return response.strip()
    
    def _create_error_rmarkdown(self, request: str, error: str) -> str:
        """Create a basic RMarkdown document when errors occur"""
        return f'''---
title: "Data Science Analysis"
author: "R Data Science Agent"
date: "{datetime.now().strftime('%Y-%m-%d')}"
output: html_document
---

## Analysis Request
{request}

## Error Occurred
An error occurred during analysis generation:
```
{error}
```

## Basic R Setup
```{{r setup}}
library(dplyr)
library(ggplot2)
library(knitr)

# Your analysis code would go here
print("Please review the analysis request and try again.")
```
'''
    
    def save_rmarkdown(self, result: RDataScienceResult, filename: str = None) -> str:
        """Save the RMarkdown document to a file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analysis_{timestamp}.Rmd"
        
        if not filename.endswith('.Rmd'):
            filename += '.Rmd'
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(result.rmarkdown_code)
            
            logger.info(f"RMarkdown document saved to: {filename}")
            return filename
        except Exception as e:
            logger.error(f"Error saving file: {e}")
            raise
    
    def quick_analysis(self, request: str) -> str:
        """Quick method that returns only the RMarkdown code"""
        result = self.analyze(request)
        return result.rmarkdown_code
    
    def get_analysis_summary(self, result: RDataScienceResult) -> Dict:
        """Get a summary of the analysis"""
        return {
            "analysis_type": result.analysis_type,
            "complexity_level": result.complexity_level,
            "estimated_runtime": result.estimated_runtime,
            "packages_count": len(result.packages_used),
            "packages_used": result.packages_used,
            "key_insights": result.key_insights,
            "document_length": len(result.rmarkdown_code),
            "code_chunks": result.rmarkdown_code.count('```{r'),
            "markdown_sections": result.rmarkdown_code.count('#')
        }


# Utility class for specialized analysis types
class RDataScienceTemplates:
    """Pre-built templates for common data science analyses"""
    
    @staticmethod
    def get_eda_template() -> str:
        return "Create a comprehensive exploratory data analysis with data profiling, visualizations, correlation analysis, and insights"
    
    @staticmethod
    def get_modeling_template(model_type: str) -> str:
        templates = {
            "regression": "Build a regression model with feature selection, validation, and interpretation",
            "classification": "Create a classification model with cross-validation and performance metrics",
            "clustering": "Perform clustering analysis with optimal cluster selection and validation",
            "time_series": "Conduct time series analysis with forecasting and trend analysis"
        }
        return templates.get(model_type, "Build an appropriate statistical model for the data")
    
    @staticmethod
    def get_visualization_template() -> str:
        return "Create a comprehensive data visualization report with multiple chart types and interactive elements"


# Factory functions for easy creation
def create_r_agent(google_api_key: str = None, model_name: str = DEFAULT_MODEL, temperature: float = DEFAULT_TEMPERATURE) -> ExpertRDataScientist:
    """
    Create an ExpertRDataScientist instance
    
    Args:
        google_api_key: Google API key (if None, will try to get from environment)
        model_name: Gemini model to use
        temperature: Temperature for code generation
        
    Returns:
        ExpertRDataScientist instance
    """
    if google_api_key is None:
        google_api_key = os.getenv("GOOGLE_API_KEY")
    
    return ExpertRDataScientist(
        model_name=model_name,
        temperature=temperature,
        google_api_key=google_api_key
    )


def interactive_mode():
    """Run the agent in interactive mode"""
    print("=== Expert R Data Science Agent (Interactive Mode) ===")
    print("Type 'quit', 'exit', or 'q' to stop")
    print("Type 'templates' to see available templates")
    print("Analysis types: auto, eda, modeling, visualization, general")
    print()
    
    try:
        agent = create_r_agent()
        print("R Data Science Agent initialized successfully!\n")
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    while True:
        try:
            request = input("Enter your analysis request: ").strip()
            
            if request.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            elif request.lower() == 'templates':
                print("\nAvailable Templates:")
                print("- EDA:", RDataScienceTemplates.get_eda_template())
                print("- Regression:", RDataScienceTemplates.get_modeling_template("regression"))
                print("- Classification:", RDataScienceTemplates.get_modeling_template("classification"))
                print("- Visualization:", RDataScienceTemplates.get_visualization_template())
                print()
                continue
            elif not request:
                continue
            
            analysis_type = input("Analysis type (auto/eda/modeling/visualization/general) [auto]: ").strip()
            if not analysis_type:
                analysis_type = "auto"
            
            save_file = input("Save to file? (y/n) [y]: ").strip().lower()
            if not save_file:
                save_file = "y"
            
            print(f"\nGenerating {analysis_type} analysis...")
            result = agent.analyze(request, analysis_type)
            
            # Display summary
            summary = agent.get_analysis_summary(result)
            print(f"\nAnalysis Summary:")
            print(f"- Type: {summary['analysis_type']}")
            print(f"- Complexity: {summary['complexity_level']}")
            print(f"- Packages: {summary['packages_count']} ({', '.join(summary['packages_used'][:5])}...)")
            print(f"- Code chunks: {summary['code_chunks']}")
            print(f"- Document length: {summary['document_length']} characters")
            
            if save_file == 'y':
                filename = agent.save_rmarkdown(result)
                print(f"✅ RMarkdown saved to: {filename}")
            else:
                print(f"\nRMarkdown Code Preview (first 500 chars):")
                print("-" * 50)
                print(result.rmarkdown_code[:500] + "..." if len(result.rmarkdown_code) > 500 else result.rmarkdown_code)
            
            print("\n" + "="*80 + "\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(
        description="Expert R Data Science Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python r_coding_agent.py --interactive
  python r_coding_agent.py --request "Analyze customer churn patterns" --type eda --save
  python r_coding_agent.py --request "Build a regression model" --type modeling --output results.Rmd
  python r_coding_agent.py --request "Create visualizations" --type visualization --model gemini-1.5-pro
        """
    )
    
    parser.add_argument(
        "--request", "-r",
        type=str,
        help="Analysis request description"
    )
    
    parser.add_argument(
        "--type", "-t",
        type=str,
        choices=["auto", "eda", "modeling", "visualization", "general"],
        default="auto",
        help="Analysis type (default: auto)"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Gemini model to use (default: {DEFAULT_MODEL})"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Temperature for code generation (default: {DEFAULT_TEMPERATURE})"
    )
    
    parser.add_argument(
        "--save", "-s",
        action="store_true",
        help="Save RMarkdown to file with auto-generated name"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file for the RMarkdown document"
    )
    
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Show only analysis summary, not the full RMarkdown code"
    )
    
    parser.add_argument(
        "--template",
        type=str,
        choices=["eda", "regression", "classification", "clustering", "visualization"],
        help="Use a pre-built template"
    )
    
    args = parser.parse_args()
    
    # Interactive mode
    if args.interactive:
        interactive_mode()
        return
    
    # Template mode
    if args.template:
        if args.template == "eda":
            args.request = RDataScienceTemplates.get_eda_template()
            args.type = "eda"
        elif args.template == "visualization":
            args.request = RDataScienceTemplates.get_visualization_template()
            args.type = "visualization"
        else:
            args.request = RDataScienceTemplates.get_modeling_template(args.template)
            args.type = "modeling"
        
        print(f"Using {args.template} template: {args.request}")
    
    # Single request mode
    if not args.request:
        parser.print_help()
        return
    
    try:
        # Create agent
        agent = create_r_agent(model_name=args.model, temperature=args.temperature)
        
        # Generate analysis
        print(f"Generating {args.type} analysis with {args.model}...")
        result = agent.analyze(args.request, args.type)
        
        # Display results
        summary = agent.get_analysis_summary(result)
        print(f"\nAnalysis Summary:")
        print(f"- Type: {summary['analysis_type']}")
        print(f"- Complexity: {summary['complexity_level']}")
        print(f"- Runtime: {result.estimated_runtime}")
        print(f"- Packages: {summary['packages_count']} ({', '.join(summary['packages_used'][:5])}...)")
        print(f"- Code chunks: {summary['code_chunks']}")
        print(f"- Key insights: {len(result.key_insights)}")
        
        if not args.summary_only:
            print(f"\nRMarkdown Document:")
            print("=" * 80)
            print(result.rmarkdown_code)
            print("=" * 80)
        
        # Save to file if requested
        filename = None
        if args.save:
            filename = agent.save_rmarkdown(result)
        elif args.output:
            filename = agent.save_rmarkdown(result, args.output)
        
        if filename:
            print(f"\n✅ RMarkdown document saved to: {filename}")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()