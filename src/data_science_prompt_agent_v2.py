#!/usr/bin/env python3
"""
Data Science Prompt Translator Tool
====================================

An independent LangChain tool for translating data analysis requests into detailed R programming prompts.

Usage:
    As a module:
        from data_science_tool import DataSciencePromptTool, create_data_science_tool
        
        tool = create_data_science_tool()
        result = tool._run("analyze customer churn", "telecom dataset")

    As a standalone script:
        python data_science_tool.py

    With command line arguments:
        python data_science_tool.py --request "predict house prices" --context "real estate data"

Author: Your Name
Version: 1.0.0
"""

import os
import sys
import argparse
from typing import Dict, List, Optional, Any, Type
from datetime import datetime
import json

try:
    # LangChain imports
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.memory import ConversationBufferWindowMemory
    from langchain.schema import BaseMessage, HumanMessage, AIMessage
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from langchain.tools import BaseTool
    from langchain.callbacks.base import BaseCallbackHandler
    
    # Pydantic for data validation
    from pydantic import BaseModel, Field
    
except ImportError as e:
    print(f"Missing required dependencies: {e}")
    print("Install with: pip install langchain langchain-google-genai pydantic")
    sys.exit(1)


# Configuration
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MEMORY_K = 5


class DataAnalysisToolInput(BaseModel):
    """Input schema for the DataSciencePromptTool"""
    request: str = Field(description="The user's data analysis request")
    context: Optional[str] = Field(default="", description="Additional context or background information")


class DataAnalysisRequest(BaseModel):
    """Structure for data analysis requests"""
    request_text: str = Field(description="The user's data analysis request")
    context: Optional[str] = Field(default=None, description="Additional context or background")
    data_type: Optional[str] = Field(default=None, description="Type of data (CSV, JSON, etc.)")
    analysis_type: Optional[str] = Field(default=None, description="Preferred analysis type")


class ConversationTracker(BaseCallbackHandler):
    """Callback to track and log conversations"""
    
    def __init__(self):
        self.conversations = []
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """Track when LLM starts"""
        self.conversations.append({
            "timestamp": datetime.now().isoformat(),
            "type": "llm_start",
            "prompts": prompts
        })
    
    def on_llm_end(self, response, **kwargs) -> None:
        """Track when LLM ends"""
        self.conversations.append({
            "timestamp": datetime.now().isoformat(),
            "type": "llm_end",
            "response": str(response)
        })


class DataSciencePromptTool(BaseTool):
    """
    LangChain tool for translating user requests into detailed R programming prompts.
    Can be used by master agents to generate technical specifications for data analysis tasks.
    """
    
    name: str = "data_science_prompt_translator"
    description: str = (
        "Translates user data analysis requests into detailed technical prompts for R code generation. "
        "Use this when you need to convert a general data analysis request into a comprehensive, "
        "technical specification that includes R packages, methodologies, and implementation details. "
        "Input should be the analysis request and optional context."
    )
    args_schema: Type[BaseModel] = DataAnalysisToolInput
    
    # Define the fields that will be set during initialization
    google_api_key: str = Field(description="Google API key for Gemini")
    memory_k: int = Field(default=5, description="Number of previous interactions to remember")
    llm: Any = Field(default=None, description="The language model instance")
    memory: Any = Field(default=None, description="Conversation memory")
    conversation_tracker: Any = Field(default=None, description="Conversation tracker")
    prompt_template: Any = Field(default=None, description="Prompt template")
    chain: Any = Field(default=None, description="LLM Chain")
    
    def __init__(self, google_api_key: str, memory_k: int = DEFAULT_MEMORY_K, **kwargs):
        """
        Initialize the Data Science Prompt Tool
        
        Args:
            google_api_key: Google API key for Gemini
            memory_k: Number of previous interactions to remember
        """
        # Initialize with the required fields
        super().__init__(
            google_api_key=google_api_key,
            memory_k=memory_k,
            **kwargs
        )
        
        # Initialize components after super().__init__
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize the LLM, memory, and other components"""
        # Initialize Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model=DEFAULT_MODEL,
            google_api_key=self.google_api_key,
            temperature=DEFAULT_TEMPERATURE,
            convert_system_message_to_human=True
        )
        
        # Initialize memory
        self.memory = ConversationBufferWindowMemory(
            k=self.memory_k,
            memory_key="chat_history",
            return_messages=True
        )
        
        # Initialize conversation tracker
        self.conversation_tracker = ConversationTracker()
        
        # Create the main prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["input", "chat_history"],
            template="""
You are a Data Science Prompt Translator Agent specializing in R programming. Your role is to translate user requests into detailed, technical prompts for an R code-generating agent.

IMPORTANT ASSUMPTIONS:
- The dataset is already loaded and available in R (like the iris dataset)
- All analysis must be performed using R programming language
- Your output will be fed to an R code-generating agent that will write the actual R code
- Focus ONLY on creating a clear, detailed prompt - do not generate any R code yourself

Your expertise includes R-based:
- Statistical analysis and hypothesis testing (stats, car, lmtest packages)
- Machine learning and predictive modeling (caret, randomForest, e1071 packages)
- Data visualization (ggplot2, plotly, lattice packages)
- Data manipulation and preprocessing (dplyr, tidyr, data.table packages)
- Model evaluation and validation techniques
- Time series analysis (forecast, tseries packages)

Previous conversation context:
{chat_history}

User Request: {input}

Translate this request into a comprehensive, technical prompt for an R code-generating agent. The prompt should include:

**ANALYSIS SPECIFICATION:**
- Clear objective statement for R implementation
- Specific analysis type (descriptive, diagnostic, predictive, prescriptive)
- Target variables and features to focus on in R
- Expected statistical/ML techniques to implement in R

**R TECHNICAL REQUIREMENTS:**
- R data preprocessing steps using dplyr/tidyr
- Specific R packages and functions to use
- R-specific algorithms or methods (lm, glm, randomForest, etc.)
- Evaluation metrics and validation approaches in R
- Required visualizations using ggplot2 or base R

**R OUTPUT SPECIFICATIONS:**
- Exact R objects and deliverables expected (models, plots, data frames, summaries)
- R code structure and organization preferences
- R documentation and commenting requirements using roxygen2 style
- R-specific data export/save requirements

**R IMPLEMENTATION DETAILS:**
- Preferred R packages from CRAN
- R code complexity level and style (tidyverse vs base R)
- R error handling with tryCatch and edge cases
- Performance considerations for R (vectorization, memory management)
- R Markdown integration if needed

Format your response as a single, cohesive prompt that the R code-generating agent can follow step-by-step to implement the requested analysis using R. Be specific about R packages, functions, and methodology.

The R code-generating agent should be able to take your prompt and immediately start writing R code without needing clarification about R-specific implementation details.
"""
        )
        
        # Create the main chain
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt_template,
            memory=self.memory,
            callbacks=[self.conversation_tracker],
            verbose=False  # Set to False for tool usage
        )
    
    def _run(self, request: str, context: str = "") -> str:
        """
        Execute the tool to generate a detailed R programming prompt
        
        Args:
            request: The user's data analysis request
            context: Additional context or background information
            
        Returns:
            Detailed R programming prompt as a string
        """
        try:
            # Validate and structure the request
            analysis_request = DataAnalysisRequest(
                request_text=request,
                context=context if context else None
            )
            
            # Combine request and context into a single input
            combined_input = f"Request: {analysis_request.request_text}"
            if analysis_request.context:
                combined_input += f"\nAdditional Context: {analysis_request.context}"
            
            # Generate the detailed prompt
            response = self.chain.predict(input=combined_input)
            
            return response
            
        except Exception as e:
            return f"Error processing request: {str(e)}"
    
    async def _arun(self, request: str, context: str = "") -> str:
        """Async version of _run"""
        return self._run(request, context)
    
    def get_conversation_history(self) -> List[BaseMessage]:
        """Get the current conversation history"""
        return self.memory.chat_memory.messages
    
    def clear_memory(self):
        """Clear the conversation memory"""
        self.memory.clear()
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get a summary of the current memory state"""
        messages = self.memory.chat_memory.messages
        return {
            "total_messages": len(messages),
            "memory_window": self.memory_k,
            "recent_topics": [msg.content[:100] + "..." if len(msg.content) > 100 else msg.content 
                            for msg in messages[-3:]] if messages else []
        }
    
    def save_conversation_log(self, filename: str = None) -> str:
        """Save conversation history to a file"""
        if filename is None:
            filename = f"conversation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump({
                "memory_summary": self.get_memory_summary(),
                "conversation_tracker": self.conversation_tracker.conversations,
                "messages": [{"type": type(msg).__name__, "content": msg.content} 
                           for msg in self.get_conversation_history()]
            }, f, indent=2)
        
        return filename


# Factory functions for easy creation
def create_data_science_tool(google_api_key: str = None, memory_k: int = DEFAULT_MEMORY_K) -> DataSciencePromptTool:
    """
    Create a DataSciencePromptTool instance
    
    Args:
        google_api_key: Google API key (if None, will try to get from environment)
        memory_k: Number of previous interactions to remember
        
    Returns:
        DataSciencePromptTool instance
    """
    if google_api_key is None:
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("Google API key must be provided or set in GOOGLE_API_KEY environment variable")
    
    return DataSciencePromptTool(
        google_api_key=google_api_key,
        memory_k=memory_k
    )


def create_multiple_tools(google_api_key: str = None, count: int = 1, memory_k: int = DEFAULT_MEMORY_K) -> List[DataSciencePromptTool]:
    """
    Create multiple DataSciencePromptTool instances (useful for parallel processing)
    
    Args:
        google_api_key: Google API key
        count: Number of tool instances to create
        memory_k: Number of previous interactions to remember
        
    Returns:
        List of DataSciencePromptTool instances
    """
    return [create_data_science_tool(google_api_key, memory_k) for _ in range(count)]


def interactive_mode():
    """Run the tool in interactive mode"""
    print("=== Data Science Prompt Translator (Interactive Mode) ===")
    print("Type 'quit', 'exit', or 'q' to stop")
    print("Type 'clear' to clear conversation memory")
    print("Type 'memory' to see memory summary")
    print()
    
    try:
        tool = create_data_science_tool()
        print("Tool initialized successfully!\n")
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    while True:
        try:
            request = input("Enter your data analysis request: ").strip()
            
            if request.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            elif request.lower() == 'clear':
                tool.clear_memory()
                print("Memory cleared!\n")
                continue
            elif request.lower() == 'memory':
                summary = tool.get_memory_summary()
                print(f"Memory Summary: {summary}")
                continue
            elif not request:
                continue
            
            context = input("Enter context (optional): ").strip()
            print("\nGenerating detailed prompt...\n")
            
            result = tool._run(request, context)
            print("Generated Prompt:")
            print("-" * 80)
            print(result)
            print("-" * 80)
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(
        description="Data Science Prompt Translator Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data_science_tool.py --interactive
  python data_science_tool.py --request "analyze customer churn" --context "telecom dataset"
  python data_science_tool.py --request "predict house prices" --save-log
        """
    )
    
    parser.add_argument(
        "--request", "-r",
        type=str,
        help="Data analysis request"
    )
    
    parser.add_argument(
        "--context", "-c",
        type=str,
        default="",
        help="Additional context for the request"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    
    parser.add_argument(
        "--memory-k", "-k",
        type=int,
        default=DEFAULT_MEMORY_K,
        help=f"Number of previous interactions to remember (default: {DEFAULT_MEMORY_K})"
    )
    
    parser.add_argument(
        "--save-log",
        action="store_true",
        help="Save conversation log to file"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file for the generated prompt"
    )
    
    args = parser.parse_args()
    
    # Interactive mode
    if args.interactive:
        interactive_mode()
        return
    
    # Single request mode
    if not args.request:
        parser.print_help()
        return
    
    try:
        tool = create_data_science_tool(memory_k=args.memory_k)
        result = tool._run(args.request, args.context)
        
        print("Generated Prompt:")
        print("=" * 80)
        print(result)
        print("=" * 80)
        
        # Save to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                f.write(result)
            print(f"\nPrompt saved to: {args.output}")
        
        # Save conversation log if requested
        if args.save_log:
            log_file = tool.save_conversation_log()
            print(f"Conversation log saved to: {log_file}")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()