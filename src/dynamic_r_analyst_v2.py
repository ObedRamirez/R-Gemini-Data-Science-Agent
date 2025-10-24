import streamlit as st
import pandas as pd
import subprocess
import os
import shutil
from langchain.agents import AgentExecutor, Tool
from langchain.agents import create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Import sub-agents
from data_science_prompt_agent_v2 import create_data_science_tool
from data_science_coding_agent_v2 import create_r_agent
from data_science_audit_agent_v2 import RCodeAuditor

# --- Configuration ---
# Use Streamlit secrets for API key in production, with a fallback for local development
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Google API key must be provided or set in GOOGLE_API_KEY environment variable")
DATA_FILENAME = "data.csv"  # Consistent filename for uploaded data

# --- Sub-agent Instantiation ---
# Instantiate the sub-agents once to be used by the tools
try:
    prompt_tool_instance = create_data_science_tool(google_api_key=GOOGLE_API_KEY)
    coding_agent_instance = create_r_agent(google_api_key=GOOGLE_API_KEY)
    audit_agent_instance = RCodeAuditor(google_api_key=GOOGLE_API_KEY)
except ValueError as e:
    st.error(f"Failed to initialize AI agents. Please check your GOOGLE_API_KEY. Error: {e}")
    st.stop()


# --- Tool Functions for the Master Agent ---

def generate_detailed_prompt(query: str) -> str:
    """Generates a detailed technical prompt from a user query."""
    with st.status("1. Generating detailed prompt...", expanded=True) as status:
        detailed_prompt = prompt_tool_instance._run(request=query)
        st.expander("View Detailed Prompt").code(detailed_prompt, language='markdown')
        status.update(label="✅ Detailed prompt generated.", state="complete")
    return detailed_prompt

def generate_rmarkdown_code(detailed_prompt: str) -> str:
    """Generates R Markdown code from a detailed prompt."""
    with st.status("2. Generating R Markdown code...", expanded=True) as status:
        result = coding_agent_instance.analyze(analysis_request=detailed_prompt)
        st.expander("View Generated Rmd Code").code(result.rmarkdown_code, language='r')
        status.update(label="✅ R Markdown code generated.", state="complete")
    return result.rmarkdown_code

def audit_and_correct_code(rmarkdown_code: str) -> str:
    """Audits and corrects R Markdown code."""
    with st.status("3. Auditing generated code...", expanded=True) as status:
        corrected_code = audit_agent_instance.quick_fix(code=rmarkdown_code, file_extension=".Rmd")
        st.expander("View Audited Rmd Code").code(corrected_code, language='r')
        status.update(label="✅ Code audited and corrected.", state="complete")
    return corrected_code

def execute_rmarkdown_and_get_result(rmarkdown_code: str) -> str:
    """
    Writes R Markdown code to a file, executes it in Docker to produce an HTML report,
    and returns the path to the HTML report.
    """
    with st.status("4. Executing R Markdown in Docker...", expanded=True) as status:
        rmd_filename = "dynamic_script.Rmd"
        with open(rmd_filename, "w", encoding="utf-8") as f:
            f.write(rmarkdown_code)
        st.write(f"✅ `{rmd_filename}` written.")

        output_html = "result.html"
        final_html_in_streamlit = "final_result.html"

        if os.path.exists(output_html):
            os.remove(output_html)
        if os.path.exists(final_html_in_streamlit):
            os.remove(final_html_in_streamlit)

        try:
            work_dir = os.path.abspath(os.getcwd())
            st.write("Building Docker image `r-executor-v2`...")
            build_process = subprocess.run(
                ["docker", "build", "-t", "r-executor-v2", "."],
                capture_output=True, text=True, check=True, cwd=work_dir
            )
            st.expander("View Docker Build Log").code(build_process.stdout)

            st.write("Running Docker container...")
            run_process = subprocess.run(
                ["docker", "run", "--rm", "-v", f"{work_dir}:/app", "r-executor-v2"],
                capture_output=True, text=True, check=True, cwd=work_dir
            )
            st.expander("View Docker Run Log").code(run_process.stdout)

            if os.path.exists(output_html):
                shutil.copy(output_html, final_html_in_streamlit)
                status.update(label="✅ Execution successful!", state="complete")
                return f"Execution successful. Report is available at {final_html_in_streamlit}"
            else:
                st.error("Execution finished, but the result.html file was not found.")
                status.update(label="⚠️ Execution failed: result.html not generated.", state="error")
                return "Execution failed: result.html not generated."

        except subprocess.CalledProcessError as e:
            st.error(f"Error executing R Markdown in Docker: {e}")
            st.error(f"Stderr: {e.stderr}")
            st.expander("View Failing Rmd Code").code(rmarkdown_code, language='r')
            status.update(label=f"❌ Execution failed!", state="error")
            return f"Execution failed with error: {e.stderr}"

# --- Master Agent Setup ---
tools = [
    Tool(
        name="GenerateDetailedPrompt",
        func=generate_detailed_prompt,
        description="Use this tool first. It takes a user's natural language query and converts it into a detailed technical prompt for the coding agent."
    ),
    Tool(
        name="GenerateRMarkdownCode",
        func=generate_rmarkdown_code,
        description="Use this tool after generating a detailed prompt. It takes the detailed prompt and generates the full R Markdown code for the analysis."
    ),
    Tool(
        name="AuditAndCorrectCode",
        func=audit_and_correct_code,
        description="Use this tool after generating the R Markdown code. It reviews the code for errors and quality issues and returns a corrected version."
    ),
    Tool(
        name="ExecuteRMarkdownAndGetResult",
        func=execute_rmarkdown_and_get_result,
        description="Use this tool last, after the code has been generated and audited. It executes the final R Markdown code and returns the path to the HTML result file."
    )
]

template = '''You are the Dynamic R Analyst, a master agent that orchestrates a team of sub-agents to perform data analysis.
Your goal is to take a user's request, generate and audit the necessary R Markdown code, execute it, and return the path to the final HTML report.

You will follow a strict workflow:
1.  **Generate Detailed Prompt**: Use the `GenerateDetailedPrompt` tool with the user's input to create a technical prompt for the coding agent.
2.  **Generate R Markdown Code**: Use the `GenerateRMarkdownCode` tool with the detailed prompt from the previous step.
3.  **Audit and Correct Code**: Use the `AuditAndCorrectCode` tool with the generated R Markdown code to check for errors and improve it.
4.  **Execute and Get Result**: Use the `ExecuteRMarkdownAndGetResult` tool with the final, audited code to run the analysis and get the path to the HTML report.

**Instructions for data loading:**
- If a user has uploaded a file, it will be available in the container at `/app/data.csv`. The code **must** load it using `read.csv("/app/data.csv")`.
- The `GenerateDetailedPrompt` tool needs to be aware of the uploaded file and its columns.
- If no file is uploaded, you should use a default dataset like `iris` or `mtcars`.

You have access to the following tools:

{tools}

Use the tools in the sequence described above. Do not skip steps.

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have the final result (the message from the execution tool), you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [The final response from the `ExecuteRMarkdownAndGetResult` tool, which includes the path to the HTML file.]
```

Begin!

New input: {input}
{agent_scratchpad}'''

prompt = ChatPromptTemplate.from_template(template)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("Dynamic R Analyst v2")
st.write("This agent uses sub-agents for prompting, coding, auditing, and execution to generate and run R Markdown analysis.")

uploaded_file = st.file_uploader("Upload a CSV file (optional)", type="csv")

query_input_placeholder = "e.g., 'Create a summary table and a boxplot of sepal length for each species in the iris dataset.'"
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:")
        st.dataframe(df.head())
        df.to_csv(DATA_FILENAME, index=False)
        st.session_state.column_names = df.columns.tolist()
        query_input_placeholder = f"e.g., 'Analyze my uploaded data. The columns are: {', '.join(df.columns.tolist())}'."
    except Exception as e:
        st.error(f"Error reading or saving the uploaded CSV file: {e}")


query = st.text_input(
    "Describe the data analysis you want to perform:",
    placeholder=query_input_placeholder,
)

if query:
    final_query = query
    if "column_names" in st.session_state:
        final_query += f"\n\nCONTEXT: The user has uploaded a dataset named `{DATA_FILENAME}` with the following columns: {st.session_state.column_names}. The analysis must use this file."

    st.markdown("---")
    st.subheader("Agent Execution Log")
    res = agent_executor.invoke({"input": final_query})
    
    st.markdown("---")
    st.subheader("Final Result")
    st.info(res["output"], icon="🤖")

    output_message = res["output"]
    html_files = [word for word in output_message.split() if word.endswith('.html')]

    if html_files:
        html_file_path = html_files[0]
        if os.path.exists(html_file_path):
            st.subheader("📊 Analysis Report")
            with open(html_file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=800, scrolling=True)
        else:
            st.error(f"Agent reported completion, but the file '{html_file_path}' was not found.")
    else:
        st.warning("Agent finished, but did not return a path to a viewable HTML report.")