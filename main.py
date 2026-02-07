import os
import sys
import re
import logging
import subprocess
import tempfile
from typing import TypedDict, List, Dict, Any, Optional

# Core LangChain & LangGraph imports
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langgraph.graph import StateGraph, END
from langchain_community.utils.math import cosine_similarity

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
# It is recommended to use environment variables or a .env file for sensitive keys
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "YOUR_API_KEY"
os.environ["LANGCHAIN_PROJECT"] = "Llama-3.2-Code-Repair-Production"

MODEL_NAME = "llama3.2:1b"


# --- STATE DEFINITION ---
class AgentState(TypedDict):
    """
    Represents the internal state of the agent.
    """
    code: str
    error: str
    docs: List[str]
    best_doc: Optional[str]
    iterations: int
    max_iterations: int
    fixed: bool
    history: List[str]


# --- UTILITIES ---
def extract_python_code(text: str) -> str:
    """
    Robustly extracts python code from LLM markdown response.
    """
    pattern = r"```python\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback to removing backticks if block not specified correctly
    return text.replace("```", "").strip()


# --- NODES ---

def retrieve_docs(state: AgentState) -> Dict[str, Any]:
    """
    Node: Context Retrieval.
    Fetches relevant documentation snippets for the current error.
    """
    logger.info("Retrieving documentation candidates")

    # Static knowledge base for demonstration purposes
    knowledge_base = [
        "Python Syntax: Function definitions must end with a colon (:).",
        "Common Error: SyntaxError often occurs due to unclosed parentheses or missing quotes.",
        "Best Practice: print() function requires string arguments to be quoted.",
        "Variable Interpolation: Use f-strings (f'...') for dynamic strings in Python 3.6+."
    ]
    return {"docs": knowledge_base}


def rerank_docs(state: AgentState) -> Dict[str, Any]:
    """
    Node: Semantic Re-ranking.
    Uses embeddings to find the most relevant document snippet to the error.
    """
    logger.info("Selecting most relevant documentation snippet via Cosine Similarity")

    embedder = OllamaEmbeddings(model=MODEL_NAME)

    query_vector = embedder.embed_query(state["error"])
    doc_vectors = embedder.embed_documents(state["docs"])

    similarities = cosine_similarity([query_vector], doc_vectors)[0]
    best_idx = similarities.argmax()

    return {"best_doc": state["docs"][best_idx]}


def repair_code(state: AgentState) -> Dict[str, Any]:
    """
    Node: Logic Repair.
    Utilizes the LLM to provide a fix based on context and history.
    """
    logger.info(f"Initiating repair attempt {state['iterations'] + 1}")

    llm = OllamaLLM(model=MODEL_NAME, temperature=0)

    prompt = (
        "SYSTEM: You are a senior Python developer. Fix the provided code based on the error and context.\n"
        f"CONTEXT: {state['best_doc']}\n"
        f"ERROR: {state['error']}\n"
        f"CURRENT_CODE: {state['code']}\n\n"
        "INSTRUCTION: Return ONLY the corrected code inside a ```python code block. No explanations."
    )

    raw_response = llm.invoke(prompt)
    clean_code = extract_python_code(raw_response)

    return {
        "code": clean_code,
        "iterations": state["iterations"] + 1,
        "history": state["history"] + [clean_code]
    }


def validate_code(state: AgentState) -> Dict[str, Any]:
    """
    Node: Runtime Validation.
    Executes the code in an isolated subprocess to verify the fix.
    """
    logger.info("Executing runtime validation")

    with tempfile.NamedTemporaryFile(suffix=".py", mode='w', encoding='utf-8', delete=False) as tmp:
        tmp.write(state["code"])
        tmp_path = tmp.name

    try:
        process = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=5
        )

        if process.returncode == 0:
            logger.info("Validation successful")
            return {"fixed": True, "error": ""}

        # Capture the specific error message for the next iteration
        error_msg = process.stderr.strip().splitlines()[-1] if process.stderr else "Runtime Error"
        logger.warning(f"Validation failed: {error_msg}")
        return {"fixed": False, "error": error_msg}

    except Exception as e:
        logger.error(f"Execution failed: {str(e)}")
        return {"fixed": False, "error": str(e)}
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# --- GRAPH CONSTRUCTION ---

def execution_router(state: AgentState) -> str:
    """
    Determines the next step in the workflow.
    """
    if state["fixed"]:
        return "end"
    if state["iterations"] >= state["max_iterations"]:
        logger.error("Max iterations reached. Agent failed to fix code.")
        return "end"
    return "repair"


def build_workflow() -> StateGraph:
    builder = StateGraph(AgentState)

    # Define Nodes
    builder.add_node("retrieve", retrieve_docs)
    builder.add_node("rerank", rerank_docs)
    builder.add_node("repair", repair_code)
    builder.add_node("validate", validate_code)

    # Define Edges
    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "rerank")
    builder.add_edge("rerank", "repair")
    builder.add_edge("repair", "validate")

    # Add Conditional Routing
    builder.add_conditional_edges(
        "validate",
        execution_router,
        {
            "end": END,
            "repair": "repair"
        }
    )

    return builder.compile()


# --- ENTRY POINT ---

if __name__ == "__main__":
    # Initialize the graph
    app = build_workflow()

    # Initial input data
    initial_payload = {
        "code": "def greet() print('Hello World')",  # Missing colon
        "error": "SyntaxError: expected ':'",
        "docs": [],
        "best_doc": None,
        "iterations": 0,
        "max_iterations": 3,
        "fixed": False,
        "history": []
    }

    logger.info("Starting Agentic Workflow Execution")

    try:
        for output in app.stream(initial_payload):
            # Output represents updates from each node
            for node_name, state_update in output.items():
                logger.debug(f"Node '{node_name}' completed execution")

        logger.info("Workflow execution complete")
    except Exception as e:
        logger.critical(f"Workflow crashed: {str(e)}")