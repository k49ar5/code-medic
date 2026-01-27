import os
import sys
import subprocess
import tempfile
from typing import Annotated, TypedDict, List
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langgraph.graph import StateGraph, END
from langchain_community.utils.math import cosine_similarity


# --- 1. STATE DEFINITION ---
class AgentState(TypedDict):
    code: str
    error: str
    docs: List[str]
    best_doc: str
    iterations: int
    max_iterations: int  # Safeguard against infinite loops
    fixed: bool
    history: List[str]  # Agent's short-term memory to avoid repeating mistakes


# Initialize Local LLM (Llama 3)
# Temperature 0 is crucial for consistent code generation
llm = OllamaLLM(model="llama3", temperature=0)
embeddings = OllamaEmbeddings(model="llama3")


# --- 2. NODES (Function Definitions) ---

def retrieve_docs(state: AgentState):
    """
    Simulates a Vector Database retrieval.
    In a production app, this would be a Qdrant/ChromaDB query.
    """
    print("--- [NODE: Retrieval] Fetching candidates from documentation ---")
    raw_docs = [
        "Python Syntax: Function definitions must end with a colon (:).",
        "Common Error: SyntaxError often occurs due to unclosed parentheses or missing quotes.",
        "Best Practice: print() function requires string arguments to be quoted.",
        "Tip: Use f-strings for variable interpolation in Python 3.6+."
    ]
    return {"docs": raw_docs}


def rerank_docs(state: AgentState):
    """
    Two-Stage Retrieval: Using Cosine Similarity to select the most relevant snippet.
    This reduces noise and saves context window space.
    """
    print("--- [NODE: Re-ranker] Selecting most relevant context ---")
    query_emb = embeddings.embed_query(state["error"])
    doc_embs = embeddings.embed_documents(state["docs"])

    scores = cosine_similarity([query_emb], doc_embs)[0]
    best_index = scores.argmax()

    return {"best_doc": state["docs"][best_index]}


def repair_code(state: AgentState):
    """
    The Reasoning Node: Injects current error, documentation, and history
    into the prompt to guide the LLM.
    """
    print(f"--- [NODE: Repair] Attempt {state['iterations'] + 1} ---")

    # We pass the last failed attempt to the model to prevent logical loops
    attempts_history = "\n".join(state["history"][-2:])

    prompt = f"""
    Current Code: {state['code']}
    Error encountered: {state['error']}
    Context from Docs: {state['best_doc']}

    Previous failed attempts:
    {attempts_history}

    Instruction: Fix the code. Use only the provided context. 
    Do not repeat previous mistakes. Return ONLY the raw code.
    """

    response = llm.invoke(prompt)
    clean_code = response.strip().replace("```python", "").replace("```", "")

    return {
        "code": clean_code,
        "iterations": state["iterations"] + 1,
        "history": state["history"] + [clean_code]
    }


def test_code(state: AgentState):
    """
    The Validation Node: Executes the code in a subprocess to get
    real feedback from the Python interpreter.
    """
    print("--- [NODE: Validator] Executing code for verification ---")

    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode='w', encoding='utf-8') as tmp:
        tmp.write(state["code"])
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=5
        )
        # Clean up temp file
        os.remove(tmp_path)

        if result.returncode == 0:
            print(">> Success: Code executed without errors.")
            return {"fixed": True, "error": ""}
        else:
            # Capture the last line of the traceback for the next iteration
            last_err = result.stderr.strip().splitlines()[-1] if result.stderr else "Unknown error"
            print(f">> Failure: {last_err}")
            return {"fixed": False, "error": last_err}

    except Exception as e:
        if os.path.exists(tmp_path): os.remove(tmp_path)
        return {"fixed": False, "error": str(e)}


# --- 3. GRAPH CONSTRUCTION ---

def router(state: AgentState):
    """Decides whether to retry or stop based on results and safeguards."""
    if state["fixed"]:
        return "end"
    if state["iterations"] >= state["max_iterations"]:
        print(">> Safeguard: Max iterations reached. Stopping.")
        return "end"
    return "repair"


workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("retrieve", retrieve_docs)
workflow.add_node("rerank", rerank_docs)
workflow.add_node("repair", repair_code)
workflow.add_node("test", test_code)

# Build Edges
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "rerank")
workflow.add_edge("rerank", "repair")
workflow.add_edge("repair", "test")

# Add Conditional Logic (The Loop)
workflow.add_conditional_edges(
    "test",
    router,
    {"end": END, "repair": "repair"}
)

app = workflow.compile()

# --- 4. EXECUTION ---

if __name__ == "__main__":
    initial_input = {
        "code": "def greet(): print('Hello World')",  # Missing colon (initially) or other syntax error
        "error": "SyntaxError: expected ':'",
        "docs": [],
        "best_doc": "",
        "iterations": 0,
        "max_iterations": 3,
        "fixed": False,
        "history": []
    }

    print("--- STARTING AGENTIC WORKFLOW ---")
    for output in app.stream(initial_input):
        for node, data in output.items():
            print(f"Finished node: {node}")
    print("--- WORKFLOW COMPLETE ---")