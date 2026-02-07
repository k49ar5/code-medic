# Code Medic: Self-Correcting Agentic RAG

Code Medic is an autonomous agent designed for automated Python code repair. The system transitions from traditional linear RAG to a state-based cyclic architecture, enabling the model to verify its own output through real-time execution feedback and rigorous automated testing.

## System Architecture

The agent is implemented as a state machine using LangGraph, adhering to an advanced iterative workflow:

1.  **Retrieval**: Fetches technical documentation candidates using localized Ollama embeddings.
2.  **Re-ranking**: Utilizes Cosine Similarity for second-stage filtering to select the most relevant context, optimizing the LLM context window.
3.  **Repair**: Generates a code fix based on the current error, retrieved documentation, and short-term memory of previous failed attempts.
4.  **Validation**: Executes the generated code in an isolated subprocess to capture real-time interpreter feedback.
5.  **Quality Assurance**: Employs Pytest suites to verify the functional integrity of the repaired artifacts.
6.  **Feedback Loop**: If validation fails, the traceback is captured and reinjected into the state for subsequent repair iterations.

## Key Technical Features

* **Cyclic State Management**: Mitigates hallucinations by grounding the LLM in actual Python interpreter feedback.
* **LLMOps & Observability**: Integrated with LangSmith for full-stack tracing, latency monitoring, and debugging of the agent's reasoning chains.
* **Automated Testing Framework**: A dedicated Pytest directory ensures that repair logic remains consistent and prevents regressions in the agent's performance.
* **Short-term Memory**: The AgentState tracks iteration history to avoid repetitive logical errors during the repair process.
* **Subprocess Isolation**: Code verification is performed in a controlled environment to safely capture stderr for diagnostic purposes.
* **Deterministic Safeguards**: Configurable iteration limits to ensure operational stability and resource management.

## Tech Stack

* **Orchestration**: LangGraph
* **Observability**: LangSmith
* **Testing**: Pytest
* **LLM**: Llama 3.2 (via Ollama)
* **Vector Operations**: Cosine Similarity via LangChain
* **Environment**: Python 3.10+

## Setup and Installation

### 1. Prerequisites
Ensure Ollama is installed and the required model is pulled:
```bash
ollama pull llama3.2:1b
```
## 2. Environment Configuration

Create a `.env` file in the root directory and add it to your `.gitignore` to keep your credentials safe:

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="[https://api.smith.langchain.com](https://api.smith.langchain.com)"
LANGCHAIN_API_KEY="your_langsmith_api_key"
LANGCHAIN_PROJECT="Code-Medic-Agent"
```
## 3 Install Dependencies:
   ```bash
      pip install langgraph langchain_ollama langchain_community pytest python-dotenv
   ```
## Usage

### Running the Agent
To start the autonomous repair process, execute the following command in your terminal:

```bash
python main.py
```

## Running the Test Suite

To run the automated unit and integration tests and verify the system's performance, execute the following command:

```bash
python -m pytest -v
```

[!NOTE] This project was developed as an advanced implementation of Self-Corrective RAG (Retrieval-Augmented Generation) specifically tailored for Python development environments.
