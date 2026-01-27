# Code Medic: Self-Correcting Agentic RAG

Code Medic is an autonomous agent designed for automated Python code repair. It transitions from traditional linear RAG to a state-based cyclic architecture, allowing the model to verify its own output through real-time execution feedback.

## System Architecture

The agent is built as a state machine using LangGraph, following the ReAct (Reason + Act) pattern:

1. **Retrieval**: Fetches technical documentation candidates using Ollama embeddings.
2. **Re-ranking**: Performs a second-stage filter using Cosine Similarity to select the most relevant context, reducing noise in the prompt.
3. **Repair**: Generates a code fix based on the current error, documentation, and a short-term memory of previous failed attempts.
4. **Validation**: Executes the generated code in an isolated subprocess.
5. **Feedback Loop**: If execution fails, the traceback is captured and fed back into the state for the next repair iteration.

[Image of an agentic RAG loop diagram showing Retrieval, Re-ranking, Repair, and Validation nodes]

## Key Technical Features

- **Cyclic State Management**: Prevents hallucinations by grounding the LLM in actual interpreter feedback.
- **Short-term Memory**: The AgentState tracks iteration history to avoid repeating failed logic.
- **Subprocess Isolation**: Code verification is performed in a controlled environment to capture stderr for diagnostic purposes.
- **Deterministic Safeguards**: Includes maximum iteration limits to ensure operational stability and cost control.

## Stack

- **Orchestration**: LangGraph
- **LLM**: Llama 3 (via Ollama)
- **Vector Operations**: Cosine Similarity via LangChain
- **Environment**: Python 3.10+

## Setup

1. Ensure Ollama is installed and Llama 3 is pulled:
   ```bash
   ollama pull llama3
2. Install requirements:
   ```bash
   pip install langgraph langchain_ollama langchain_community
4. Run the agent:
    ```bash
   python main.py
