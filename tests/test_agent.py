import pytest
from main import build_workflow, AgentState

@pytest.fixture
def agent_executor():
    """Provides a compiled instance of the LangGraph workflow for testing."""
    return build_workflow()

def test_syntax_repair_logic(agent_executor):
    """
    Test if the agent can successfully fix a missing colon syntax error.
    """
    # Arrange: Setup initial state
    initial_state = {
        "code": "def calculate_dose() print('10mg')",
        "error": "SyntaxError: expected ':'",
        "docs": [],
        "best_doc": None,
        "iterations": 0,
        "max_iterations": 2,
        "fixed": False,
        "history": []
    }

    # Act: Execute the graph
    # Using invoke() is preferred for testing as it returns the final state
    result = agent_executor.invoke(initial_state)

    # Assert: Verify the outcome
    assert result["fixed"] is True
    assert "def calculate_dose():" in result["code"]
    assert result["iterations"] > 0

def test_max_iteration_safeguard(agent_executor):
    """
    Ensure the agent terminates if it cannot fix the code within limits.
    """
    # Arrange: Unfixable code
    initial_state = {
        "code": "invalid_syntax!!!",
        "error": "Generic Error",
        "docs": ["Irrelevant documentation"],
        "best_doc": None,
        "iterations": 0,
        "max_iterations": 1,
        "fixed": False,
        "history": []
    }

    # Act
    result = agent_executor.invoke(initial_state)

    # Assert
    assert result["iterations"] == 1
    # The agent should stop even if not fixed