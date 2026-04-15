import pytest
import os
from main_2 import BISovereignEngine # Ensure your engine is in your_module.py

# --- 1. Test Configuration & Setup ---
KB_CORPUS = [
    {"text": "How many customers do I have", "action": "COUNT", "entity": "customer"},
    {"text": "How many products do I have", "action": "COUNT", "entity": "product"},
    {"text": "How many products are in stock", "action": "COUNT", "entity": "stock"},
    {"text": "List all my customers", "action": "LIST", "entity": "customer"},
]

@pytest.fixture(scope="module")
def engine():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("Skipping tests because OPENAI_API_KEY is not set.")
    return BISovereignEngine(openai_api_key=api_key, kb_data=KB_CORPUS)

# --- 2. The Parameterized Test Dataset ---
TEST_CASES = [
    # TYPE: Easy Matches (Direct or near-direct phrasing)
    ("How many customers?", "SUCCESS", "How many customers do I have"),
    ("list customers", "SUCCESS", "List all my customers"),
    ("stock levels", "SUCCESS", "How many products are in stock"),

    # TYPE: Hard Matches (Testing the Logic Gate & Shorthand)
    # "count" vs "list" is the critical BI differentiator
    ("customer count", "SUCCESS", "How many customers do I have"),
    ("all my buyers", "SUCCESS", "List all my customers"), # Synonym mapping
    ("total products", "SUCCESS", "How many products do I have"),

    # TYPE: Out of Scope (The Kill Switch)
    ("What is the weather in London?", "OUT_OF_SCOPE", None),
    ("Where is my nearest office?", "OUT_OF_SCOPE", None),
    ("Tell me a joke about data", "OUT_OF_SCOPE", None),

    # TYPE: Ambiguous (Should trigger the Confidence Gap)
    # Queries that don't specify LIST or COUNT clearly
    ("customers", "AMBIGUOUS", None), 
    ("product information", "AMBIGUOUS", None),
]

# --- 3. The Test Execution ---
@pytest.mark.parametrize("query, expected_status, expected_text", TEST_CASES)
def test_bi_engine_logic(engine, query, expected_status, expected_text):
    """
    Validates that the Neural-Symbolic bridge correctly classifies and 
    ranks queries based on the Action/Entity logic gate and Gap Analysis.
    """
    result = engine.search(query)
    
    # Assert Status (SUCCESS, OUT_OF_SCOPE, or AMBIGUOUS)
    assert result["status"] == expected_status, f"Query '{query}' failed. Expected {expected_status}, got {result['status']}"

    # Assert correct KB item was selected if it was a SUCCESS
    if expected_status == "SUCCESS":
        assert result["match"]["text"] == expected_text
        # Ensure logic gate worked: Action and Entity must match the query intent
        assert result["score"] > 0.7  # Basic quality check

    # Specific check for Out of Scope to ensure no match was "forced"
    if expected_status == "OUT_OF_SCOPE":
        assert result["match"] is None