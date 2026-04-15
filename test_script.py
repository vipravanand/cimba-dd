import pytest
import os
from semantic_search import SemanticSearhEngine 


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
    return SemanticSearhEngine(openai_api_key=api_key, kb_data=KB_CORPUS)

# The Parameterized Test Dataset( query , expected_status, expected_candidate)
TEST_CASES = [
    ("How many customers?", "SUCCESS", "How many customers do I have"),
    ("list customers", "SUCCESS", "List all my customers"),
    ("stock levels", "SUCCESS", "How many products are in stock"),
    ("customer count", "SUCCESS", "How many customers do I have"),
    ("all my buyers", "SUCCESS", "List all my customers"), 
    ("total products", "SUCCESS", "How many products do I have"),
    ("customers", "SUCCESS", "List all my customers"),
    ("product information", "AMBIGUOUS", None),
    ("generate report", "AMBIGUOUS", None)
]


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

