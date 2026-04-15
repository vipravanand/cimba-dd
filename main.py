

from pydantic import BaseModel, Field
from openai import OpenAI
from typing import List
import os

llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class QueryExpander(BaseModel):
    # Change List to List[str]
    queries: List[str] = Field(
        max_length=5, 
        min_length=5, 
        description="Rewritten version of the original query"
    )
def query_expansion(user_query): 
    prompt = f"""
### Role
You are a Semantic Search Engineer. Your goal is to rewrite a shorthand user query into 5 high-recall variations that align with the linguistic structure of a Knowledge Base (KB).

### KB Style Reference
The KB is written in full, interrogative sentences. 
Example style: 
- "How many customers do I have?"
- "How many products do I have?"

### Instructions
1. **Structural Alignment**: Transform the shorthand into a full natural language question starting with "How many", "What is", or "Can you tell me".
2. **Synonym Expansion**: For the core nouns and verbs, use professional synonyms to increase the surface area for the bi-encoder (e.g., for "count", use "total", "volume", "sum", or "quantity").
3. **Diversity**: Ensure each of the 5 variations uses a slightly different phrasing to capture different vector embeddings.
4. **Output**: Return ONLY a bulleted list of the 5 expanded queries.

### Input Query
Target: "{user_query}"
"""
    
    msgs = [{"role": "system", "content": prompt}]
    response = llm.chat.completions.parse( 
        temperature=0.2, 
        model = "gpt-4o-mini", 
        response_format=QueryExpander, 
        messages=msgs
    )

    return response.choices[0].message.parsed.model_dump()["queries"]


def get_embeddings(input): 
    pass


def kb_search(query): 
    queries = query_expansion(query)
    kb = [
        "How many customers do I have", 
        "How may products do I have" 
    ]

    kb_embeddings = get_embeddings( kb) 
    queries_final = [query] + queries
    query_embeddings = get_embeddings( queries_final)

    

if __name__ == "__main__":
    print(query_expansion("my customer count"))





