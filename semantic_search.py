from fastembed.common.types import NumpyArray


import os
import numpy as np
from typing import List, Dict,  Any
from pydantic import BaseModel, Field
from openai import OpenAI
from fastembed import LateInteractionTextEmbedding
from fastembed.postprocess import Muvera


class QueryAnalysis(BaseModel):
    action: str = Field(description="Must be LIST, COUNT, SUM, COMPARE or UNKNOWN")
    entity: str = Field(description="Must be customer, product, stock, or UNKNOWN")
    is_out_of_scope: bool = Field(description="True if query is not about business ")
    queries: List[str] = Field(min_length=3, max_length=3, description="3 semantic variations of the query")

class SemanticSearhEngine:
    def __init__(self, openai_api_key: str, kb_data: List[Dict[str, str]]):

        self.client = OpenAI(api_key=openai_api_key)
        self.kb_data = kb_data
        
        self.model_name = "answerdotai/answerai-colbert-small-v1"
        self.model = LateInteractionTextEmbedding(model_name=self.model_name)
        
        self.muvera = Muvera.from_multivector_model(
            model=self.model,
            k_sim=6,     
            dim_proj=32,
            r_reps=20 
        )
        
        self.confidence_gap_threshold = 0.15  # ~15% gap between top 2 required for a clear triaging
        self.min_raw_score_floor = 700.0      # DOT Product scores 

        print("Indexing KB with MUVERA...")
        kb_texts = [item['text'] for item in self.kb_data]
        kb_embs = list[NumpyArray](self.model.embed(kb_texts))
        
        self.kb_vectors = [self.muvera.process_document(emb) for emb in kb_embs]

    def _relative_scaling(self, scores: np.ndarray) -> np.ndarray:
        """Scales scores to 0-1 range for the KB-Candidates"""
        s_min, s_max = scores.min(), scores.max()
        if s_max == s_min:
            return np.ones_like(scores)
        return (scores - s_min) / (s_max - s_min)

    def _get_raw_muvera_scores(self, user_query: str) -> np.ndarray:
        """Retrieves dot product scores for a query, candidate pair"""
        query_emb = list(self.model.query_embed([user_query]))[0]
        query_vec = self.muvera.process_document(query_emb)
        return np.array([np.dot(query_vec, kb_vec) for kb_vec in self.kb_vectors])

    def _query_rewriting_expansion(self, user_query: str) -> QueryAnalysis:
  
        prompt = f"""
ROLE: You are a Senior BI Intent Parser. Your job is to translate messy, shorthand user queries into a structured logical intent and three semantic variations for vector search.

DATA CATALOG:
- Allowed Entities: [customer, product, stock, UNKNOWN]
- Allowed Actions: [LIST, COUNT, SUM,  UNKNOWN]

USER QUERY: "{user_query}"

TASK:
1. CANONICAL MAPPING:
   - Identify the primary ACTION. (e.g., "How many", "Total", "Number of" -> COUNT; "Show me", "Who are", "List" -> LIST).
   - Identify the primary ENTITY. If the entity is a synonym (e.g., "buyers" or "clients"), map it to "customer".
   - If the query is unrelated to the catalog (e.g., weather, jokes, general help), set is_out_of_scope = True and Action/Entity = "UNKNOWN".

2. SEMANTIC SPANNING (Generate 3 Hypotheses):
   To maximize retrieval accuracy, provide 3 distinct variations that might match a formal report title:
   - Variation 1 (Formal/Instructional): A direct, professional report title (e.g., "Total Customer Count Summary").
   - Variation 2 (Natural Language): How a human would ask it clearly (e.g., "How many customers are currently in the system?").
   - Variation 3 (Telegraphic/Keyword): A condensed version focused on nouns and verbs (e.g., "Customer count total").

"""

        completion = self.client.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}],
            response_format=QueryAnalysis,
        )
        return completion.choices[0].message.parsed

    def search(self, user_query: str) -> Dict[str, Any]:
 
        analysis = self._query_rewriting_expansion(user_query) #Query Expansion
        
        if analysis.is_out_of_scope or analysis.action == "UNKNOWN" or analysis.entity == "UNKNOWN": # Filter Out of Scope Queries 
            return {"status": "AMBIGUOUS", "match": None, "score": 0.0}

        master_raw_scores = np.zeros(len(self.kb_data))
        for query in analysis.queries:
            hyp_scores = self._get_raw_muvera_scores(query)
            master_raw_scores = np.maximum(master_raw_scores, hyp_scores)


        scaled_scores = self._relative_scaling(master_raw_scores) # score 

        # PHASE 4: Applying the Logic Gate
        final_rankings = []
        for i, scaled_score in enumerate(scaled_scores):

            final_score = scaled_score 
            final_rankings.append({
                "item": self.kb_data[i],
                "score": final_score,
                "raw": master_raw_scores[i]
            })

        final_rankings.sort(key=lambda x: x['score'], reverse=True)
        winner = final_rankings[0]
        runner_up = final_rankings[1] if len(final_rankings) > 1 else None

        # Minimal Dot Product Score
        if winner['raw'] < self.min_raw_score_floor:
            return {"status": "OUT_OF_SCOPE", "match": None, "score": winner['score']}

        # GAP Between the best candidate and 2nd best candidate 
        if runner_up:
            margin = winner['score'] - runner_up['score']
            if margin < self.confidence_gap_threshold:
                return {
                    "status": "AMBIGUOUS",
                    "match": winner['item'],
                    "margin": margin,
                    "note": "Top two reports are semantically too close."
                }

        return {"status": "SUCCESS", "match": winner['item'], "score": winner['score']}



if __name__ == "__main__":

    kb = [
        {"text": "How many customers do I have", "action": "COUNT", "entity": "customer"},
        {"text": "How many products do I have", "action": "COUNT", "entity": "product"},
        {"text": "How many products are in stock", "action": "COUNT", "entity": "stock"},
        {"text": "List all my customers", "action": "LIST", "entity": "customer"},
    ]


    engine = SemanticSearhEngine(openai_api_key=os.getenv("OPENAI_API_KEY"), kb_data=kb)

    result = engine.search("customer count")
    print(result)
    result = engine.search("total locations")
    print(result)
