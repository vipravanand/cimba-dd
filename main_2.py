import os
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from pydantic import BaseModel, Field
from openai import OpenAI
from fastembed import LateInteractionTextEmbedding
from fastembed.postprocess import Muvera

# --- 1. Structured Schema for Neural-Symbolic Bridge ---
class QueryAnalysis(BaseModel):
    action: str = Field(description="Must be LIST, COUNT, SUM, or UNKNOWN")
    entity: str = Field(description="Must be customer, product, stock, or UNKNOWN")
    is_out_of_scope: bool = Field(description="True if query is not about data reports")
    hypotheses: List[str] = Field(description="3 semantic variations of the query")

class BISovereignEngine:
    def __init__(self, openai_api_key: str, kb_data: List[Dict[str, str]]):
        """
        kb_data format: [{"text": "...", "action": "LIST", "entity": "customer"}]
        """
        self.client = OpenAI(api_key=openai_api_key)
        self.kb_data = kb_data
        
        # 1. Initialize MUVERA (Late Interaction)
        self.model_name = "answerdotai/answerai-colbert-small-v1"
        self.model = LateInteractionTextEmbedding(model_name=self.model_name)
        
        self.muvera = Muvera.from_multivector_model(
            model=self.model,
            k_sim=6,     # Tokens to consider for interaction
            dim_proj=32, # Projection dimension
            r_reps=20    # Number of representatives
        )
        
        # 2. Threshold Configuration
        self.confidence_gap_threshold = 0.25  # Winner must beat runner-up by 25% relative margin
        self.min_raw_score_floor = 500.0      # Ignore matches that are too weak (noise)
        
        # 3. Pre-index Knowledge Base
        print("Indexing KB with MUVERA...")
        kb_texts = [item['text'] for item in self.kb_data]
        kb_embs = list(self.model.embed(kb_texts))
        # We store RAW vectors (not normalized) to preserve 'Intensity'
        self.kb_vectors = [self.muvera.process_document(emb) for emb in kb_embs]

    def _relative_scaling(self, scores: np.ndarray) -> np.ndarray:
        """Scales scores to 0-1 range based ONLY on the current result set."""
        s_min, s_max = scores.min(), scores.max()
        if s_max == s_min:
            return np.ones_like(scores)
        return (scores - s_min) / (s_max - s_min)

    def _get_raw_muvera_scores(self, query: str) -> np.ndarray:
        """Retrieves raw dot product scores for a single query variation."""
        query_emb = list(self.model.query_embed([query]))[0]
        query_vec = self.muvera.process_document(query_emb)
        return np.array([np.dot(query_vec, kb_vec) for kb_vec in self.kb_vectors])

    def _distill_and_expand(self, user_query: str) -> QueryAnalysis:
        """LLM Step: Intent extraction + Triple Semantic Spanning."""
        prompt = f"""Analyze the BI request: '{user_query}'
        Data Catalog Entities: [customer, product, stock, UNKNOWN]
        Allowed Actions: [LIST, COUNT, SUM, UNKNOWN]

        Task:
        1. Identify the 'action' and 'entity'. Use 'UNKNOWN' if not found.
        2. Set is_out_of_scope=True if the query is unrelated to the catalog 
        3. Provide 3 distinct semantic variations (hypotheses) that could match a formal report title.
        """
        completion = self.client.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are a BI Intent Parser."}, 
                      {"role": "user", "content": prompt}],
            response_format=QueryAnalysis,
        )
        return completion.choices[0].message.parsed

    def search(self, user_query: str) -> Dict[str, Any]:
        # PHASE 1: Triple Expansion & Logic Extraction
        analysis = self._distill_and_expand(user_query)
        
        if analysis.is_out_of_scope or analysis.action == "UNKNOWN" or analysis.entity == "UNKNOWN":
            return {"status": "OUT_OF_SCOPE", "match": None, "score": 0.0}

        # PHASE 2: Spanning Retrieval (Intensity-aware)
        # We find the best raw score across all 3 rewrites for each KB entry
        master_raw_scores = np.zeros(len(self.kb_data))
        for hyp in analysis.hypotheses:
            hyp_scores = self._get_raw_muvera_scores(hyp)
            master_raw_scores = np.maximum(master_raw_scores, hyp_scores)

        # PHASE 3: Local Sharpening (Relative Scaling)
        # This makes the 1.3x multipliers much more powerful
        scaled_scores = self._relative_scaling(master_raw_scores)

        # PHASE 4: Applying the Logic Gate
        final_rankings = []
        for i, scaled_score in enumerate(scaled_scores):
            # Logic Multipliers (The "Instructional" weight)
            action_mult = 1.3 if analysis.action == self.kb_data[i]['action'] else 0.4
            entity_mult = 1.1 if analysis.entity == self.kb_data[i]['entity'] else 0.2
            
            final_score = scaled_score * action_mult * entity_mult
            final_rankings.append({
                "item": self.kb_data[i],
                "score": final_score,
                "raw": master_raw_scores[i]
            })

        # PHASE 5: The "Gap" Decision
        final_rankings.sort(key=lambda x: x['score'], reverse=True)
        winner = final_rankings[0]
        runner_up = final_rankings[1] if len(final_rankings) > 1 else None

        # Guardrail 1: Minimal Intensity Floor
        if winner['raw'] < self.min_raw_score_floor:
            return {"status": "OUT_OF_SCOPE", "match": None, "score": winner['score']}

        # Guardrail 2: The Confidence Gap Analysis
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

# --- Usage Example ---
kb = [
    {"text": "How many customers do I have", "action": "COUNT", "entity": "customer"},
    {"text": "How many products do I have", "action": "COUNT", "entity": "product"},
    {"text": "How many products are in stock", "action": "COUNT", "entity": "stock"},
    {"text": "List all my customers", "action": "LIST", "entity": "customer"},
]

engine = BISovereignEngine(openai_api_key=os.getenv("OPENAI_API_KEY"), kb_data=kb)

# Test Shorthand Query
result = engine.search("customer count")
print(result)
result = engine.search("total locations")
print(result)
