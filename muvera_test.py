from fastembed import LateInteractionTextEmbedding
from fastembed.postprocess import Muvera

from fastembed import LateInteractionTextEmbedding
print(LateInteractionTextEmbedding.list_supported_models())

model_name = "answerdotai/answerai-colbert-small-v1"
model = LateInteractionTextEmbedding(model_name=model_name)

# # Create MUVERA processor with recommended parameters
muvera = Muvera.from_multivector_model(
    model=model,
    k_sim=6,
    dim_proj=32,
    r_reps=20
)

kb_emb= model.embed(["How many customers do I have" , "How may products do I have", "How many products are in stock", "List all my customers"])
query_embed = model.query_embed(["How may products do I have"])
kb_muvera = [
muvera.process_document(emb) for emb in kb_emb
]
query_muvera = [
muvera.process_document(emb) for emb in query_embed
]

import numpy as np
score = [np.dot(query_muvera[0], kb_vec) for kb_vec in kb_muvera]
print(score)