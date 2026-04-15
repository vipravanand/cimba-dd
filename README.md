# HOW TO RUN
## Quick Start (uv)
uv sync <br> 
OPENAI_API_KEY="your_api_key_here" uv run python semantic_search.py <br> 
<br>
For tests:<br> 
OPENAI_API_KEY="your_api_key_here" uv run pytest test_script.py -v <br> 


#  Submission FAQs

Q. Key Components of this semantic search pipeline <br> 
- KB Ingestion - Pre-computed Muvera embeddings for the knowledge-base <br> 
- KB enrichment - metadata enrichment of the existing KB with ACTION and ENTITY ( required in the rule engine for filtering out Out-of-Scope user queries)<br> 
- Query Expansion/Rewriting : Creating 3 version of the queries for improved query formation for better precision <br> 
-  Logic Gate: In the previous stage, the query metadata is extracted and compared against the kb metadata to deterministically eliminate out of scope queries.<br> 

Semantic Search <br>
-  Filter out out-of-scope queries based on deterministic logic <br> 	
- dot product between query-embedding and kb-embeddings ( embeddings generated using Muvera approach for token wise late interaction model for better results) to rank the best matching kb-candidates <br> 
- gap-out ( high score plus difference between score of top two candidate) strategy to classify High Confidence match and best answer selection <br> 

Q. Why use  Query Expansion/Rewriting <br> 
 Query Expansion/ Rewriting gives better performance against shorthand /ambiguous user generated query that may trip in vector search.<br>  

Q: Why MUVERA (Late Interaction/ColBERT) instead of standard Bi-Encoders <br> 
The BI queries have both an intent ( How many , Compare, etc) and Entity ( Customers, Products). In the candidate encoders  during dev, muvera performed better than bi-encoders in making a clear distinction, biencoders tended to have a higher score for partial matches. <br> 
Also, using a bi-encoder would have necessitated a Reranker ( Cross Encoder ) stage as well for better precision. <br> 
The idea was to adopt a cleaner solution here using Muvera.<br> 
Q. Why not simply use a CrossEncoder directly instead of Muvera <br> 
Cross encoder was a choice I experimented with, the results were comparable in terms candidates picked, however it did not yield vehemently differing scores between great match vs poor match, as it is primarily a ranker <br> 
Also, while in the current corpus , Cross Encoder could be used directly because of a smaller number of candidates, , but typically in a production environment with a higher number of candidate documents,  - cross encoder may have to be paired with Bi encoders for first pass filter. Muvera is more single shot and keeps the system complexity lower. <br> 
How does it handle ambiguous queries or closely related outputs ? <br> 
The top two candidates are considered before the final answer and unless there is a vehement gap between the best score and second best score - that is the the difference in scores is narrow than the threshold, then the triaging is  classified as AMBIGUOUS. <br> 
This pipeline can be used for building an embedding fine tuning dataset as well. <br> 

### Key Challenges Faced
-> Calibrating the score to >0.95 or <0.5 - cosine similarity scores or relevance score did not land up in the such high ranges <br> 

# EVALS

Defined a Vetted KB corpus as following : 
- "How many customers do I have", 
- "How many products do I have", 
- "How many products are in stock", 
- List all my customers" 

## Test Results 

Query

How many customers?
SUCCESS
"How many customers do I have"
✅ 

list customers
SUCCESS
"List all my customers"
✅

stock levels
SUCCESS
"How many products are in stock"
✅

customer count
SUCCESS
"How many customers do I have"
✅ 

all my buyers
SUCCESS
"List all my customers"
✅ 

customers
SUCCESS
"List all my customers"
✅ 
generate report
AMBIGUOUS
None
✅ 

total products
SUCCESS
"How many products do I have"
❌ 

product information
AMBIGUOUS
None
❌ Fail


