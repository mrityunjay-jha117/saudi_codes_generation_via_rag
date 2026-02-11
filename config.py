import os
from dotenv import load_dotenv

load_dotenv()

# LLM & Embedding - AWS Bedrock Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
LLM_MODEL = "anthropic.claude-3-5-sonnet-20240620-v1:0" 
LLM_TEMPERATURE = 0.0

# Normalization Step Configuration
NORMALIZATION_LLM_MODEL = "anthropic.claude-3-haiku-20240307-v1:0" # Cheaper/Faster model for Step 1
NORMALIZATION_LLM_TEMPERATURE = 0.0


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "combinedcode"
PINECONE_INFERENCE_MODEL = "multilingual-e5-large"
PINECONE_NAMESPACES = ["sbs_v2", "gmdn_v2", "gtin_v2"]

# Rate Limiting & Performance
MATCH_BATCH_DELAY = 2.0  # Seconds (reduced for speed)
ASYNC_CONCURRENCY = 5   # Parallel requests (increased for speed)

# --- V2 Redesign Settings ---
# Routing & Retrieval
# Routing & Retrieval
ROUTING_PRIMARY_K = 20     # Candidates from predicted namespace (Increased to avoid missing drugs)
ROUTING_SECONDARY_K = 5     # Candidates from other namespaces (fallback)
MAX_CANDIDATES_TO_LLM = 15  # Context window limit (Increased to see more options)



# Post-LLM validation
ENABLE_POST_LLM_VALIDATION = True  # Toggle post-LLM safety net

# Paths
INPUT_DIR = "./data/input/"
OUTPUT_DIR = "./data/output/"
