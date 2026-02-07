import os
from dotenv import load_dotenv

load_dotenv()

# LLM & Embedding - AWS Bedrock Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
LLM_MODEL = "anthropic.claude-3-haiku-20240307-v1:0" 
LLM_TEMPERATURE = 0.0

# If no AWS credentials, fall back to local Ollama.
# Set USE_LOCAL_LLM=1 or true to force Ollama even when AWS vars are set (e.g. expired token).
_force_local = os.getenv("USE_LOCAL_LLM", "").strip().lower() in ("1", "true", "yes")
USE_LOCAL = _force_local or (
    AWS_ACCESS_KEY_ID is None or (isinstance(AWS_ACCESS_KEY_ID, str) and AWS_ACCESS_KEY_ID.strip() == "") or
    AWS_SECRET_ACCESS_KEY is None or (isinstance(AWS_SECRET_ACCESS_KEY, str) and AWS_SECRET_ACCESS_KEY.strip() == "")
)


# Retrieval
TOP_K = 10  # Increased to 10 for better recall
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "saudi_billing_codes"

# REMOVED: AUTO_ACCEPT_THRESHOLD — replaced by enriched retrieval and post-LLM validation

# --- V2 Redesign Settings ---
# Two-pass retrieval
TOP_K_GENERAL = 15          # Candidates from general (unfiltered) search
TOP_K_FILTERED = 15         # Candidates from specialty-filtered search
MAX_CANDIDATES_TO_LLM = 20  # Max merged candidates sent to the LLM

# Query expansion
ENABLE_QUERY_EXPANSION = True   # Toggle abbreviation expansion
ENABLE_SPECIALTY_FILTER = True  # Toggle two-pass filtered retrieval

# Post-LLM validation
ENABLE_POST_LLM_VALIDATION = True  # Toggle post-LLM safety net

# Paths
REFERENCE_FILE = "./data/reference/Saudi Billing Codes_Sample.xlsx"
INPUT_DIR = "./data/input/"
OUTPUT_DIR = "./data/output/"
