import os
from dotenv import load_dotenv

load_dotenv()

# LLM & Embedding
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = "gpt-4o-mini"  # Fast, cheap, reliable
LLM_TEMPERATURE = 0.0

# If no OpenAI key, fall back to local Ollama
USE_LOCAL = OPENAI_API_KEY is None or OPENAI_API_KEY.strip() == ""


# Retrieval
TOP_K = 10
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "saudi_billing_codes"

# Auto-accept threshold: if vector similarity > this, skip LLM call
# Lower = faster (more auto-accepts), Higher = more accurate (more LLM calls)
AUTO_ACCEPT_THRESHOLD = 0.85  # Lowered from 0.95 for speed

# Paths
REFERENCE_FILE = "./data/reference/Saudi Billing Codes_Sample.xlsx"
INPUT_DIR = "./data/input/"
OUTPUT_DIR = "./data/output/"
