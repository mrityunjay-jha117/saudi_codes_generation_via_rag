import os
from dotenv import load_dotenv

load_dotenv()

# LLM & Embedding
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LLM_MODEL = "gemini-2.0-flash"  # Using 2.0-flash (higher rate limits than 2.5-pro)
LLM_TEMPERATURE = 0.0

# If no Gemini key, fall back to local models
USE_LOCAL = GEMINI_API_KEY is None or GEMINI_API_KEY.strip() == ""


# Retrieval
TOP_K = 10
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "saudi_billing_codes"

# Auto-accept threshold: if vector similarity > this, skip LLM call
AUTO_ACCEPT_THRESHOLD = 0.95

# Paths
REFERENCE_FILE = "./data/reference/Saudi Billing Codes_Sample.xlsx"
INPUT_DIR = "./data/input/"
OUTPUT_DIR = "./data/output/"
