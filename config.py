import os
from dotenv import load_dotenv

load_dotenv()

# LLM & Embedding - AWS Bedrock Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
LLM_MODEL = "anthropic.claude-3-haiku-20240307-v1:0" 
LLM_TEMPERATURE = 0.0

# If no AWS credentials, fall back to local Ollama
USE_LOCAL = (AWS_ACCESS_KEY_ID is None or AWS_ACCESS_KEY_ID.strip() == "" or 
             AWS_SECRET_ACCESS_KEY is None or AWS_SECRET_ACCESS_KEY.strip() == "")


# Retrieval
TOP_K = 10  # Increased to 10 for better recall
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "saudi_billing_codes"

# Auto-accept threshold: if vector similarity > this, skip LLM call
# Lower = faster (more auto-accepts), Higher = more accurate (more LLM calls)
# Optimized: 0.90 for good balance between speed and accuracy
AUTO_ACCEPT_THRESHOLD = 0.90 

# Paths
REFERENCE_FILE = "./data/reference/Saudi Billing Codes_Sample.xlsx"
INPUT_DIR = "./data/input/"
OUTPUT_DIR = "./data/output/"
