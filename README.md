# Saudi Billing Code Matcher

A RAG-powered tool for automatically matching healthcare service descriptions to standardized Saudi billing codes.

## Overview

This project uses Retrieval-Augmented Generation (RAG) to match free-text healthcare service descriptions to standardized billing codes from three Saudi healthcare code systems:

- **SBS** (Saudi Billing System) - Medical procedures and surgeries
- **GTIN** (Global Trade Item Number) - Pharmaceutical drugs
- **GMDN** (Global Medical Device Nomenclature) - Medical devices and equipment

## Problem Statement

Healthcare providers in Saudi Arabia need to convert approximately 1,906 free-text service descriptions into standardized billing codes. Manual matching is time-consuming and error-prone. This tool automates the process using:

1. Vector similarity search to find candidate matches
2. LLM-based intelligent selection of the best match
3. Confidence scoring and reasoning for each match

## Architecture

### Two-Phase System

**Phase 1: Indexing (One-time setup)**

```
Reference Excel → Document Builder → Embedding Model → ChromaDB Vector Store
```

**Phase 2: Matching (Per query)**

```
Service Description → Vector Search (Top-K) → LLM Decision → Structured Output
```

### Technology Stack

| Component     | Technology                               | Purpose                         |
| ------------- | ---------------------------------------- | ------------------------------- |
| Language      | Python 3.11+                             | Core implementation             |
| Vector Store  | ChromaDB                                 | Persistent vector database      |
| Embeddings    | sentence-transformers (all-MiniLM-L6-v2) | Local text-to-vector conversion |
| LLM           | Google Gemini API or Ollama              | Intelligent match selection     |
| Orchestration | LangChain                                | RAG pipeline management         |
| Excel I/O     | pandas + openpyxl                        | Data processing                 |
| UI            | Streamlit                                | Web interface                   |

## Project Structure

```
saudi-code-matcher/
├── .env                    # API keys (GEMINI_API_KEY)
├── .gitignore             # Git ignore rules
├── requirements.txt       # Python dependencies
├── config.py              # Configuration settings
├── ingest.py              # Reference data indexing
├── prompts.py             # LLM prompt templates
├── matcher.py             # Core RAG matching engine
├── app.py                 # Streamlit web UI
├── test_matcher.py        # Unit tests
├── data/
│   ├── reference/         # Saudi_Billing_Codes_Sample.xlsx
│   ├── input/             # Code_Mapping_Sheet_Sample.xlsx
│   └── output/            # Generated results
├── chroma_db/             # Vector database (auto-generated)
└── docs/                  # Documentation
    ├── CLAUDE.md          # Original specification
    └── doc.md             # Detailed implementation guide
```

## Installation

### Prerequisites

- Python 3.11 or higher
- pip package manager
- (Optional) Ollama for local LLM

### Setup Steps

1. Clone or download the project:

```bash
git clone https://github.com/mrityunjay-jha117/saudi_codes_generation_via_rag
```

2. Create a virtual environment (recommended):

```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Configure API keys:

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_api_key_here
```

Or use local Ollama (no API key needed):

```bash
# Install Ollama from https://ollama.com/download
ollama pull gemma2:2b
```

Then set in `config.py`:

```python
USE_LOCAL = True
```

## Usage

### 1. Index Reference Data

First, build the vector database from reference codes:

```bash
python ingest.py
```

Expected output:

```
SBS: 9, GTIN: 9, GMDN: 8, Total: 26
Indexed 26 documents into ChromaDB at './chroma_db'
```

### 2. Run the Web Interface

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### 3. Using the Interface

**Sidebar: Reference Data**

- Upload `Saudi_Billing_Codes_Sample.xlsx`
- Click "Index Reference Data"

**Main Area: Code Matching**

- Upload `Code_Mapping_Sheet_Sample.xlsx`
- Click "Match Codes"
- Download results when complete

**Test Single Match**

- Enter a service description
- View matched code, confidence, and reasoning

## Configuration

Edit `config.py` to customize:

```python
# LLM Settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LLM_MODEL = "models/gemini-2.0-flash"
LLM_TEMPERATURE = 0.0

# Use local Ollama instead of Gemini
USE_LOCAL = False  # Set to True for Ollama

# Retrieval Settings
TOP_K = 10  # Number of candidates to retrieve
AUTO_ACCEPT_THRESHOLD = 0.95  # Skip LLM if similarity > 0.95

# Paths
REFERENCE_FILE = "./data/reference/Saudi Billing Codes_Sample.xlsx"
INPUT_DIR = "./data/input/"
OUTPUT_DIR = "./data/output/"
```

## How It Works

### Document Construction

Each reference code is converted to a text document with a system prefix:

**SBS Example:**

```
[SBS] Cisternal puncture. Cisternal puncture
Metadata: {code: "39003-00-00", system: "SBS", description: "Cisternal puncture"}
```

**GTIN Example:**

```
[GTIN] BEPANTHEN - DEXPANTHENOL 5 G/ 100 G
Metadata: {code: "6285074000864", system: "GTIN", description: "BEPANTHEN"}
```

**GMDN Example:**

```
[GMDN] 1,25-Dihydroxy vitamin D3 IVD, calibrator. A material which is used...
Metadata: {code: "38242", system: "GMDN", description: "1,25-Dihydroxy vitamin D3 IVD, calibrator"}
```

### Matching Process

1. **Embedding**: Convert input text to vector using sentence-transformers (local, no API call)
2. **Retrieval**: Search ChromaDB for top 10 similar documents (local, fast)
3. **LLM Decision**: Send candidates to LLM with prompt asking for best match
4. **Parsing**: Extract structured JSON response with code, confidence, and reasoning
5. **Fallback**: If LLM fails, use top vector result with LOW confidence

### Output Format

Each matched row contains:

| Column              | Description                      | Example                                    |
| ------------------- | -------------------------------- | ------------------------------------------ |
| Service Code        | Original code (unchanged)        | FMORT-0706                                 |
| Service Description | Original description (unchanged) | COSTOPLASTY                                |
| Matched Code        | Selected billing code            | 39703-03-00                                |
| Code System         | SBS, GTIN, or GMDN               | SBS                                        |
| Matched Description | Official description             | Aspiration of brain cyst                   |
| Confidence          | HIGH, MEDIUM, LOW, or NONE       | MEDIUM                                     |
| Reasoning           | LLM explanation                  | Costoplasty is a rib surgical procedure... |

## Troubleshooting

### Issue: 429 RESOURCE_EXHAUSTED Error

**Cause**: Gemini API quota exceeded

**Solutions**:

1. Wait for quota reset (1 minute for rate limit, 24 hours for daily limit)
2. Use a different API key
3. Switch to local Ollama (unlimited, free)

To use Ollama:

```bash
ollama pull gemma2:2b
```

In `config.py`:

```python
USE_LOCAL = True
```

### Issue: Embedding Model Download Slow

**Cause**: First run downloads sentence-transformers model (90MB)

**Solution**: Wait for download to complete. Model is cached for future use.

### Issue: ChromaDB Not Found

**Cause**: Vector database not created

**Solution**: Run indexing first:

```bash
python ingest.py
```

### Issue: Import Errors

**Cause**: Missing dependencies

**Solution**: Reinstall requirements:

```bash
pip install -r requirements.txt
```

## Testing

Run unit tests:

```bash
pytest test_matcher.py -v
```

Test single match from command line:

```python
from matcher import CodeMatcher
matcher = CodeMatcher()
result = matcher.match_single("Cisternal puncture")
print(result)
```

Expected output:

```json
{
  "matched_code": "39003-00-00",
  "code_system": "SBS",
  "matched_description": "Cisternal puncture",
  "confidence": "HIGH",
  "reasoning": "Exact match found in SBS codes"
}
```

## Performance

- **Indexing**: 26 documents in under 5 seconds
- **Single match**: 1-2 seconds (with LLM), 0.1 seconds (vector only)
- **Batch processing**: Approximately 30-60 seconds per 100 rows (depends on LLM speed)

## API Costs

### Using Gemini API (Free Tier)

- Embedding: Local (free)
- LLM calls: 15 requests/minute, 1,500 requests/day
- Cost for 1,906 rows: $0 (within free tier)

### Using Ollama (Local)

- All operations: Free, unlimited
- Requires: 4GB RAM, 2GB disk space

## Data Privacy

- **Local embeddings**: All text-to-vector conversion happens on your machine
- **Local vector store**: ChromaDB runs locally, no data sent to external servers
- **LLM calls**: Only candidate codes and input text sent to Gemini API (if using cloud LLM)
- **Ollama option**: Complete privacy, all processing local

## Limitations

- **Sample data**: Current reference data has only 26 codes (production will have thousands)
- **LLM accuracy**: Depends on quality of reference descriptions and LLM knowledge
- **Rate limits**: Free tier Gemini has strict limits (use Ollama for unlimited)
- **Language**: Currently optimized for English medical terminology

## Future Enhancements

- Async batch processing for faster throughput
- Caching of matched descriptions to avoid duplicate LLM calls
- Human review queue for LOW confidence matches
- Support for Arabic medical terminology
- Integration with production billing systems
- Advanced re-ranking with cross-encoder models

## License

Internal use only. Not for public distribution.

## Support

For issues or questions, refer to:

- `docs/CLAUDE.md` - Original specification
- `docs/doc.md` - Detailed implementation guide

## Version History

- **v2.0** (Feb 2026) - Migrated to google.genai SDK, added Ollama support
- **v1.0** (Feb 2026) - Initial implementation with OpenAI/Gemini support
