# CLAUDE.md — Saudi Billing Code Matcher (RAG POC)

## 1. GOAL

Build a RAG-powered tool that takes a spreadsheet of **free-text healthcare service descriptions** and matches each row to the best standardized Saudi billing code from three reference code systems: **SBS** (procedures), **GTIN** (drugs), **GMDN** (devices).

The approach:
1. Embed all reference code descriptions into a **ChromaDB** vector store
2. For each input service description, **retrieve** the top-K most similar reference codes
3. Send the candidates to an **LLM** which picks the best match and returns structured JSON
4. Write results back to an output Excel file

This is a **POC**. Keep it simple. Minimal UI (Streamlit). No over-engineering.

---

## 2. INPUT FILES

Two Excel files are provided. Their exact schemas are documented below. **Only use the fields explicitly listed. Ignore all other columns.**

### 2.1 Reference Data: `Saudi_Billing_Codes_Sample.xlsx`

This file has **3 sheets**, each representing a different code system:

#### Sheet: `SBS V3 Tabular List` (Medical Procedures)

| Field to Use | Type | Role |
|---|---|---|
| `SBS Code Hyphenated` | string | **OUTPUT** — the code to return (e.g., `39003-00-00`) |
| `Short Description` | string | **MATCH FIELD** — primary text to embed |
| `Long Description` | string | **MATCH FIELD** — secondary text to embed (often identical to Short) |

**Ignore all other columns** (SBS Code numeric, Chapter number, Chapter Name, Block Number, Block Name, Clinical Explanation, Guideline, Includes, Excludes).

**Data cleaning**: Drop rows where `SBS Code Hyphenated` is NaN/null. After cleaning: **9 rows**.

**Document construction for embedding**:
```
If Short Description == Long Description:
    text = "[SBS] {Short Description}"
Else:
    text = "[SBS] {Short Description}. {Long Description}"

metadata = {
    "code": row["SBS Code Hyphenated"],   # e.g., "39003-00-00"
    "system": "SBS",
    "description": row["Short Description"]
}
```

#### Sheet: `GTIN` (Pharmaceutical Drugs)

| Field to Use | Type | Role |
|---|---|---|
| `CODE` | integer | **OUTPUT** — the GTIN code to return (e.g., `6285074000864`) |
| `DISPLAY` | string | **MATCH FIELD** — brand/trade name (e.g., `BEPANTHEN`) |
| `INGREDIENTS` | string | **MATCH FIELD** — active ingredient (e.g., `DEXPANTHENOL`) |
| `STRENGTH` | string | **MATCH FIELD** — dosage strength (e.g., `5 G/ 100 G`) |

**Ignore all other columns** (CATEGORY1, OTHER_CODES_VALUE, PRICE, GRANULAR_UNIT, UNIT_TYPE, MANUFACTURER, REG_OWNER, DOSAGE_FORM, ROA, PACKAGE_TYPE, PACKAGE_SIZE, RELEASE_DATE, RECEIVED_DATE, OTHER_CODES_TYPE).

**Rows: 9**. No cleaning needed.

**Document construction**:
```
text = "[GTIN] {DISPLAY} - {INGREDIENTS} {STRENGTH}"

metadata = {
    "code": str(row["CODE"]),   # e.g., "6285074000864"
    "system": "GTIN",
    "description": row["DISPLAY"]
}
```

#### Sheet: `GMDN` (Medical Devices)

| Field to Use | Type | Role |
|---|---|---|
| `GMDN_termCode` | integer | **OUTPUT** — the term code to return (e.g., `38242`) |
| `termName` | string | **MATCH FIELD** — device name |
| `termDefinition` | string | **MATCH FIELD** — full definition text |

**Rows: 8**. No cleaning needed.

**Document construction**:
```
text = "[GMDN] {termName}. {termDefinition[:300]}"
# Truncate termDefinition to 300 chars max (some definitions are very long)

metadata = {
    "code": str(row["GMDN_termCode"]),   # e.g., "38242"
    "system": "GMDN",
    "description": row["termName"]
}
```

**Total documents to index: 9 (SBS) + 9 (GTIN) + 8 (GMDN) = 26 documents.**

---

### 2.2 Input Data: `Code_Mapping_Sheet_Sample.xlsx`

This file has **3 sheets** (Sheet1, Sheet2, Sheet3). All 3 sheets share the same schema:

| Field | Type | Usage |
|---|---|---|
| `Service Code` | string | Preserved in output unchanged. NOT used for matching. |
| `Service Description` | string | **THE INPUT** — this is the text we match against the vector DB |
| `NPHIES Code ` | empty | Ignore (note: has trailing space in column name) |
| `Description ` | empty | Ignore (note: has trailing space in column name) |
| `Other Code Value ` | empty | Ignore (note: has trailing space in column name) |

**Row counts**: Sheet1 = 5, Sheet2 = 159, Sheet3 = 1,742. **Total: 1,906 rows** to match.

**Only read `Service Code` and `Service Description` from the input. Drop the three empty NPHIES columns.**

---

## 3. OUTPUT FORMAT

For each input row, produce these columns in the output Excel:

| Column | Source |
|---|---|
| `Service Code` | Copied from input (unchanged) |
| `Service Description` | Copied from input (unchanged) |
| `Matched Code` | The code returned by the LLM: an SBS Code Hyphenated, GTIN CODE, or GMDN termCode |
| `Code System` | `SBS`, `GTIN`, or `GMDN` |
| `Matched Description` | Official description from the reference data |
| `Confidence` | `HIGH`, `MEDIUM`, `LOW`, or `NONE` — set by the LLM |
| `Reasoning` | 1-sentence explanation from the LLM |

The output Excel must preserve the original 3-sheet structure (Sheet1, Sheet2, Sheet3).

---

## 4. ARCHITECTURE

```
Phase 1 (one-time): Reference Excel → Document Builder → Embedding Model → ChromaDB

Phase 2 (per row):   Service Description → ChromaDB top-K retrieval
                                          → LLM prompt with candidates
                                          → Structured JSON response
                                          → Output Excel row
```

### Tech Stack

| Component | Choice |
|---|---|
| Language | Python 3.11+ |
| Vector Store | ChromaDB (persistent, file-based) |
| Embedding | OpenAI `text-embedding-3-small` (primary) OR `all-MiniLM-L6-v2` via sentence-transformers (free fallback) |
| LLM | OpenAI `gpt-4o-mini` (primary) OR local Ollama (free fallback) |
| Orchestration | LangChain |
| Excel I/O | pandas + openpyxl |
| UI | Streamlit (single page) |

---

## 5. PROJECT STRUCTURE

```
saudi-code-matcher/
├── .env                    # OPENAI_API_KEY=sk-...
├── requirements.txt
├── config.py               # All settings in one place
├── ingest.py               # Read reference Excel → build ChromaDB
├── prompts.py              # LLM prompt template + candidate formatter
├── matcher.py              # RAG engine: retrieve + LLM + parse
├── app.py                  # Streamlit UI
├── test_matcher.py         # Tests
├── data/
│   ├── reference/          # Place Saudi_Billing_Codes_Sample.xlsx here
│   ├── input/              # Place Code_Mapping_Sheet_Sample.xlsx here
│   └── output/             # Generated result files go here
└── chroma_db/              # Auto-created by ChromaDB
```

**That's it. 6 Python files + 1 config + 1 test file. Do not create additional modules, packages, or abstractions.**

---

## 6. IMPLEMENTATION — FILE BY FILE

Build these files in EXACTLY this order. Each file depends on the ones before it.

---

### 6.1 `requirements.txt`

```
langchain>=0.3.0
langchain-openai>=0.2.0
langchain-community>=0.3.0
chromadb>=0.5.0
pandas>=2.2.0
openpyxl>=3.1.2
streamlit>=1.38.0
sentence-transformers>=3.0.0
python-dotenv>=1.0.0
pytest>=8.3.0
```

---

### 6.2 `config.py`

```python
import os
from dotenv import load_dotenv

load_dotenv()

# LLM & Embedding
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.0

# If no OpenAI key, fall back to local models
USE_LOCAL = OPENAI_API_KEY is None or OPENAI_API_KEY.strip() == ""

# Retrieval
TOP_K = 10
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "saudi_billing_codes"

# Auto-accept threshold: if vector similarity > this, skip LLM call
AUTO_ACCEPT_THRESHOLD = 0.95

# Paths
REFERENCE_FILE = "./data/reference/Saudi_Billing_Codes_Sample.xlsx"
INPUT_DIR = "./data/input/"
OUTPUT_DIR = "./data/output/"
```

**CRITICAL**: The `USE_LOCAL` flag must automatically detect whether an OpenAI key is present. If not, the system must fall back to `all-MiniLM-L6-v2` for embeddings and local Ollama for LLM without any code changes.

---

### 6.3 `ingest.py`

This file does THREE things:
1. Reads the reference Excel file
2. Builds LangChain `Document` objects with the exact text/metadata formats from Section 2.1
3. Embeds them into ChromaDB

**Critical rules**:
- Drop SBS rows where `SBS Code Hyphenated` is NaN
- Convert GTIN `CODE` (integer) to string in metadata
- Convert GMDN `GMDN_termCode` (integer) to string in metadata
- Truncate GMDN `termDefinition` to 300 characters
- Prefix each document text with `[SBS]`, `[GTIN]`, or `[GMDN]`
- Use `str(value).strip()` on all text fields to handle NaN and whitespace
- Handle the `USE_LOCAL` flag: if True, use `HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")` instead of `OpenAIEmbeddings`
- **Delete existing ChromaDB collection before re-indexing** to avoid duplicates on re-run
- Print the count of documents ingested per system (SBS: X, GTIN: X, GMDN: X, Total: X)

**When run as `__main__`**: Execute the full pipeline and print sample documents.

**Verification**: After indexing, perform a test query: `vector_store.similarity_search("puncture", k=3)` and print the results. The top result should be an SBS puncture code.

---

### 6.4 `prompts.py`

Contains the LLM prompt template and a helper function to format retrieved candidates.

**The prompt template** (this is the most important piece of the system):

```
You are a Saudi healthcare billing code expert specialized in NPHIES compliance.

Your task: Match the input service description to the BEST candidate code from the retrieved list below.

## Code Systems
- SBS: Saudi Billing System codes for medical/surgical PROCEDURES
- GTIN: Global Trade Item Numbers for PHARMACEUTICAL DRUGS
- GMDN: Global Medical Device Nomenclature for DEVICES and EQUIPMENT

## Input Service Description
{service_description}

## Retrieved Candidate Codes
{candidates}

## Matching Rules
1. Analyze the clinical/medical meaning of the input, not just keyword overlap.
2. SBS codes are for procedures/surgeries/examinations/consultations.
3. GTIN codes are for drugs — match on brand name, generic ingredient name, or strength.
4. GMDN codes are for devices, IVD kits, reagents, and medical equipment.
5. If the input is a procedure but no SBS code fits well, try GTIN and GMDN candidates.
6. If NONE of the candidates are a reasonable clinical match, set matched_code to null.

## Confidence Guide
- HIGH: You are very confident this is the correct code (exact or near-exact match)
- MEDIUM: This is likely correct but there is some ambiguity
- LOW: This is the best available option but you are not confident
- NONE: No candidate is a reasonable match

Respond with ONLY valid JSON, no markdown fences, no extra text:
{{"matched_code": "<code or null>", "code_system": "<SBS|GTIN|GMDN|null>", "matched_description": "<official description from candidate or null>", "confidence": "<HIGH|MEDIUM|LOW|NONE>", "reasoning": "<one sentence>"}}
```

**The `format_candidates` function**:
- Takes a list of retrieved LangChain documents
- Returns a numbered string like:
```
1. [SBS] Code: 39003-00-00 | Cisternal puncture
2. [SBS] Code: 39006-00-00 | Ventricular puncture
3. [GTIN] Code: 6285074000864 | BEPANTHEN - DEXPANTHENOL 5 G/ 100 G
```
- Format: `{i}. [{system}] Code: {code} | {page_content stripped of prefix}`

---

### 6.5 `matcher.py`

The core RAG engine. Contains a `CodeMatcher` class with:

#### `__init__(self)`
- Load ChromaDB vector store from persistent directory
- Initialize embedding model (OpenAI or local based on `USE_LOCAL`)
- Initialize LLM (OpenAI ChatOpenAI or Ollama based on `USE_LOCAL`)

#### `match_single(self, service_description: str) -> dict`
1. Query ChromaDB with `similarity_search_with_relevance_scores(service_description, k=TOP_K)`
2. **Auto-accept optimization**: If the top result's relevance score > `AUTO_ACCEPT_THRESHOLD`, skip the LLM call entirely and return:
   ```python
   {
       "matched_code": top_doc.metadata["code"],
       "code_system": top_doc.metadata["system"],
       "matched_description": top_doc.metadata["description"],
       "confidence": "HIGH",
       "reasoning": f"Auto-matched (similarity={score:.3f})"
   }
   ```
3. Otherwise, format the candidates using `format_candidates()`, build the prompt, call the LLM
4. Parse the LLM's JSON response
5. **JSON parse fallback**: If parsing fails (LLM returned markdown fences or extra text), try stripping ```json and ``` fences and re-parsing. If still fails, use the top retrieval result with confidence=LOW
6. Return the result dict

#### `match_batch(self, input_excel_path: str, output_excel_path: str) -> list`
1. Read the input Excel file with `pd.ExcelFile`
2. For each sheet:
   - Read only `Service Code` and `Service Description` columns
   - Iterate through rows, call `match_single()` for each
   - Print progress: `[Sheet1] [3/5] Dental Examination by Specialist...`
   - Collect results into new columns
3. Write all sheets to output Excel using `pd.ExcelWriter` with `openpyxl` engine
4. Return summary statistics: `{sheet_name: {total, high, medium, low, none}}`

**Critical**: 
- Handle `NaN` or empty Service Description gracefully — return confidence=NONE
- Convert all metadata code values to strings (GTIN codes are large integers)
- Add rate limiting: `time.sleep(0.1)` between LLM calls to avoid API throttling
- Print a summary table at the end showing match statistics per sheet

#### `match_batch_async(self, input_excel_path: str, output_excel_path: str) -> list`
An optimized version using `asyncio` to send LLM calls concurrently:
- Use `asyncio.Semaphore(10)` to limit concurrent requests to 10
- Use `llm.ainvoke()` for async LLM calls
- Still retrieve from ChromaDB synchronously (it's fast enough)
- Print progress updates every 50 rows

---

### 6.6 `app.py`

Streamlit UI. **Single page. Minimal.** Three sections:

#### Sidebar: Reference Data
- File uploader for the reference Excel
- "Index Reference Data" button
- On click: call `ingest.py`'s `build_documents()` and `create_vector_store()`
- Show success message with document count
- Also show a status indicator: "Vector DB Ready ✓" or "No Vector DB — upload reference data first"

#### Main Area: Code Matching
- File uploader for the mapping sheet Excel
- "Match Codes" button (primary, large)
- On click:
  - Show a progress bar
  - Call `matcher.match_batch()`
  - Show summary stats per sheet (table with HIGH/MEDIUM/LOW/NONE counts)
  - Show preview of first 10 rows per sheet using `st.dataframe()`
  - Show a download button for the output Excel

#### Single Test Match (below main)
- A text input field: "Test a single description"
- On submit: call `matcher.match_single()` and display the JSON result
- This lets users quickly test individual descriptions before running the full batch

**Critical Streamlit rules**:
- Use `st.set_page_config(page_title="Saudi Code Matcher", layout="wide")`
- Use `tempfile.NamedTemporaryFile` for handling uploaded files
- Use `st.spinner()` during processing
- Use `st.cache_resource` for the CodeMatcher instance to avoid reinitializing on every interaction
- Create `data/output/` directory if it doesn't exist using `os.makedirs(exist_ok=True)`

---

### 6.7 `test_matcher.py`

Tests using pytest. **Must be runnable with `pytest test_matcher.py -v`**.

#### Test: `test_build_documents`
- Call `build_documents()` with the reference file
- Assert total document count is 26 (9 SBS + 9 GTIN + 8 GMDN)
- Assert all docs have `metadata["code"]` and `metadata["system"]`
- Assert no document has NaN in `page_content`
- Assert SBS docs start with `[SBS]`, GTIN with `[GTIN]`, GMDN with `[GMDN]`

#### Test: `test_document_metadata_types`
- All `metadata["code"]` values are strings (not int or float)
- All `metadata["system"]` values are one of `{"SBS", "GTIN", "GMDN"}`

#### Test: `test_format_candidates`
- Create mock documents, pass to `format_candidates()`
- Assert output is a numbered list with correct format
- Assert each line contains the code and system tag

#### Test: `test_vector_store_retrieval` (integration test, needs ChromaDB populated)
- Query "puncture" → top result should be SBS with code containing "39" prefix
- Query "BEPANTHEN" → top result should be GTIN with code starting with "628"
- Query "vitamin D3" → top result should be GMDN with code "38242" or "38243"
- Mark this test with `@pytest.mark.integration` so it can be skipped if vector DB isn't populated

#### Test: `test_match_single_json_fallback`
- Mock the LLM to return invalid JSON
- Assert `match_single()` falls back to the top retrieval result with confidence=LOW

---

## 7. EXECUTION ORDER

When building this project, follow this exact sequence:

```
Step 1: Create project structure (folders, .env, requirements.txt)
Step 2: pip install -r requirements.txt
Step 3: Write config.py
Step 4: Write ingest.py → run it → verify "Indexed 26 documents" prints
Step 5: Write prompts.py
Step 6: Write matcher.py → test with: python -c "from matcher import CodeMatcher; m = CodeMatcher(); print(m.match_single('Cisternal puncture'))"
Step 7: Write test_matcher.py → run pytest
Step 8: Write app.py → run: streamlit run app.py
Step 9: Run full batch match and verify output Excel
```

---

## 8. VERIFICATION CHECKLIST

After building, verify ALL of these:

### 8.1 Ingest Verification
- [ ] `python ingest.py` prints `SBS: 9, GTIN: 9, GMDN: 8, Total: 26`
- [ ] `chroma_db/` directory is created and non-empty
- [ ] Test query "puncture" returns SBS results
- [ ] Test query "BEPANTHEN" returns GTIN results
- [ ] Test query "vitamin D3" returns GMDN results
- [ ] No NaN values in any document text or metadata

### 8.2 Matcher Verification
- [ ] `match_single("Cisternal puncture")` returns code `39003-00-00`, system `SBS`, confidence `HIGH`
- [ ] `match_single("BEPANTHEN")` returns a GTIN code starting with `628507400`
- [ ] `match_single("vitamin D3 IVD calibrator")` returns code `38242`, system `GMDN`
- [ ] `match_single("completely random nonsense xyz")` returns confidence `LOW` or `NONE`
- [ ] JSON parse fallback works (test by temporarily making prompt return bad JSON)

### 8.3 Batch Verification
- [ ] Output Excel has 3 sheets matching input sheet names
- [ ] Sheet1 has 5 rows, Sheet2 has 159 rows, Sheet3 has 1,742 rows
- [ ] All rows have `Matched Code`, `Code System`, `Confidence`, `Reasoning` columns
- [ ] No `NaN` in Service Code or Service Description columns
- [ ] Original Service Code and Service Description values are preserved unchanged
- [ ] Summary stats print at the end

### 8.4 Streamlit Verification
- [ ] Page loads at `localhost:8501`
- [ ] Reference file upload + indexing works
- [ ] Mapping file upload + matching works
- [ ] Download button produces valid Excel
- [ ] Single test match input works

---

## 9. IMPORTANT CONSTRAINTS

1. **Only 6 Python files**: config.py, ingest.py, prompts.py, matcher.py, app.py, test_matcher.py. Do not create __init__.py files, additional packages, or abstractions.

2. **Only the listed fields from Excel**: Do not read, process, or store any fields not explicitly listed in Section 2.

3. **The [SBS]/[GTIN]/[GMDN] prefix tags** in document text are mandatory. They help the embedding model distinguish between code systems and help the LLM understand what type of code each candidate is.

4. **String conversion for codes**: GTIN CODE is an integer (e.g., 6285074000864). GMDN termCode is an integer (e.g., 38242). Both MUST be converted to strings with `str()` before storing in metadata. SBS Code Hyphenated is already a string.

5. **ChromaDB persistence**: Always use `persist_directory` so the vector store survives between runs. Delete and recreate the collection in `ingest.py` to handle re-indexing cleanly.

6. **LLM temperature must be 0.0** for deterministic, reproducible matching.

7. **Graceful fallback**: If OpenAI key is missing, the system must work with local models. If the LLM returns unparseable JSON, fall back to the top vector retrieval result.

8. **The prompt in prompts.py is the most important piece of code**. Do not simplify or shorten it. The matching rules and confidence guide are critical for match quality.

9. **Sheet names in output must match input exactly**: Sheet1, Sheet2, Sheet3.

10. **Rate limiting**: Add `time.sleep(0.1)` between LLM API calls in synchronous batch mode. In async mode, use `Semaphore(10)`.

---

## 10. SAMPLE DATA FOR QUICK TESTING

Use these for quick verification without running the full 1,906-row batch:

| Input | Expected Code | Expected System | Expected Confidence |
|---|---|---|---|
| Cisternal puncture | 39003-00-00 | SBS | HIGH |
| Neuroendoscopy | 40903-00-00 | SBS | HIGH |
| Ventricular puncture | 39006-00-00 | SBS | HIGH |
| BEPANTHEN | 6285074000864 or 6285074000857 | GTIN | HIGH or MEDIUM |
| ZOVIRAX | 6285096000200 or 6285096000194 | GTIN | HIGH or MEDIUM |
| Montelukast 4mg | 6285147000012 | GTIN | HIGH |
| Salbutamol tablets | 6251159026081 | GTIN | MEDIUM or HIGH |
| vitamin D3 IVD calibrator | 38242 | GMDN | HIGH |
| Beta-D-glucan IVD control | 45703 | GMDN | HIGH |
| Random procedure not in ref data | null | null | NONE |

---

## 11. COMMON PITFALLS TO AVOID

1. **Do NOT strip the `[SBS]`/`[GTIN]`/`[GMDN]` prefix** when displaying candidates to the LLM. The prefix helps the LLM understand which code system each candidate belongs to.

2. **Do NOT use `similarity_search` for auto-accept logic.** You need `similarity_search_with_relevance_scores` to get the actual similarity score for the threshold check.

3. **Do NOT hardcode the reference file path in ingest.py.** Read it from `config.REFERENCE_FILE` so the Streamlit app can pass a different path.

4. **ChromaDB `from_documents` will ADD to an existing collection, creating duplicates on re-run.** Always delete the existing collection first:
   ```python
   client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
   try:
       client.delete_collection(config.COLLECTION_NAME)
   except:
       pass
   ```

5. **The input Excel has trailing spaces in column names**: `'NPHIES Code '`, `'Description '`, `'Other Code Value '`. Either strip column names with `df.columns = df.columns.str.strip()` or simply select only the two columns you need: `df[["Service Code", "Service Description"]]`.

6. **GTIN CODE values are large integers** (e.g., 6285074000864). If you read them with pandas and write them back, they may be converted to floats (6.285074e+12). Always cast to `str(int(value))` immediately after reading.

7. **Some Service Description values may be NaN** (empty rows at the end of sheets). Check for this and skip them.

8. **The LLM may wrap its JSON response in markdown code fences** like ` ```json ... ``` `. The JSON parser must strip these before parsing.

---

## 12. FUTURE ENHANCEMENTS (NOT IN POC SCOPE)

Do not implement these now. Documented for awareness only:

- **Async batch processing** with concurrent LLM calls (optimization)
- **Batch multiple rows per LLM call** (send 5-10 descriptions per prompt)
- **Human review queue** for LOW confidence matches
- **Caching** of matched descriptions to avoid re-calling LLM for duplicates
- **Production vector store** (PostgreSQL + pgvector instead of ChromaDB)
- **Re-ranking** with cross-encoder model before LLM call
- **Full reference data** (production will have 10,000+ codes instead of 26 samples)
