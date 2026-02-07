# INSTRUCTIONS.md — Complete System Redesign Implementation Guide

## PURPOSE

This document provides step-by-step implementation instructions to redesign the Saudi Billing Code Matcher from its current V1 architecture (which discards 80% of reference data) to a V2 architecture that leverages ALL 12 columns of SBS reference data, implements two-pass filtered retrieval, query expansion for abbreviations, and metadata-aware LLM prompting.

The redesign addresses the root cause of matching failures: the vector store currently indexes only Short Description and Long Description, ignoring Chapter Name, Block Name, Clinical Explanation, Includes, Excludes, and Guideline fields — all of which contain critical disambiguation signals.

---

## CURRENT STATE SUMMARY

### Files to Modify

| File | Lines | What Changes |
|---|---|---|
| `config.py` | 32 | Add new config constants for two-pass retrieval, query expansion |
| `ingest.py` | 328 | Complete rewrite of SBS document building to use all 12 columns |
| `prompts.py` | 129 | Replace prompt with metadata-aware V3 prompt, rewrite `format_candidates()` |
| `matcher.py` | 781 | Add two-pass retrieval, query expansion, post-LLM validation, fix auto-accept |
| `app.py` | 241 | Minor UI updates to reflect new capabilities |
| `test_matcher.py` | 246 | Update tests for enriched documents, new metadata fields, new retrieval |

### New File to Create

| File | Purpose |
|---|---|
| `query_expansion.py` | SBS-vocabulary-aware abbreviation expansion and specialty detection |

### Files That Stay the Same

| File | Why |
|---|---|
| `requirements.txt` | No new dependencies needed |
| `.gitignore` | No changes needed |

---

## PHASE 1: FOUNDATION — Enriched SBS Indexing (Priority: CRITICAL)

This is the single highest-impact change. Every subsequent improvement depends on this.

### Step 1.1: Update `config.py`

**File:** `config.py`
**What to do:** Add new configuration constants that the rest of the system will use.

Add the following constants AFTER the existing ones:

```python
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
```

Also remove or comment out the `AUTO_ACCEPT_THRESHOLD` line — the current auto-accept logic (text containment check) is fundamentally flawed and will be replaced:

```python
# REMOVED: AUTO_ACCEPT_THRESHOLD = 0.90  # Replaced by enriched retrieval
```

Keep `TOP_K = 10` for backward compatibility but it will no longer be the primary retrieval parameter.

---

### Step 1.2: Rewrite SBS Document Building in `ingest.py`

**File:** `ingest.py`
**What to do:** Replace the SBS section of `build_documents()` to use ALL 12 columns.

#### 1.2.1: Add a `clean_field()` helper at the top of the file (after imports)

This helper is used throughout ingestion to handle NaN, None, and empty string values consistently:

```python
def clean_field(value) -> str:
    """Clean a DataFrame cell value, returning empty string for null/nan."""
    if pd.isna(value):
        return ""
    s = str(value).strip()
    if s.lower() in ("nan", "none", ""):
        return ""
    return s
```

Place this right after the existing `get_embedding_function()` function (around line 48).

#### 1.2.2: Add a `build_enriched_sbs_document()` function

This function builds a single enriched Document from one SBS row. Place it after `clean_field()`:

```python
def build_enriched_sbs_document(row: pd.Series) -> Document | None:
    """
    Build a richly annotated Document from an SBS reference row.
    Uses ALL available columns: Specialty (Chapter), Category (Block),
    Clinical Explanation, Includes, Excludes, Guideline.
    Returns None for empty/invalid rows.
    """
    code = clean_field(row.get("SBS Code Hyphenated"))
    short = clean_field(row.get("Short Description"))

    if not code or not short:
        return None

    long_desc = clean_field(row.get("Long Description"))
    chapter = clean_field(row.get("Chapter Name"))
    block = clean_field(row.get("Block Name"))
    clinical = clean_field(row.get("Cliincal Explanation "))  # Note: typo + trailing space in source data
    includes = clean_field(row.get("Includes"))
    excludes = clean_field(row.get("Excludes"))
    guideline = clean_field(row.get("Guideline"))
    chapter_num = row.get("Chapter number")
    block_num = clean_field(row.get("Block Number"))

    # Build enriched page_content
    parts = [f"[SBS] Code: {code}"]
    if chapter:
        parts.append(f"Specialty: {chapter}")
    if block:
        parts.append(f"Category: {block}")
    parts.append(f"Description: {short}")
    if long_desc and long_desc != short:
        parts.append(f"Detail: {long_desc}")
    if clinical:
        parts.append(f"Clinical context: {clinical}")
    if includes:
        parts.append(f"Includes: {includes}")
    if excludes:
        parts.append(f"Excludes: {excludes}")
    if guideline:
        parts.append(f"Guideline: {guideline}")

    page_content = ". ".join(parts)

    # Get SBS Code (numeric) if available
    sbs_code_numeric = ""
    if "SBS Code" in row.index and pd.notna(row.get("SBS Code")):
        try:
            sbs_code_numeric = str(int(float(row["SBS Code"])))
        except (ValueError, TypeError):
            sbs_code_numeric = str(row["SBS Code"]).strip()

    # Build metadata for filtered retrieval
    metadata = {
        "code": code,
        "code_hyphenated": code,
        "code_numeric": sbs_code_numeric,
        "system": "SBS",
        "description": short,
        "chapter_name": chapter.upper().strip() if chapter else "",
        "chapter_number": float(chapter_num) if pd.notna(chapter_num) else -1,
        "block_name": block,
        "block_number": block_num,
        "has_excludes": bool(excludes),
        "has_includes": bool(includes),
        "has_clinical_explanation": bool(clinical),
    }

    return Document(page_content=page_content, metadata=metadata)
```

**IMPORTANT NOTE about column name:** The actual Excel column for Clinical Explanation has a typo AND a trailing space: `"Cliincal Explanation "`. You MUST match this exactly when reading from the DataFrame. Verify by printing `df_sbs.columns.tolist()` during development.

#### 1.2.3: Replace the SBS section in `build_documents()`

In the existing `build_documents()` function, find the SBS section (approximately lines 70-109) and replace it entirely:

**REMOVE** the current SBS loop (lines 75-109 approximately).

**REPLACE WITH:**

```python
    # ── SBS Documents (ENRICHED with all metadata) ──
    df_sbs = pd.read_excel(xls, sheet_name="SBS V3 Tabular List")
    df_sbs = df_sbs.dropna(subset=["SBS Code Hyphenated"])

    for _, row in df_sbs.iterrows():
        doc = build_enriched_sbs_document(row)
        if doc:
            docs.append(doc)
            counts["SBS"] += 1
```

**DO NOT** change the GTIN or GMDN sections — they remain as-is.

#### 1.2.4: Verify the change

After making this change, run `python ingest.py` and verify:

1. Document count is the same (9 SBS from sample data)
2. SBS documents now contain Specialty, Category, Clinical context, etc.
3. Print a sample SBS document to confirm enriched text:

```
[SBS] Code: 39703-03-00. Specialty: PROCEDURES ON NERVOUS SYSTEM.
Category: Cranial tap or puncture. Description: Aspiration of brain cyst.
Clinical context: Navigation-guided cyst aspiration...
Excludes: Drainage of infected cyst (39900-00-00 [8])
```

4. GTIN and GMDN documents are unchanged
5. ChromaDB stores the new metadata fields (check with `collection.get()`)

---

## PHASE 2: QUERY EXPANSION — Abbreviation Handling (Priority: CRITICAL)

### Step 2.1: Create `query_expansion.py`

**File:** `query_expansion.py` (NEW FILE)
**What to do:** Create a module for expanding customer abbreviations and detecting specialties.

This is a new file. It has two main components:

1. **`SBS_VOCABULARY_MAP`** — A dictionary mapping common abbreviations/shorthand to expanded SBS terminology
2. **`expand_query()`** — Function that expands abbreviations BEFORE vector search
3. **`detect_specialty()`** — Function that identifies the clinical specialty from a query for filtered retrieval

```python
"""
query_expansion.py - SBS-vocabulary-aware query expansion and specialty detection.

Maps customer shorthand, abbreviations, and common terms to formal SBS terminology.
Called BEFORE vector search to improve retrieval quality.
"""

import re


# ── Abbreviation → Expansion Map ──
# Keys: customer shorthand (lowercase)
# Values: expanded SBS terminology to append to the search query
SBS_VOCABULARY_MAP = {
    # Dental - Endodontic
    "rct":              "root canal treatment endodontic",
    "r.c.t.":           "root canal treatment endodontic",
    "r.c.t":            "root canal treatment endodontic",
    "re-rct":           "root canal retreatment endodontic revision",
    "re rct":           "root canal retreatment endodontic revision",
    "pulpotomy":        "pulp therapy endodontic",
    "pulpectomy":       "pulp removal endodontic",

    # Dental - Prosthodontic
    "pfm":              "porcelain fused to metal crown metallic substructure fixed prosthodontic",
    "pfm crown":        "porcelain fused to metal crown metallic substructure fixed prosthodontic",
    "zirconia crown":   "zirconia all-ceramic crown fixed prosthodontic",
    "e-max":            "lithium disilicate all-ceramic crown fixed prosthodontic",
    "emax":             "lithium disilicate all-ceramic crown fixed prosthodontic",
    "rpd":              "removable partial denture prosthodontic",
    "fpd":              "fixed partial denture bridge prosthodontic",

    # Dental - Restorative
    "composite":        "composite resin direct restoration restorative",
    "amalgam":          "amalgam metallic restoration direct restorative",
    "tooth colored":    "composite resin direct restoration restorative",
    "tooth-colored":    "composite resin direct restoration restorative",
    "tooth colour":     "composite resin direct restoration restorative",
    "filling":          "dental restoration direct restorative",
    "surface":          "dental restoration surface restorative",

    # Dental - Radiograph
    "opg":              "orthopantomogram panoramic dental radiograph imaging",
    "iopa":             "intraoral periapical radiograph dental imaging",
    "pa":               "periapical radiograph dental imaging",
    "bitewing":         "bitewing radiograph dental imaging",
    "cbct":             "cone beam computed tomography dental imaging",

    # Dental - General
    "extraction":       "tooth extraction dental surgery oral",
    "impaction":        "impacted tooth extraction surgical dental",
    "scaling":          "dental scaling prophylaxis periodontal cleaning",
    "tmj":              "temporomandibular joint",
    "gingival":         "gingival periodontal gum",
    "veneer":           "dental veneer laminate cosmetic",
    "implant":          "dental implant endosseous",
    "denture":          "dental prosthesis denture prosthodontic",

    # Medical - General
    "cbc":              "complete blood count hematology laboratory",
    "ecg":              "electrocardiogram cardiac heart diagnostic",
    "ekg":              "electrocardiogram cardiac heart diagnostic",
    "eeg":              "electroencephalogram brain neurological diagnostic",
    "mri":              "magnetic resonance imaging diagnostic radiology",
    "ct":               "computed tomography imaging diagnostic radiology",
    "ct scan":          "computed tomography imaging diagnostic radiology",
    "xray":             "radiograph x-ray imaging diagnostic",
    "x-ray":            "radiograph x-ray imaging diagnostic",
    "ent":              "ear nose throat otolaryngology",
    "cabg":             "coronary artery bypass graft cardiac surgery",

    # Medical - Procedures
    "lumbar puncture":  "spinal tap cranial puncture cerebrospinal fluid",
    "spinal tap":       "lumbar puncture cranial tap cerebrospinal fluid",

    # Pharmaceutical
    "tab":              "tablet oral dosage form",
    "cap":              "capsule oral dosage form",
    "inj":              "injection parenteral",
    "susp":             "suspension oral liquid dosage form",
    "syr":              "syrup oral liquid dosage form",
    "iv":               "intravenous injection parenteral",
    "im":               "intramuscular injection parenteral",
    "sc":               "subcutaneous injection parenteral",
}


def expand_query(description: str) -> str:
    """
    Expand customer shorthand using SBS-informed vocabulary.
    Called BEFORE vector search. The expansion is appended in parentheses
    so the original description is preserved for the LLM.

    Args:
        description: Raw service description from customer input.

    Returns:
        Expanded description string. If no expansions apply, returns original.
    """
    if not description or not description.strip():
        return description

    desc_lower = description.lower().strip()
    expansions = []

    for trigger, expansion in SBS_VOCABULARY_MAP.items():
        # Whole-word matching to avoid false positives
        # Escape dots in trigger for regex, then do word-boundary match
        escaped = re.escape(trigger)
        pattern = r'(?:^|\b|(?<=\s))' + escaped + r'(?:\b|(?=\s)|$)'

        if re.search(pattern, desc_lower):
            if expansion not in expansions:
                expansions.append(expansion)

    # Also handle dotted abbreviations: "R.C.T." → "RCT"
    desc_no_dots = desc_lower.replace(".", "")
    for trigger, expansion in SBS_VOCABULARY_MAP.items():
        clean_trigger = trigger.replace(".", "")
        if clean_trigger != trigger and len(clean_trigger) >= 2:  # Only for dotted triggers
            if re.search(r'\b' + re.escape(clean_trigger) + r'\b', desc_no_dots):
                if expansion not in expansions:
                    expansions.append(expansion)

    if expansions:
        return f"{description} ({'; '.join(expansions)})"
    return description


# ── Specialty Detection for Filtered Retrieval ──
# Maps keyword signals to SBS Chapter Names
# These MUST match the actual Chapter Name values in your SBS reference data

SPECIALTY_KEYWORDS = {
    "DENTAL PROCEDURES": [
        "tooth", "dental", "rct", "r.c.t", "crown", "filling",
        "extraction", "impaction", "root canal", "amalgam",
        "composite", "resin", "veneer", "gingival", "periodon",
        "endodon", "prosthod", "orthodon", "implant", "denture",
        "alveol", "pfm", "porcelain", "zirconi", "surface",
        "pulp", "caries", "cavity", "enamel", "dentin",
        "molar", "premolar", "incisor", "canine", "bicuspid",
        "mandib", "maxill", "oral", "scaling", "prophylaxis",
        "opg", "bitewing", "periapical", "cbct", "rpd", "fpd",
    ],
    "PROCEDURES ON NERVOUS SYSTEM": [
        "brain", "cranial", "spinal", "intracranial",
        "ventricular", "meninges", "cerebro", "neuro",
        "lumbar puncture", "spinal tap", "eeg",
    ],
    "PROCEDURES ON CARDIOVASCULAR SYSTEM": [
        "heart", "cardiac", "coronary", "valve", "cabg",
        "pacemaker", "stent", "angioplast", "bypass",
        "ecg", "ekg", "aortic", "mitral", "tricuspid",
    ],
    "PROCEDURES ON EYE AND ADNEXA": [
        "eye", "ophthalm", "retina", "cornea", "lens",
        "cataract", "glaucoma", "vitreous", "conjunctiv",
        "lacrimal", "orbit", "eyelid",
    ],
    "PROCEDURES ON EAR AND MASTOID": [
        "ear", "tympan", "mastoid", "cochlear",
        "audiol", "hearing", "otitis",
    ],
    "PROCEDURES ON RESPIRATORY SYSTEM": [
        "lung", "pulmonary", "bronch", "trachea",
        "thorac", "pleural", "respiratory", "nasal",
        "sinus", "laryn", "pharyn", "tonsil", "adenoid",
    ],
    "PROCEDURES ON DIGESTIVE SYSTEM": [
        "gastro", "stomach", "intestin", "colon",
        "liver", "hepat", "pancrea", "biliary",
        "gallbladder", "appendix", "esophag", "rectal",
        "hernia", "bowel", "abdomen",
    ],
    "PROCEDURES ON MUSCULOSKELETAL SYSTEM": [
        "bone", "joint", "fracture", "arthroplast",
        "orthop", "spinal fusion", "knee", "hip",
        "shoulder", "ankle", "wrist", "elbow",
        "tendon", "ligament", "cartilage", "arthroscop",
    ],
    "PROCEDURES ON SKIN": [
        "skin", "dermat", "wound", "laceration",
        "burn", "graft", "flap", "lesion", "cyst",
        "abscess", "wart", "mole", "biopsy skin",
    ],
    "PROCEDURES ON URINARY SYSTEM": [
        "kidney", "renal", "ureter", "bladder",
        "urethra", "nephro", "dialysis", "urolog",
        "cystoscop", "lithotrips",
    ],
    "PATHOLOGY PROCEDURES": [
        "patholog", "histolog", "cytolog",
        "blood test", "blood typing", "blood group",
        "haematol", "hematol", "laboratory", "lab test",
        "biopsy", "specimen",
    ],
    "DIAGNOSTIC IMAGING": [
        "imaging", "radiograph", "x-ray", "xray",
        "ultrasound", "mri", "ct scan", "fluoroscop",
        "mammogra", "angiogra", "tomograph",
    ],
}


def detect_specialty(query: str) -> str | None:
    """
    Detect the clinical specialty from a query string using keyword matching.
    Returns the SBS Chapter Name if a specialty is detected, None otherwise.

    The returned value can be used as a ChromaDB metadata filter on
    the 'chapter_name' field during retrieval.

    Args:
        query: The (possibly expanded) service description.

    Returns:
        SBS Chapter Name string or None.
    """
    q = query.lower()

    for chapter_name, keywords in SPECIALTY_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return chapter_name

    return None
```

**IMPORTANT:** The `SPECIALTY_KEYWORDS` dictionary keys MUST exactly match the `chapter_name` values stored in ChromaDB metadata (which come from the Excel file's "Chapter Name" column, uppercased). After indexing the full reference data, you should:

1. Run `collection.get()` to see all unique `chapter_name` values
2. Update the keys in `SPECIALTY_KEYWORDS` to match exactly

---

## PHASE 3: TWO-PASS RETRIEVAL — Filtered Search (Priority: HIGH)

### Step 3.1: Rewrite Retrieval in `matcher.py`

**File:** `matcher.py`
**What to do:** Replace the single `similarity_search()` call with a two-pass retrieval system.

#### 3.1.1: Add import for query expansion at the top of `matcher.py`

After the existing imports (around line 20), add:

```python
from query_expansion import expand_query, detect_specialty
```

#### 3.1.2: Remove the duplicate `SentenceTransformerEmbeddings` class

The `SentenceTransformerEmbeddings` class is defined identically in both `ingest.py` (lines 21-42) and `matcher.py` (lines 31-43). Remove the duplicate from `matcher.py` and instead import it:

In `matcher.py` `__init__()`, replace lines 28-44:

```python
# BEFORE (remove this):
from sentence_transformers import SentenceTransformer

class SentenceTransformerEmbeddings:
    ... (entire class definition)

self.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
```

```python
# AFTER (use this):
from ingest import SentenceTransformerEmbeddings
self.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
```

#### 3.1.3: Add a `retrieve_candidates()` method to `CodeMatcher`

Add this new method to the `CodeMatcher` class, after the existing `get_cache_stats()` method:

```python
def retrieve_candidates(self, service_description: str) -> list:
    """
    Two-pass retrieval: general similarity + specialty-filtered search.
    Merges and deduplicates results for maximum recall with precision.

    Pass 1: General semantic search across ALL codes (unfiltered)
    Pass 2: If a specialty is detected, search WITHIN that specialty only

    Args:
        service_description: The (possibly expanded) service description.

    Returns:
        Deduplicated list of candidate Documents.
    """
    # Pass 1: General search
    general_results = self.vector_store.similarity_search(
        service_description,
        k=config.TOP_K_GENERAL if hasattr(config, 'TOP_K_GENERAL') else config.TOP_K,
    )

    # Pass 2: Specialty-filtered search (if enabled and specialty detected)
    filtered_results = []
    if getattr(config, 'ENABLE_SPECIALTY_FILTER', False):
        specialty = detect_specialty(service_description)
        if specialty:
            try:
                filtered_results = self.vector_store.similarity_search(
                    service_description,
                    k=config.TOP_K_FILTERED if hasattr(config, 'TOP_K_FILTERED') else config.TOP_K,
                    filter={"chapter_name": specialty},
                )
            except Exception as e:
                # Filtered search may fail if no docs match the filter
                print(f"  Filtered search failed for specialty '{specialty}': {e}")

    # Merge and deduplicate (preserve order: general first, then filtered additions)
    seen_codes = set()
    merged = []

    for doc in general_results + filtered_results:
        code = doc.metadata.get("code", "")
        system = doc.metadata.get("system", "")
        key = f"{system}:{code}"

        if key not in seen_codes:
            seen_codes.add(key)
            merged.append(doc)

    max_candidates = getattr(config, 'MAX_CANDIDATES_TO_LLM', 20)
    return merged[:max_candidates]
```

#### 3.1.4: Rewrite `match_single()` to use new retrieval + query expansion

Replace the existing `match_single()` method. The key changes are:

1. **Remove the broken auto-accept logic** (text containment check)
2. **Add query expansion** before retrieval
3. **Use `retrieve_candidates()`** instead of direct `similarity_search()`
4. **Pass the ORIGINAL description** (not expanded) to the LLM prompt

Here is the complete replacement for `match_single()`:

```python
def match_single(self, service_description: str) -> dict:
    """
    Match a single service description to a billing code.

    Pipeline:
    1. Validate input
    2. Check cache
    3. Expand query (abbreviations → formal SBS terminology)
    4. Two-pass retrieval (general + specialty-filtered)
    5. Format candidates with enriched metadata
    6. LLM evaluation with metadata-aware prompt
    7. Post-LLM validation
    8. Cache and return

    Args:
        service_description: The input service description to match.

    Returns:
        Dictionary with matched_code, code_system, matched_description,
        confidence, and reasoning.
    """
    # Handle empty or NaN service descriptions
    if not service_description or pd.isna(service_description):
        return {
            "matched_code": None,
            "code_system": None,
            "matched_description": None,
            "confidence": "NONE",
            "reasoning": "Empty or invalid service description",
        }

    service_description = str(service_description).strip()
    if not service_description or service_description.lower() == 'nan':
        return {
            "matched_code": None,
            "code_system": None,
            "matched_description": None,
            "confidence": "NONE",
            "reasoning": "Empty or invalid service description",
        }

    # Check cache
    cache_key = service_description.strip().lower()
    if cache_key in self.cache:
        self.cache_hits += 1
        return self.cache[cache_key].copy()

    self.cache_misses += 1

    # Step 1: Query expansion (abbreviation → formal terminology)
    if getattr(config, 'ENABLE_QUERY_EXPANSION', False):
        expanded_query = expand_query(service_description)
    else:
        expanded_query = service_description

    # Step 2: Two-pass retrieval with expanded query
    docs = self.retrieve_candidates(expanded_query)

    if not docs:
        result = {
            "matched_code": None,
            "code_system": None,
            "matched_description": None,
            "confidence": "NONE",
            "reasoning": "No candidates found in vector database",
        }
        self.cache[cache_key] = result
        return result

    top_doc = docs[0]

    # Step 3: Build prompt with ORIGINAL description (not expanded)
    # and enriched candidates (with full metadata)
    candidates_text = format_candidates(docs)
    prompt = MATCH_PROMPT.format(
        service_description=service_description,  # Original, not expanded
        candidates=candidates_text,
    )

    # Step 4: Call LLM with retry logic
    max_retries = 5
    base_delay = 5

    response_text = None
    for attempt in range(max_retries):
        try:
            response = self.llm.invoke(prompt)
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            break
        except Exception as e:
            error_msg = str(e)
            if "ThrottlingException" in error_msg or "Too many requests" in error_msg:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"Rate limited, retrying in {delay}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
                else:
                    print(f"Max retries reached for throttling")
                    result = {
                        "matched_code": None,
                        "code_system": None,
                        "matched_description": None,
                        "confidence": "NONE",
                        "reasoning": "Rate limit exceeded, please try again later",
                    }
                    self.cache[cache_key] = result
                    return result
            else:
                print(f"LLM Error: {error_msg}")
                result = self._extract_metadata(top_doc)
                result["confidence"] = "LOW"
                result["reasoning"] = f"LLM call failed: {error_msg[:100]}; using vector similarity"
                self.cache[cache_key] = result
                return result

    if response_text is None:
        result = self._extract_metadata(top_doc)
        result["confidence"] = "LOW"
        result["reasoning"] = "LLM call failed after retries; using vector similarity"
        self.cache[cache_key] = result
        return result

    # Step 5: Parse LLM response
    result = self._parse_llm_response(response_text, top_doc, docs)

    # Step 6: Post-LLM validation (safety net)
    if getattr(config, 'ENABLE_POST_LLM_VALIDATION', False):
        result = self._post_validate(result, service_description, docs)

    # Cache and return
    self.cache[cache_key] = result
    return result
```

#### 3.1.5: Add post-LLM validation method

Add this new method to `CodeMatcher` after `_parse_llm_response()`:

```python
def _post_validate(self, result: dict, original_input: str, candidates: list) -> dict:
    """
    Post-LLM validation safety net. Catches remaining LLM errors by
    applying hard rules that override the LLM's decision.

    Rules:
    1. If input contains dental abbreviations but match is non-dental → NONE
    2. If input is a drug name but matched to a procedure code → NONE
    3. If matched code's Excludes field matches the input → NONE

    Args:
        result: The LLM's parsed result dict.
        original_input: The original (unexpanded) service description.
        candidates: The candidate documents used for matching.

    Returns:
        Possibly modified result dict.
    """
    if not result.get("matched_code"):
        return result  # Already null, nothing to validate

    input_lower = original_input.lower()
    matched_code = result.get("matched_code", "")
    code_system = result.get("code_system", "")

    # Rule 1: Dental abbreviation → non-dental code
    dental_signals = ["rct", "r.c.t", "root canal", "crown", "filling",
                      "extraction", "dental", "tooth", "pfm", "veneer",
                      "composite", "amalgam", "endodon", "pulp"]
    input_is_dental = any(s in input_lower for s in dental_signals)

    if input_is_dental and code_system == "SBS":
        # Find the matched document to check its chapter
        for doc in candidates:
            if doc.metadata.get("code") == matched_code:
                chapter = doc.metadata.get("chapter_name", "").upper()
                if chapter and "DENTAL" not in chapter and "ORAL" not in chapter:
                    result["matched_code"] = None
                    result["code_system"] = None
                    result["matched_description"] = None
                    result["confidence"] = "NONE"
                    result["reasoning"] = (
                        f"Post-validation rejected: dental input matched to "
                        f"non-dental chapter ({chapter})"
                    )
                    return result
                break

    # Rule 2: Check Excludes field
    for doc in candidates:
        if doc.metadata.get("code") == matched_code:
            page_content = doc.page_content.lower()
            # Extract excludes text from enriched document
            if "excludes:" in page_content:
                excludes_start = page_content.index("excludes:") + len("excludes:")
                # Find the next section or end of string
                excludes_text = page_content[excludes_start:]
                next_section = excludes_text.find(". guideline:")
                if next_section == -1:
                    next_section = len(excludes_text)
                excludes_text = excludes_text[:next_section].strip()

                # Check if input matches excludes content
                input_words = set(input_lower.split())
                excludes_words = set(excludes_text.split())
                overlap = input_words & excludes_words
                # If significant overlap with excludes, reject
                meaningful_overlap = overlap - {"the", "a", "an", "of", "and", "or", "in", "to", "for"}
                if len(meaningful_overlap) >= 3:
                    result["matched_code"] = None
                    result["code_system"] = None
                    result["matched_description"] = None
                    result["confidence"] = "NONE"
                    result["reasoning"] = (
                        f"Post-validation rejected: input matches Excludes field "
                        f"(overlap: {', '.join(list(meaningful_overlap)[:5])})"
                    )
                    return result
            break

    return result
```

---

## PHASE 4: METADATA-AWARE PROMPT — LLM Leverages Enriched Data (Priority: HIGH)

### Step 4.1: Replace `MATCH_PROMPT` in `prompts.py`

**File:** `prompts.py`
**What to do:** Replace the entire `MATCH_PROMPT` string with the V3 metadata-aware version.

Replace the existing `MATCH_PROMPT` (lines 7-94) with the following. The key additions are:

- **USING CANDIDATE METADATA** section explaining Specialty, Category, Clinical Context, Excludes, Includes fields
- **CHECK 7 (EXCLUDES)** — hard rejection when input matches Excludes
- **CHECK 8 (INCLUDES)** — strong positive signal when input matches Includes
- **ABBREVIATION SAFETY** warning
- Updated mandatory rejections (now 8 rules instead of 6)

The full replacement prompt is provided in the `Complete_System_Redesign.md` file under "CHANGE 6: Updated Prompt." Copy it exactly from there.

### Step 4.2: Rewrite `format_candidates()` in `prompts.py`

Replace the existing `format_candidates()` function (lines 97-128) to pass the FULL enriched `page_content` to the LLM instead of stripping it:

```python
def format_candidates(retrieved_docs) -> str:
    """
    Format retrieved documents with full enriched context for the LLM prompt.
    Now passes complete metadata (Specialty, Category, Excludes, etc.)
    instead of just the short description.

    Args:
        retrieved_docs: List of LangChain Document objects from retrieval.

    Returns:
        A formatted string with numbered candidates including full context.
    """
    formatted = []

    for i, doc in enumerate(retrieved_docs, 1):
        meta = doc.metadata
        system = meta.get("system", "UNKNOWN")
        code = meta.get("code", "")

        # Use the full enriched page_content (includes Specialty, Category,
        # Clinical Context, Excludes, Includes, Guideline)
        content = doc.page_content

        # Strip the system prefix since we add it in the format below
        for prefix in ["[SBS] ", "[GTIN] ", "[GMDN] "]:
            if content.startswith(prefix):
                content = content[len(prefix):]
                break

        formatted.append(f"Candidate {i}: [{system}] Code: {code}\n  {content}")

    return "\n\n".join(formatted)
```

**Key difference from current version:** Instead of single-line entries like:

```
1. [SBS] Code: 39003-00-00 | Cisternal puncture
```

The new format produces multi-line entries like:

```
Candidate 1: [SBS] Code: 39003-00-00
  Code: 39003-00-00. Specialty: PROCEDURES ON NERVOUS SYSTEM.
  Category: Cranial tap or puncture. Description: Cisternal puncture.
  Clinical context: ...
```

This gives the LLM the full context needed to evaluate each candidate.

---

## PHASE 5: UPDATE TESTS (Priority: MEDIUM)

### Step 5.1: Update `test_matcher.py`

**File:** `test_matcher.py`
**What to do:** Update existing tests and add new tests for enriched documents, query expansion, and two-pass retrieval.

#### 5.1.1: Fix the inconsistent import

In `TestVectorStoreRetrieval`, lines 159 and 179 import from `langchain_community.vectorstores` instead of `langchain_chroma`. Fix both to use:

```python
from langchain_chroma import Chroma
```

#### 5.1.2: Update `TestBuildDocuments` for enriched SBS documents

The `test_build_documents_have_system_prefix` test should still pass because enriched SBS documents still start with `[SBS]`. However, add a new test to verify enrichment:

```python
def test_sbs_documents_have_enriched_metadata(self):
    """Test that SBS documents include Chapter/Block metadata."""
    from ingest import build_documents

    docs = build_documents(config.REFERENCE_FILE)
    sbs_docs = [d for d in docs if d.metadata["system"] == "SBS"]

    for doc in sbs_docs:
        # Verify enriched metadata fields exist
        assert "chapter_name" in doc.metadata, "SBS doc missing chapter_name"
        assert "block_name" in doc.metadata, "SBS doc missing block_name"
        assert "has_excludes" in doc.metadata, "SBS doc missing has_excludes"
        assert "has_includes" in doc.metadata, "SBS doc missing has_includes"

        # Verify enriched page_content contains Specialty/Category markers
        # (if the source data has chapter/block info)
        if doc.metadata["chapter_name"]:
            assert "Specialty:" in doc.page_content, \
                f"Enriched SBS doc missing Specialty in content: {doc.page_content[:80]}"
```

#### 5.1.3: Add new test class for query expansion

```python
class TestQueryExpansion:
    """Tests for the query expansion module."""

    def test_rct_expansion(self):
        """Test that RCT abbreviation is expanded."""
        from query_expansion import expand_query

        result = expand_query("RCT")
        assert "root canal" in result.lower()
        assert "endodontic" in result.lower()

    def test_re_rct_expansion(self):
        """Test that RE-RCT abbreviation is expanded."""
        from query_expansion import expand_query

        result = expand_query("RE-RCT")
        assert "retreatment" in result.lower()

    def test_pfm_expansion(self):
        """Test that PFM abbreviation is expanded."""
        from query_expansion import expand_query

        result = expand_query("PFM Crown")
        assert "porcelain" in result.lower()
        assert "metal" in result.lower()

    def test_no_expansion_for_normal_text(self):
        """Test that normal text is not expanded."""
        from query_expansion import expand_query

        result = expand_query("Cisternal puncture")
        assert result == "Cisternal puncture"

    def test_specialty_detection_dental(self):
        """Test dental specialty detection."""
        from query_expansion import detect_specialty

        result = detect_specialty("root canal treatment endodontic")
        assert result is not None
        assert "DENTAL" in result.upper()

    def test_specialty_detection_none(self):
        """Test that unknown text returns None."""
        from query_expansion import detect_specialty

        result = detect_specialty("random unrelated text xyz")
        assert result is None
```

#### 5.1.4: Update `test_format_candidates_output` if it depends on old format

The existing test checks for `lines[0].startswith("1.")`. Since the new format uses `"Candidate 1:"`, update the assertion:

```python
assert lines[0].startswith("Candidate 1:"), "First candidate should start with 'Candidate 1:'"
```

---

## PHASE 6: UPDATE STREAMLIT UI (Priority: LOW)

### Step 6.1: Minor UI Updates in `app.py`

**File:** `app.py`
**What to do:** Update the sidebar configuration display and add new status indicators.

#### 6.1.1: Update sidebar configuration section

Replace the configuration display (lines 93-104) to show V2 settings:

```python
    # Configuration info
    st.subheader("Configuration")
    if config.USE_LOCAL:
        st.info("Using LOCAL models (no AWS credentials found)")
        st.caption("Embedding: all-MiniLM-L6-v2 (HuggingFace)")
        st.caption("LLM: Ollama gemma2:2b")
    else:
        st.info("Using AWS Bedrock")
        st.caption("Embedding: all-MiniLM-L6-v2 (HuggingFace)")
        st.caption(f"LLM: {config.LLM_MODEL}")
        st.caption(f"Region: {config.AWS_REGION}")

    st.divider()
    st.subheader("V2 Features")
    st.caption(f"Query Expansion: {'ON' if getattr(config, 'ENABLE_QUERY_EXPANSION', False) else 'OFF'}")
    st.caption(f"Specialty Filter: {'ON' if getattr(config, 'ENABLE_SPECIALTY_FILTER', False) else 'OFF'}")
    st.caption(f"Post-LLM Validation: {'ON' if getattr(config, 'ENABLE_POST_LLM_VALIDATION', False) else 'OFF'}")
    st.caption(f"Candidates: {getattr(config, 'TOP_K_GENERAL', 10)} general + {getattr(config, 'TOP_K_FILTERED', 10)} filtered")
```

---

## PHASE 7: RE-INDEX AND VERIFY (Priority: CRITICAL)

This phase is not code changes — it is the validation procedure after all code changes are complete.

### Step 7.1: Delete Existing Vector Store

```bash
# Delete the old ChromaDB data
rm -rf ./chroma_db
```

### Step 7.2: Re-index with Enriched Documents

```bash
python ingest.py
```

**Verify output shows:**
- Same document counts as before (9 SBS + 9 GTIN + 8 GMDN = 26 for sample data)
- SBS documents now contain "Specialty:", "Category:", "Clinical context:", "Excludes:", "Includes:" where available
- Verification queries still return correct results

### Step 7.3: Run Tests

```bash
pytest test_matcher.py -v
```

All tests should pass, including the new enrichment and query expansion tests.

### Step 7.4: Test Known Failure Cases

These are the specific cases that the redesign is meant to fix. Test them manually:

```python
from matcher import CodeMatcher
matcher = CodeMatcher()

# These should NO LONGER return wrong codes:
print(matcher.match_single("RE-RCT"))              # Should NOT return wart removal
print(matcher.match_single("R.C.T. (Birooted)"))   # Should NOT return blood typing
print(matcher.match_single("PFM Crown"))            # Should NOT return zirconia
print(matcher.match_single("3 Surface Filling"))    # Should NOT return 3D photography
print(matcher.match_single("Dental Examination"))   # Should return dental exam, not random exam

# These should return null (no match) rather than a wrong match:
# If the correct code doesn't exist in the reference data, null/NONE is the correct answer.
```

### Step 7.5: Full Batch Test

```bash
# Run the full mapping sheet through the new pipeline
python -c "
from matcher import CodeMatcher
matcher = CodeMatcher()
matcher.match_batch('./data/input/Code Mapping Sheet_Sample.xlsx', './data/output/v2_matched.xlsx')
"
```

Compare V2 output against V1 output. Look for:
- Fewer NONE results (better retrieval finds more correct candidates)
- Fewer obviously wrong HIGH-confidence matches (metadata checks catch cross-domain errors)
- More accurate dental code matches (abbreviation expansion works)

---

## IMPLEMENTATION CHECKLIST

Execute these in order. Each step depends on the ones before it.

```
STEP   FILE                    CHANGE                                   STATUS
──────────────────────────────────────────────────────────────────────────────
 1     config.py               Add V2 config constants                  [ ]
 2     config.py               Remove/comment AUTO_ACCEPT_THRESHOLD     [ ]
 3     ingest.py               Add clean_field() helper                 [ ]
 4     ingest.py               Add build_enriched_sbs_document()        [ ]
 5     ingest.py               Replace SBS section in build_documents() [ ]
 6     ingest.py               Run ingest.py, verify enriched output    [ ]
 7     query_expansion.py      Create new file with vocabulary map      [ ]
 8     query_expansion.py      Implement expand_query()                 [ ]
 9     query_expansion.py      Implement detect_specialty()             [ ]
10     matcher.py              Import query_expansion                   [ ]
11     matcher.py              Remove duplicate SentenceTransformer     [ ]
12     matcher.py              Add retrieve_candidates() method         [ ]
13     matcher.py              Rewrite match_single() with new pipeline [ ]
14     matcher.py              Add _post_validate() method              [ ]
15     prompts.py              Replace MATCH_PROMPT with V3 version     [ ]
16     prompts.py              Rewrite format_candidates()              [ ]
17     test_matcher.py         Fix inconsistent Chroma import           [ ]
18     test_matcher.py         Add enriched metadata tests              [ ]
19     test_matcher.py         Add query expansion tests                [ ]
20     test_matcher.py         Update format_candidates test            [ ]
21     app.py                  Update sidebar config display            [ ]
22     (none)                  Delete ./chroma_db and re-index          [ ]
23     (none)                  Run all tests                            [ ]
24     (none)                  Test known failure cases manually         [ ]
25     (none)                  Run full batch and compare output         [ ]
```

---

## THINGS TO WATCH OUT FOR

### 1. Column Name Typos in Source Data

The SBS Excel file has a typo in the Clinical Explanation column name: `"Cliincal Explanation "` (double "i" and trailing space). You MUST use this exact string when reading from the DataFrame. If you "fix" the typo, your code will silently get empty values.

**How to verify:** After reading the Excel file, print `df_sbs.columns.tolist()` and check the exact column names.

### 2. ChromaDB Metadata Filter Limitations

ChromaDB metadata filters only work with exact string matching. The `chapter_name` field must be stored consistently (all uppercase, trimmed) for the specialty filter to work:

```python
# Good: "PROCEDURES ON NERVOUS SYSTEM" (uppercase, trimmed)
# Bad:  "Procedures on Nervous System" (mixed case)
# Bad:  " PROCEDURES ON NERVOUS SYSTEM " (whitespace)
```

The `build_enriched_sbs_document()` function handles this with `chapter.upper().strip()`.

### 3. ChromaDB `chapter_number` Must Be Numeric

ChromaDB metadata values must be strings, ints, floats, or bools. If `chapter_number` is NaN, you cannot store it as None — ChromaDB will reject it. Use `-1` as a sentinel value for missing chapter numbers.

### 4. GTIN and GMDN Are Unchanged

The redesign focuses on SBS codes. GTIN (pharmaceutical drugs) and GMDN (medical devices) document building and retrieval remain exactly as they are. Do not modify those sections.

### 5. Query Expansion is for RETRIEVAL Only

The expanded query is sent to ChromaDB for vector search. The ORIGINAL (unexpanded) description is sent to the LLM in the prompt. This is intentional — the LLM should see exactly what the customer wrote, not our expansion. The expansion's job is to get the right candidates into the retrieval results.

### 6. The Prompt Must Match the Document Format

The LLM prompt now tells the model to look for "Specialty:", "Category:", "Excludes:", etc. These labels MUST match exactly what `build_enriched_sbs_document()` puts into the `page_content`. If you change the labels in one place, change them everywhere.

### 7. Two-Pass Retrieval May Return Duplicates

The `retrieve_candidates()` method deduplicates by `{system}:{code}` key. However, if the same code appears in both general and filtered results with slightly different metadata (e.g., different document IDs), both will be included. The dedup logic handles this correctly as long as you key on `code` + `system`.

### 8. Full Reference Data May Have Different Column Names

The sample data file (`Saudi Billing Codes_Sample.xlsx`) has been verified to have the column names used in this document. However, the full production data file (`Saudi Billing Codes Updated List.xlsx`) may have slightly different column names. After switching to the full data file:

1. Print `df.columns.tolist()` for each sheet
2. Compare against the column names used in `build_enriched_sbs_document()`
3. Update column references if they differ

### 9. Specialty Keywords Must Match Actual Chapter Names

The `SPECIALTY_KEYWORDS` dict in `query_expansion.py` uses assumed chapter names (e.g., "DENTAL PROCEDURES"). After indexing the full reference data, extract the actual chapter names and update the dictionary keys:

```python
import chromadb
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("saudi_billing_codes")
results = collection.get()
chapter_names = set(m.get("chapter_name", "") for m in results["metadatas"] if m.get("system") == "SBS")
print(sorted(chapter_names))
```

### 10. Rate Limiting for Full Data

The full reference data has thousands of codes. The sample has 26. When switching to full data:
- Indexing will take longer (minutes instead of seconds)
- The batch size of 500 in `create_vector_store()` is fine
- Vector search will be slower with more documents
- Consider increasing `TOP_K_GENERAL` and `TOP_K_FILTERED` proportionally

---

## EXPECTED OUTCOMES

After implementing all phases:

| Error Type | Before (V1) | After (V2) | Why |
|---|---|---|---|
| RE-RCT → Wart removal | Happens | Fixed | Query expansion maps "RCT" → endodontic; specialty filter excludes skin codes |
| R.C.T. → Blood typing | Happens | Fixed | Dotted abbreviation expansion + dental specialty filter |
| PFM → Zirconia | Happens | Fixed | Query expansion adds "metal substructure"; enriched docs show material |
| 3-Surface → 3D Photography | Happens | Fixed | Enriched Category field distinguishes "restorative" from "imaging" |
| Dental Exam → Random Exam | Happens | Fixed | Specialty filter returns only dental chapter codes |
| All matches HIGH confidence | Happens | Fixed | Metadata checks give LLM real disambiguation signals |
| Excludes violations | Happens | Fixed | Check 7 in prompt + post-LLM validation catch these |

**Overall expected accuracy improvement: 80-90% reduction in cross-domain matching errors.**

---

## FILE DEPENDENCY ORDER

Build/modify files in this order to avoid import errors:

```
1. config.py              (no dependencies)
2. query_expansion.py     (no dependencies)
3. ingest.py              (depends on: config)
4. prompts.py             (no dependencies)
5. matcher.py             (depends on: config, ingest, prompts, query_expansion)
6. app.py                 (depends on: config, ingest, matcher)
7. test_matcher.py        (depends on: all of the above)
```

---

## ROLLBACK PLAN

If the V2 changes cause unexpected issues:

1. **Config:** Remove the V2 settings block. Restore `AUTO_ACCEPT_THRESHOLD`.
2. **Ingest:** Revert `build_documents()` SBS section to original simple format.
3. **Matcher:** Revert `match_single()` to original with text containment auto-accept.
4. **Prompts:** Revert `MATCH_PROMPT` and `format_candidates()`.
5. **Delete `query_expansion.py`.**
6. **Delete `./chroma_db` and re-run `python ingest.py`** to rebuild with old format.

All V2 changes are backward-compatible — the GTIN/GMDN pipeline is untouched, and the SBS enrichment is purely additive (more metadata, not different data).
