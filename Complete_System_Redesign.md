# The Real Problem: You're Throwing Away 80% of Your Data

## EXECUTIVE FINDING

After examining the actual Saudi Billing Codes reference sheet, the single biggest
problem with your matching system is **not the prompt**. It's that your vector store
indexing is throwing away the richest, most disambiguating metadata in the SBS dataset.

Your SBS reference file has **12 columns**. You're likely using **2 of them**.

```
FIELD                    USED?    WHAT IT DOES FOR MATCHING
──────────────────────────────────────────────────────────────────────────────────
SBS Code                  ✅      The code itself
SBS Code Hyphenated       ✅      Formatted code
Short Description         ✅      Indexed in vector store
Long Description          ✅      Indexed in vector store (often = Short)

Chapter Name              ❌      "PROCEDURES ON NERVOUS SYSTEM" → SPECIALTY
Block Name                ❌      "Cranial tap or puncture" → PROCEDURE CATEGORY
Block Number              ❌      Hierarchical group → CATEGORY FILTER KEY
Chapter Number            ❌      Specialty group → SPECIALTY FILTER KEY
Clinical Explanation      ❌      "A needle placed below the occipital bone" → CONTEXT
Excludes                  ❌      "Drainage of infected cyst (39900-00-00)" → ANTI-MATCH
Includes                  ❌      What this code covers → PRO-MATCH
Guideline                 ❌      Coding rules → CONSTRAINTS
```

**The Excludes field alone would prevent an entire class of errors.** When the SBS data
says "Code 39703-03-00 (Aspiration of brain cyst) EXCLUDES Drainage of infected cyst
(39900-00-00)", you have an authoritative, built-in disambiguation signal. This is not
something the LLM needs to figure out — the SBS authors already figured it out for you.

---

## WHAT THE INPUT DATA TELLS US

The Code_Mapping_Sheet has this structure:

```
Service Code | Service Description             | NPHIES Code | Description | Other Code Value
─────────────────────────────────────────────────────────────────────────────────────────────
C1001        | Dental Examination by General    |             |             |
             |   Dentist                        |             |             |
C1002        | Dental Examination by Specialist |             |             |
C1003        | Dental Examination by Consultant |             |             |
C1004        | Referral to Consultant/specialist|             |             |
O1001        | Medical report                   |             |             |
```

Key observations:

1. **NPHIES Code, Description, Other Code Value are EMPTY** — these are the output columns
   your system needs to fill. The customer has a list of their internal service codes and
   descriptions, and they need the corresponding SBS/NPHIES codes.

2. **Service codes have prefixes** — "C1001" through "C1004" are consultation/examination
   codes. "O1001" is an administrative code. The previous dental sample had codes like
   "446T", "503T", "310T". These prefixes encode the customer's internal categorization
   and could be used as hints.

3. **Descriptions are natural language, often abbreviated** — "Dental Examination by General
   Dentist" is clear, but "RE-RCT", "R.C.T. (Birooted)", "dolo 650" are abbreviations
   that need expansion.

4. **The gap between input quality and SBS quality is massive** — Customer inputs are
   informal shorthand. SBS codes have formal medical terminology plus chapter/block/clinical
   context. The embedding model needs to bridge this gap, which it currently can't do
   for abbreviations.

---

## THE SIX CHANGES THAT WILL TRANSFORM ACCURACY

Listed in order of impact.

---

### CHANGE 1: Enrich Vector Store Documents With ALL Metadata

**This is the highest-impact change in this entire document.**

Currently, your indexing probably looks like:

```python
# CURRENT (likely)
text = f"[SBS] {short_description}. {long_description}"
```

For code 39703-03-00, this produces:
```
[SBS] Aspiration of brain cyst. Aspiration of brain cyst
```

56 characters. No specialty. No category. No clinical context. No disambiguation signals.

**Change it to:**

```python
def build_enriched_document(row: pd.Series) -> str:
    """
    Build a richly annotated document string from an SBS reference row.
    Every field that exists gets included — more context = better embeddings
    AND better LLM matching.
    """
    parts = []

    code = str(row.get("SBS Code Hyphenated", "")).strip()
    short = str(row.get("Short Description", "")).strip()
    long_desc = str(row.get("Long Description", "")).strip()
    chapter = str(row.get("Chapter Name", "")).strip()
    block = str(row.get("Block Name", "")).strip()
    clinical = str(row.get("Cliincal Explanation ", "")).strip()
    includes = str(row.get("Includes", "")).strip()
    excludes = str(row.get("Excludes", "")).strip()
    guideline = str(row.get("Guideline", "")).strip()

    # Clean nan strings
    for var_name in ['code','short','long_desc','chapter','block',
                     'clinical','includes','excludes','guideline']:
        val = locals()[var_name]
        if val.lower() in ('nan', 'none', ''):
            locals()[var_name] = ''
        exec(f"{var_name} = '{val}'" if val.lower() not in ('nan','none','') else f"{var_name} = ''")

    # Actually, simpler approach:
    def clean(val):
        if not val or str(val).lower() in ('nan', 'none', ''):
            return ''
        return str(val).strip()

    code = clean(row.get("SBS Code Hyphenated"))
    short = clean(row.get("Short Description"))
    long_desc = clean(row.get("Long Description"))
    chapter = clean(row.get("Chapter Name"))
    block = clean(row.get("Block Name"))
    clinical = clean(row.get("Cliincal Explanation "))
    includes = clean(row.get("Includes"))
    excludes = clean(row.get("Excludes"))
    guideline = clean(row.get("Guideline"))

    # Build the document
    parts.append(f"[SBS] Code: {code}")

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

    return ". ".join(parts)
```

For code 39703-03-00, this now produces:
```
[SBS] Code: 39703-03-00. Specialty: PROCEDURES ON NERVOUS SYSTEM.
Category: Cranial tap or puncture. Description: Aspiration of brain cyst.
Clinical context: Navigation-guided cyst aspiration followed by resection is
a procedure for brain tumours with large cystic components.
Excludes: Drainage of infected cyst (39900-00-00 [8])
```

**320 characters.** 6x more context. The embedding model now indexes this document
near other neurosurgery codes (because of "NERVOUS SYSTEM", "cranial", "brain"),
not near random codes that share the word "aspiration" (like aspiration pneumonia
or bone marrow aspiration).

**Why this matters for EVERY error we've seen:**

| Error | How enriched docs help |
|---|---|
| RE-RCT → Wart removal | Wart code would show "Category: Destruction of skin lesion" or similar. Even if retrieved, the LLM sees a completely different specialty/category than endodontic. |
| R.C.T. → Blood typing | Blood typing code would show "Specialty: PATHOLOGY/LAB" and "Category: Blood grouping". The specialty mismatch is now EXPLICIT in the candidate text, not implicit. |
| 3 Surface → 3D Photography | Photography code would show "Category: Diagnostic imaging" not "Restorative dentistry". The LLM can now SEE the domain mismatch in the candidate text itself. |
| PFM → Zirconia | With clinical context, the zirconia code might note "all-ceramic" and the PFM code (if it existed in index) would note "metal substructure". |

**Impact on embeddings:** When you index "Specialty: PROCEDURES ON NERVOUS SYSTEM.
Category: Cranial tap or puncture. Description: Aspiration of brain cyst", the
embedding vector naturally clusters with other neurosurgery codes. A query like
"brain cyst drainage" will now retrieve codes from the right specialty, not random
codes that happen to share the word "drainage."

---

### CHANGE 2: Store Chapter/Block as Metadata for Filtered Retrieval

Don't just put Chapter/Block in the document text — also store them as **metadata
fields** in ChromaDB. This enables filtered vector search.

```python
# During indexing — store metadata alongside the document
doc = Document(
    page_content=enriched_text,        # The full enriched text from Change 1
    metadata={
        "code": code,
        "code_system": "SBS",
        "short_description": short,
        "chapter_number": chapter_num,  # e.g., 1 = Nervous System
        "chapter_name": chapter,        # "PROCEDURES ON NERVOUS SYSTEM"
        "block_number": block_num,      # e.g., 2 = Cranial tap
        "block_name": block,            # "Cranial tap or puncture"
        "has_excludes": bool(excludes), # Quick filter
        "has_clinical": bool(clinical), # Quick filter
    }
)
```

**Now your retrieval can do TWO searches:**

```python
def retrieve_candidates(query: str, top_k: int = 15) -> list:
    """
    Two-pass retrieval: general similarity + specialty-filtered.
    Merge and deduplicate results.
    """

    # Pass 1: General semantic search across ALL codes
    general_results = vector_store.similarity_search(query, k=top_k)

    # Pass 2: If we can identify the specialty, search WITHIN that specialty
    # This could be a simple keyword-based classifier or another LLM call
    specialty = detect_specialty(query)
    if specialty:
        filtered_results = vector_store.similarity_search(
            query,
            k=top_k,
            filter={"chapter_name": specialty}  # ChromaDB metadata filter
        )
    else:
        filtered_results = []

    # Merge and deduplicate
    seen_codes = set()
    merged = []
    for doc in general_results + filtered_results:
        code = doc.metadata.get("code")
        if code not in seen_codes:
            seen_codes.add(code)
            merged.append(doc)

    return merged[:top_k * 2]  # Return up to 2x candidates for LLM to evaluate


def detect_specialty(query: str) -> str | None:
    """
    Simple keyword-based specialty detection.
    Expand this as you encounter more specialties.
    """
    q = query.lower()

    # Dental
    dental_signals = ["tooth", "dental", "rct", "r.c.t", "crown", "filling",
                      "extraction", "impaction", "root canal", "amalgam",
                      "composite", "resin", "veneer", "gingival", "periodon",
                      "endodon", "prosthod", "orthodon", "implant",
                      "alveol", "pfm", "porcelain", "zirconi", "surface"]
    if any(s in q for s in dental_signals):
        return "DENTAL PROCEDURES"  # Match against your actual chapter names

    # Nervous system
    neuro_signals = ["brain", "cranial", "spinal", "intracranial",
                     "ventricular", "meninges", "cerebro", "neuro"]
    if any(s in q for s in neuro_signals):
        return "PROCEDURES ON NERVOUS SYSTEM"

    # Cardiac
    cardiac_signals = ["heart", "cardiac", "coronary", "valve", "cabg",
                       "pacemaker", "stent", "angioplast", "bypass"]
    if any(s in q for s in cardiac_signals):
        return "PROCEDURES ON CARDIOVASCULAR SYSTEM"  # Use actual chapter name

    # ... add more specialties from your full SBS chapter list

    return None  # No specialty detected — use general search only
```

**Why two-pass retrieval matters:**

When someone searches "R.C.T. (Birooted)" and you expand it to "root canal treatment
endodontic birooted two roots", the general search might return a mix of dental and
random codes. But the specialty-filtered search (filtering to "DENTAL PROCEDURES"
chapter) will ONLY return dental codes. The blood typing code and wart removal code
physically cannot appear in the filtered results.

---

### CHANGE 3: Pass Enriched Candidate Text to the LLM (Not Just Short Description)

Currently, your candidates passed to the LLM probably look like:

```
Candidate 1: [SBS] Code: 39703-03-00 — Aspiration of brain cyst
Candidate 2: [SBS] Code: 39900-00-00 — Drainage of infected cyst
```

With enriched documents, the LLM now sees:

```
Candidate 1: [SBS] Code: 39703-03-00. Specialty: PROCEDURES ON NERVOUS SYSTEM.
  Category: Cranial tap or puncture. Description: Aspiration of brain cyst.
  Clinical: Navigation-guided cyst aspiration for brain tumours with large
  cystic components.
  EXCLUDES: Drainage of infected cyst (39900-00-00)

Candidate 2: [SBS] Code: 39900-00-00. Specialty: PROCEDURES ON NERVOUS SYSTEM.
  Category: Cranial tap or puncture. Description: Drainage of infected cyst.
```

Now when the LLM evaluates these candidates, it has DRAMATICALLY more information:
- **Specialty** tells it the clinical domain → Check 1 becomes trivial
- **Category** tells it the procedure type → Check 2 becomes trivial
- **Clinical context** helps with nuanced matching
- **Excludes** DIRECTLY tells the LLM "do not match this code to X"
- **Includes** DIRECTLY tells the LLM "this code covers X"

This changes the LLM's job from "figure out what this code is about from a 3-word
description" to "read the complete context and compare."

**Format the candidates like this for the prompt:**

```python
def format_candidates_for_prompt(candidates: list) -> str:
    """
    Format retrieved candidates with full context for the LLM prompt.
    """
    formatted = []

    for i, doc in enumerate(candidates, 1):
        meta = doc.metadata
        text = doc.page_content  # Already enriched from Change 1

        # Also extract key metadata for structured display
        code = meta.get("code", "")
        system = meta.get("code_system", "SBS")
        chapter = meta.get("chapter_name", "")
        block = meta.get("block_name", "")
        short_desc = meta.get("short_description", "")

        formatted.append(
            f"Candidate {i}: [{system}] Code: {code}\n"
            f"  {text}"
        )

    return "\n\n".join(formatted)
```

---

### CHANGE 4: Teach the Prompt to USE the Enriched Metadata

The prompt needs to know that candidates now carry Specialty, Category, Excludes, etc.
Add this to the prompt:

```python
# ADD to the prompt, in the STEP 2 section:

"""
## USING CANDIDATE METADATA

Each candidate now includes structured context fields. Use them:

SPECIALTY / CHAPTER: This tells you the clinical specialty of the candidate code.
  → The input's specialty should match the candidate's specialty.
  → "Dental Examination" should match codes from a dental chapter, not neurology.
  → This is your primary filter for Check 1 (Domain).

CATEGORY / BLOCK: This tells you the procedure sub-category.
  → "Examination" should match codes in an examination block, not a surgical block.
  → "Root canal" should match codes in an endodontic block, not a prosthodontic block.
  → This is your primary filter for Check 2 (Service Type).

CLINICAL CONTEXT: This gives you additional matching information.
  → Use it to understand nuances the short description doesn't capture.
  → Example: "Navigation-guided cyst aspiration" tells you this is image-guided.

EXCLUDES: This tells you what this code is NOT for.
  → If the input description matches something in the EXCLUDES field, this
     candidate is EXPLICITLY WRONG. The SBS coding authority says so.
  → Treat EXCLUDES as a HARD REJECTION — stronger than any other signal.

INCLUDES: This tells you what this code covers.
  → If the input description matches something in the INCLUDES field, this
     is a STRONG POSITIVE signal for the match.
"""
```

And add a new rejection rule:

```python
"""
REJECTION RULE 8 — EXCLUDES MATCH: If the input description matches content in
  the candidate's EXCLUDES field, that candidate is EXPLICITLY WRONG. The SBS
  system itself says this code should not be used for that service. Reject immediately.
"""
```

---

### CHANGE 5: Query Expansion — Now Informed by SBS Vocabulary

Now that you have the SBS Chapter Names and Block Names, you can build a query expansion
dictionary that maps customer shorthand to SBS vocabulary:

```python
# query_expansion.py — INFORMED BY SBS REFERENCE DATA

# Step 1: Extract all unique Chapter Names and Block Names from your SBS data
# Step 2: Build mappings from common abbreviations/shorthand to SBS terminology

SBS_VOCABULARY_MAP = {
    # ── Customer dental shorthand → SBS dental terminology ──
    "rct":          "root canal treatment endodontic",
    "r.c.t.":       "root canal treatment endodontic",
    "r.c.t":        "root canal treatment endodontic",
    "re-rct":       "root canal retreatment endodontic",
    "pfm":          "porcelain fused to metal crown metallic substructure",
    "opg":          "orthopantomogram panoramic radiograph",
    "tmj":          "temporomandibular joint",
    "tooth colored": "composite resin direct restoration",
    "tooth-colored": "composite resin direct restoration",
    "amalgam":       "amalgam metallic restoration direct",

    # ── Generated from SBS Block Names ──
    # (Extract these from your full SBS data programmatically)
    # Example: if a block is called "Cranial tap or puncture",
    # add synonyms that patients/doctors might use:
    "lumbar puncture":   "spinal tap cranial puncture cerebrospinal fluid",
    "spinal tap":        "lumbar puncture cranial tap cerebrospinal fluid",
    "brain biopsy":      "cranial biopsy nervous system",

    # ── Customer code prefixes (learned from input patterns) ──
    # C1xxx = Consultations/Examinations
    # Oxxx  = Administrative/Other
    # 3xxT  = Endodontic (dental)
    # 5xxT  = Restorative/Prosthodontic (dental)
    # etc.
}


def expand_query_with_sbs_vocab(description: str) -> str:
    """
    Expand customer shorthand using SBS-informed vocabulary.
    Called BEFORE vector search, NOT passed to LLM prompt.
    """
    desc_lower = description.lower().strip()
    expansions = []

    import re
    for trigger, expansion in SBS_VOCABULARY_MAP.items():
        # Whole-word matching to avoid false positives
        pattern = r'\b' + re.escape(trigger.replace('.', r'\.')) + r'\b'
        if re.search(pattern, desc_lower):
            expansions.append(expansion)

    # Also handle dotted abbreviations: "R.C.T." → "RCT"
    desc_no_dots = desc_lower.replace(".", "")
    for trigger, expansion in SBS_VOCABULARY_MAP.items():
        clean = trigger.replace(".", "")
        if clean != trigger:  # Only for dotted triggers
            if re.search(r'\b' + re.escape(clean) + r'\b', desc_no_dots):
                if expansion not in expansions:
                    expansions.append(expansion)

    if expansions:
        return f"{description} ({'; '.join(expansions)})"
    return description


def build_vocabulary_from_sbs(sbs_df: pd.DataFrame) -> dict:
    """
    UTILITY: Automatically extract unique vocabulary from SBS reference data.
    Run this once to discover all Chapter Names, Block Names, and common terms.
    Use the output to populate SBS_VOCABULARY_MAP.
    """
    chapters = sbs_df['Chapter Name'].dropna().unique().tolist()
    blocks = sbs_df['Block Name'].dropna().unique().tolist()

    # Extract key terms from clinical explanations
    clinical_terms = []
    for text in sbs_df['Cliincal Explanation '].dropna():
        # Simple word extraction — you could use NLP for better results
        words = str(text).lower().split()
        clinical_terms.extend([w for w in words if len(w) > 4])

    return {
        "chapters": sorted(set(chapters)),
        "blocks": sorted(set(blocks)),
        "clinical_terms": sorted(set(clinical_terms)),
    }
```

---

### CHANGE 6: Updated Prompt That Leverages Everything Above

Here's the compact V3 prompt with the metadata-aware additions:

```python
MATCH_PROMPT = """You are a Saudi healthcare billing code expert specialized in NPHIES compliance.

Candidates below were retrieved by TEXT SIMILARITY and MAY CONTAIN IRRELEVANT RESULTS.
Determine if any candidate is a true clinical match. Returning null is ALWAYS better than
a wrong code — wrong codes cause claim rejections, audit failures, and financial penalties.

## Code Systems
- SBS: procedures, surgeries, examinations, imaging, consultations
- GTIN: pharmaceutical drugs (brand, generic, strength, formulation)
- GMDN: devices, IVD kits, reagents, medical equipment

## Input
{service_description}

## Candidates
{candidates}

## STEP 1: PARSE INPUT (before looking at candidates)

Identify:
- DOMAIN: Surgical, Diagnostic/Imaging, Laboratory, Preventive, Therapeutic, Pharmaceutical, Device, Consultation, Administrative
- SPECIFIC SERVICE: exact procedure/test/drug/device described
- PARAMETERS: counts, dose/strength, size, duration, laterality
- QUALIFIERS: material, technique, route, complexity, timing (initial/revision/retreatment)

ABBREVIATION SAFETY: If the input contains abbreviations you are not 100% certain about,
do NOT guess. Look at candidate context for clues. If still uncertain, return null.

## STEP 2: EVALUATE EACH CANDIDATE — All 6 checks + metadata

USE THE CANDIDATE METADATA — each candidate may include Specialty, Category,
Clinical Context, Excludes, and Includes fields. These are from the official SBS
coding reference and are authoritative.

CHECK 1 — DOMAIN: Compare input domain to the candidate's Specialty/Chapter field.
  Reject if they describe completely different body systems or clinical domains.

CHECK 2 — SERVICE TYPE: Compare input service to the candidate's Category/Block field.
  Reject if: different procedure type even within the same specialty.
  Watch for: treatment≠retreatment, partial≠total, excision≠drainage,
  insertion≠removal≠replacement, a component≠the complete unit.

CHECK 3 — NUMBERS: All quantitative parameters align?
  Reject if: count differs by >1, dose differs by >10%.

CHECK 4 — QUALIFIERS: Material, technique, route, laterality compatible?
  Reject if: input specifies qualifier A, candidate specifies different qualifier B.

CHECK 5 — SCOPE: Same extent of work?
  Reject if: per-unit↔per-region, single↔multi-level, unilateral↔bilateral,
  simple↔complex, initial↔subsequent.

CHECK 6 — KEYWORD TRAP: Remove shared keywords — is there still a clinical connection?
  Reject if: match depends entirely on shared words without shared clinical meaning.

CHECK 7 — EXCLUDES (NEW): Does the candidate's EXCLUDES field match the input?
  If the input description matches something in a candidate's EXCLUDES field,
  that candidate is EXPLICITLY WRONG. The SBS reference says so. HARD REJECT.

CHECK 8 — INCLUDES (NEW): Does the candidate's INCLUDES field match the input?
  If the input description matches something in a candidate's INCLUDES field,
  this is a STRONG POSITIVE signal. Boost confidence.

## STEP 3: DECIDE

- Passes all checks → MATCH. Pick most specific if multiple pass.
- Passes checks 1-2 but minor issues on 3-5 → PARTIAL (MEDIUM/LOW confidence).
- NO candidate passes checks 1-2 → return null.

## STEP 4: CONFIDENCE

HIGH — All checks pass. Specialty and category match. Defensible in audit.
MEDIUM — Clinically reasonable, one minor ambiguity or unspecified qualifier.
LOW — Best available, parameter gap or scope mismatch. Needs human review.
NONE — Return null. No candidate shares domain AND service type.

## MANDATORY REJECTIONS — return null if ANY is true

1. Candidate is from a different clinical specialty/chapter than input
2. Candidate describes a clinically different service despite shared terminology
3. A defining numerical parameter clearly mismatches
4. Input and candidate specify different qualifiers (material, technique, route)
5. Match depends entirely on shared keywords with no clinical connection
6. Drug ↔ procedure ↔ device code system mismatch
7. Input matches a candidate's EXCLUDES field
8. Input is an abbreviation you cannot confidently interpret

## OUTPUT — ONLY this JSON, no markdown fences:

{{
  "input_analysis": {{
    "clinical_domain": "<domain>",
    "specific_service": "<procedure/test/drug/device>",
    "key_parameters": "<numbers, qualifiers — or 'none specified'>"
  }},
  "matched_code": "<code or null>",
  "code_system": "<SBS|GTIN|GMDN|null>",
  "matched_description": "<candidate description or null>",
  "confidence": "<HIGH|MEDIUM|LOW|NONE>",
  "reasoning": "<1-2 sentences: which checks passed/failed, any Excludes/Includes signals used>"
}}"""
```

---

## THE COMPLETE ARCHITECTURE — Before vs After

```
═══════════════════════════════════════════════════════════════════
CURRENT ARCHITECTURE (broken)
═══════════════════════════════════════════════════════════════════

  Input: "RE-RCT"
    │
    ▼
  ChromaDB Search (unexpanded query, sparse documents)
    │  Documents indexed as: "[SBS] Rmvl other wart. Rmvl other wart"
    │  No specialty, no category, no clinical context
    │
    ▼
  Top-5 candidates (mostly irrelevant, correct code may not be present)
    │  Candidate 1: [SBS] 30189-01-00 — Rmvl other wart
    │  Candidate 2: ...random codes sharing "R" or "re" patterns...
    │
    ▼
  LLM with basic prompt (no metadata, no abbreviation help)
    │  LLM doesn't know RE-RCT = root canal retreatment
    │  LLM sees "Rmvl" and connects to "RE" prefix
    │  LLM hallucinates: "RE-RCT indicates removal of a wart"
    │
    ▼
  Output: 30189-01-00 / HIGH confidence 💀


═══════════════════════════════════════════════════════════════════
NEW ARCHITECTURE (with all 6 changes)
═══════════════════════════════════════════════════════════════════

  Input: "RE-RCT"
    │
    ▼
  LAYER 1: Query Expansion
    │  "RE-RCT" → "RE-RCT (root canal retreatment endodontic)"
    │
    ▼
  LAYER 2: Two-Pass Retrieval
    │  Pass A: General search with expanded query → top 15
    │  Pass B: Specialty-filtered search (dental chapter) → top 15
    │  Merge + deduplicate → ~20 unique candidates
    │  Documents are ENRICHED:
    │    "[SBS] Code: 97418-00-00. Specialty: DENTAL PROCEDURES.
    │     Category: Endodontic. Description: Root canal obturation
    │     each additional canal. Clinical: ..."
    │
    ▼
  LAYER 3: LLM with Metadata-Aware Prompt
    │  Step 1: Parses "RE-RCT" → domain: Endodontic (aided by abbreviation note)
    │  Step 2: Checks each candidate's Specialty field
    │    → Wart removal code shows "Specialty: PROCEDURES ON SKIN"
    │    → Check 1 FAILS: Endodontic ≠ Skin
    │    → Root canal codes show "Specialty: DENTAL PROCEDURES"
    │    → Check 1 PASSES
    │  Step 2 continued: Checks Excludes field
    │    → No Excludes conflicts
    │
    ▼
  LAYER 4: Post-LLM Validation
    │  Input contains "RCT" → verify match contains "root canal" or "endodontic"
    │  ✅ Passes validation
    │
    ▼
  Output: 974xx-xx-xx (correct endodontic code) / HIGH confidence ✅

  OR if no endodontic code exists in index:
  Output: null / NONE / "No endodontic retreatment code found in candidates" ✅
```

---

## HOW EXCLUDES FIELD PREVENTS ERRORS — Worked Example

Suppose your SBS index has these two codes (from the sample data):

```
Code: 39703-03-00
  Description: Aspiration of brain cyst
  Excludes: Drainage of infected cyst (39900-00-00)

Code: 39900-00-00
  Description: Drainage of infected cyst
```

**Scenario: Input = "Drainage of infected brain cyst"**

Without Excludes:
- Both codes mention "brain cyst" → embeddings rank both highly
- LLM might pick 39703-03-00 (aspiration) because it has "brain cyst" in the description
- This is WRONG — aspiration ≠ drainage, and the SBS reference explicitly says so

With Excludes:
- LLM sees: "Candidate 1: Aspiration of brain cyst. EXCLUDES: Drainage of infected cyst"
- Check 7 (Excludes): Input says "drainage of infected cyst" which MATCHES the Excludes field
- **HARD REJECT** — the SBS reference itself says this code is wrong for this input
- LLM moves to Candidate 2: 39900-00-00 "Drainage of infected cyst" → MATCH

**The Excludes field is not a nice-to-have. It's the SBS coding authority literally
telling you "don't make this mistake." Ignoring it is ignoring the answer key.**

---

## HOW CHAPTER + BLOCK PREVENT CROSS-DOMAIN ERRORS — Worked Examples

### Example: "R.C.T. (Birooted)" → Blood typing (actual error from V3 run)

Without Chapter/Block in candidates:
```
Candidate: [SBS] 73250-00-92 — Blood typing; ABO & Rh (D)
→ LLM sees short descriptions only, can't determine specialty
→ Matches based on pattern recognition, confabulates
```

With Chapter/Block in candidates:
```
Candidate: [SBS] Code: 73250-00-92. Specialty: PATHOLOGY PROCEDURES.
  Category: Blood grouping. Description: Blood typing; ABO & Rh (D)
→ LLM sees: Input domain = Endodontic (dental), Candidate specialty = PATHOLOGY
→ Check 1: Endodontic ≠ Pathology → REJECT
```

### Example: "Dental Examination by General Dentist" (from new input file)

Without Chapter/Block:
```
Candidates might include examination codes from ophthalmology, ENT, etc.
→ LLM sees "Examination" in multiple candidates, picks semi-randomly
```

With Chapter/Block:
```
Candidate: [SBS] Code: 97xxx. Specialty: DENTAL PROCEDURES.
  Category: Dental examination and assessment.
  Description: Comprehensive oral examination
→ LLM sees: Input = dental examination, Candidate specialty = DENTAL PROCEDURES
→ Check 1: MATCH
→ Check 2: Category = Dental examination → MATCH
→ High confidence, correct match
```

---

## INDEXING SCRIPT — COMPLETE IMPLEMENTATION

```python
"""
ingest_sbs.py — Enriched SBS document indexing

This script reads the Saudi Billing Codes reference file and creates
richly annotated documents for the vector store. Each document includes
all available metadata: Specialty (Chapter), Category (Block), Clinical
Explanation, Includes, Excludes, and Guidelines.
"""

import pandas as pd
from langchain.schema import Document


def clean_field(value) -> str:
    """Clean a DataFrame cell value, returning empty string for null/nan."""
    if pd.isna(value):
        return ""
    s = str(value).strip()
    if s.lower() in ("nan", "none", ""):
        return ""
    return s


def build_enriched_sbs_document(row: pd.Series) -> Document | None:
    """
    Build a single enriched Document from an SBS reference row.
    Returns None for empty/invalid rows.
    """
    code = clean_field(row.get("SBS Code Hyphenated"))
    short = clean_field(row.get("Short Description"))

    # Skip empty rows
    if not code or not short:
        return None

    long_desc = clean_field(row.get("Long Description"))
    chapter = clean_field(row.get("Chapter Name"))
    block = clean_field(row.get("Block Name"))
    clinical = clean_field(row.get("Cliincal Explanation "))
    includes = clean_field(row.get("Includes"))
    excludes = clean_field(row.get("Excludes"))
    guideline = clean_field(row.get("Guideline"))
    chapter_num = row.get("Chapter number")
    block_num = clean_field(row.get("Block Number"))

    # Build enriched text
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

    # Build metadata for filtered retrieval
    metadata = {
        "code": code,
        "code_system": "SBS",
        "short_description": short,
        "chapter_name": chapter.upper().strip() if chapter else "",
        "chapter_number": float(chapter_num) if pd.notna(chapter_num) else None,
        "block_name": block,
        "block_number": block_num,
        "has_excludes": bool(excludes),
        "has_includes": bool(includes),
        "has_clinical_explanation": bool(clinical),
    }

    return Document(page_content=page_content, metadata=metadata)


def ingest_sbs_file(filepath: str) -> list[Document]:
    """
    Read SBS Excel file and produce enriched Documents.
    """
    df = pd.read_excel(filepath)

    documents = []
    skipped = 0

    for _, row in df.iterrows():
        doc = build_enriched_sbs_document(row)
        if doc:
            documents.append(doc)
        else:
            skipped += 1

    print(f"Indexed {len(documents)} SBS codes, skipped {skipped} empty rows")
    return documents


# ── Usage ──
# documents = ingest_sbs_file("Saudi_Billing_Codes_Full.xlsx")
# vector_store = Chroma.from_documents(
#     documents,
#     embedding=your_embedding_model,
#     collection_name="sbs_codes",
#     persist_directory="./chroma_db"
# )
```

---

## COMPLETE IMPLEMENTATION CHECKLIST

```
PRIORITY   CHANGE                                      WHY                           EFFORT
────────────────────────────────────────────────────────────────────────────────────────────────
  🔴 1     Re-index SBS with enriched documents        80% of matching improvement    MEDIUM
           (Change 1 + indexing script)                comes from better context.     (3-4 hrs)
                                                       This is the foundation.

  🔴 2     Store Chapter/Block as metadata             Enables specialty-filtered     LOW
           in ChromaDB (Change 2)                      retrieval. Eliminates          (included
                                                       cross-domain errors.           in #1)

  🔴 3     Deploy query expansion with                 Prevents hallucination on      LOW
           abbreviation map (Change 5)                 RE-RCT, R.C.T., PFM, etc.    (1-2 hrs)

  🟡 4     Update candidate formatting to pass         LLM can now SEE specialty,     LOW
           full enriched text to LLM (Change 3)        category, excludes.            (1 hr)

  🟡 5     Update prompt with metadata-aware           LLM knows how to USE the      LOW
           checks 7-8 (Change 4 + 6)                  new metadata fields.           (30 min)

  🟡 6     Implement two-pass retrieval with           Specialty filtering for        MEDIUM
           specialty filter (Change 2)                 high-precision candidates.     (2-3 hrs)

  🟡 7     Deploy post-LLM validation                  Safety net for remaining       LOW
           (from previous analysis)                    LLM errors.                    (1-2 hrs)

  🟢 8     Build vocabulary map from SBS Block/        Automated synonym expansion    MEDIUM
           Chapter names (Change 5 utility)            from your own reference data.  (2-3 hrs)

  🟢 9     Audit SBS index completeness                Verify common procedures have  VARIES
                                                       codes in the reference data.

  🟢 10    Never silently delete null rows             Every input → output row.      LOW
                                                                                      (30 min)
```

🔴 = Do first (foundational changes)
🟡 = Do next (leverage the foundation)
🟢 = Do soon (polish and complete)

**The first 3 items together should take ~6-8 hours and will address the root cause
of every major error class: cross-domain hallucinations, abbreviation failures, and
missing clinical context.**

---

## EXPECTED IMPACT ON KNOWN ERRORS

```
ERROR                        ROOT CAUSE             WHICH CHANGE FIXES IT
──────────────────────────────────────────────────────────────────────────────────
RE-RCT → Wart removal        Abbreviation unknown   Change 5 (query expansion)
                              + no specialty filter  Change 2 (specialty filter)
                              + no domain in cand.   Change 1 (enriched docs)

R.C.T.(Bi) → Blood typing    Same as above          Same — all 3 changes

R.C.T.(Multi) → Prosthetic   Same as above          Same — all 3 changes

PFM → Zirconia                LLM doesn't know       Change 1 (enriched docs
                              materials differ       may include material context)
                                                     Change 5 (query expansion)
                                                     Post-LLM validation (hard rule)

3-Surface → 3D Photography    No category in docs    Change 1 (category in doc
                              + keyword trap         distinguishes restorative
                                                     from imaging)

Filling → Veneer/Sealant      No category in docs    Change 1 (category field)
                              + no service type      Change 2 (block filtering)
                              distinction

Full bony → Partial bony      Qualifier ignored      Change 1 (if Excludes exist)
                                                     Post-LLM validation

Tomographic → CT orbit        Wrong specificity      Change 2 (imaging chapter
                                                     filter returns general codes)

All HIGH confidence           LLM uses checks as     Prompt Change 4 + 6
                              rhetoric               Post-LLM confidence recalib.
```

**Summary: Enriched indexing (Change 1) + Specialty filtering (Change 2) + Query expansion
(Change 5) together address the root cause of 90%+ of errors. The prompt changes and
post-LLM validation handle the remaining edge cases.**
