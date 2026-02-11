# Architecture Analysis: Query Normalization & Namespace Routing

## Overview of Proposed Method

The proposed architecture introduces a pre-processing step ("Query Refinement Node") before vector retrieval.

1.  **Input**: Raw User Description.
2.  **Step 1 (LLM)**: Normalize query, expand synonyms, and predict domain (SBS/GMDN/GTIN).
3.  **Step 2 (Pinecone)**: Query _only_ the predicted namespace (`sbs_v2`, etc.) using the Normalized Query.
4.  **Step 3 (LLM)**: Final selection from candidates.

This is a "Router Pattern" common in Agentic RAG. While powerful, it introduces specific risks in a high-throughput batch processing system.

---

## 🚨 Identified Problems

### 1. The "Hard Partition" Failure Mode (Critical)

**Problem:** The system relies on the assumption that the LLM's "Coding Domain Guess" is 100% correct.

- **Scenario:** A user provides a description like _"Titanium Screw 3.5mm"_.
- **Failure:** The LLM guesses `GTIN` (Product). The system searches _only_ the `gtin_v2` namespace.
- **Reality:** The correct code might actually be in `GMDN` (Medical Device Category).
- **Result:** The retrieval returns **zero relevant results** from the correct category because it never looked there. The final matching step has no chance to recover. **Classification Error = Total System Failure.**

### 2. Information Loss via "Normalization"

**Problem:** The prompt instructs the LLM to "rewrite... into formal terminology".

- **Scenario:** Input is _"Panadol Extra 500mg/65mg"_.
- **Failure:** LLM rewrites to _"Paracetamol and Caffeine oral tablet"_.
- **Result:** The specific brand name ("Panadol") or exact strength might be diluted. Vector search might retrieve generic paracetamol codes but miss the exact brand entry if the vector store relies heavily on exact keyword matches.
- **Risk:** Highly specific SKU/GTIN codes rely on exact string matches (Brand, Number, SKU) which "normalization" might strip away as "noise".

### 3. Latency & Cost Doubling

**Problem:** Every single row now requires **two** serial LLM calls.

- **Impact:** If batch processing 10,000 items:
  - Old Way: 10,000 LLM calls (Selection).
  - New Way: 20,000 LLM calls (Normalization + Selection).
- **Consequence:** Processing time doubles. Cost doubles. Rate limits (TPM/RPM) are hit twice as fast.

### 4. Namespace Availability

**Problem:** The prompt assumes data is already partitioned into `sbs_v2`, `gmdn_v2`, `gtin_v2` namespaces in Pinecone.

- **Status:** Unless the data was re-indexed specifically with these namespace tags, searching a namespace returns 0 results.

---

## ✅ Proposed Solutions

### Solution 1: "Soft Routing" (Recommended)

Instead of searching _only_ the predicted namespace, search **all** namespaces but prioritize the predicted one.

- **Mechanism:**
  - Run searches in parallel: `Namespace A (Top-10)` + `namespace B (Top-3)` + `Namespace C (Top-3)`.
  - **OR** (Cheaper): Search the "Global" index (if available) and use the predicted domain as a **Metadata Filter** rather than a namespace. If the filtered search returns low scores (<0.75), automatically fallback to an unfiltered search.

### Solution 2: Hybrid Query Construction

Don't replace the user input with the normalized query. **Combine them.**

- **Mechanism:**
  - `Search Query = "{User Input} {Normalized Query} {Synonyms}"`
- **Benefit:** Keeps the exact keywords (Brand names, sizes) from the input while gaining the semantic reach of the medical terms. Best of both worlds.

### Solution 3: Model Tiering (Cost/Speed Optimization)

Do not use the same expensive model (Claude 3.5 Sonnet) for Step 1.

- **Mechanism:**
  - **Step 1 (Normalization/Routing):** Use **Claude 3 Haiku** or **Gemini Flash**. It is 10x faster and cheaper, and sufficient for rewriting/classification.
  - **Step 2 (Selection):** Keep **Claude 3.5 Sonnet** for the high-precision final match.

### Solution 4: Fallback Recovery

If Step 1 (Classification) is "Unknown" or low confidence (if the model parses it as ambiguous), triggering a "Broadcast Search" (search all namespaces).

---

## 🛠 Implementation Plan

1.  **Refactor `matcher.py`**:
    - Remove ChromaDB logic.
    - Initialize **Pinecone** client.
    - Implement `normalize_query(text)` function using the new Prompt.
2.  **Update `prompts.py`**:
    - Add the `NORMALIZATION_PROMPT`.
3.  **Refactor `match_single`**:
    - Call `normalize_query`.
    - Parse the Domain Guess.
    - **Logic Change:** Search predicted namespace first. If results are poor (score < threshold), search others. (Solving Problem #1).
    - Perform Final Selection.
