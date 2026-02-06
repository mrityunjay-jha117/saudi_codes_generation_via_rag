**QUICKINTELL**

**Saudi Billing Code Matcher**

RAG-Powered POC

Architecture, Data Model & Implementation Guide

|                    |                           |
|--------------------|---------------------------|
| **Document**       | QI-POC-RAG-2026-001       |
| **Version**        | 2.0                       |
| **Date**           | February 2026             |
| **Classification** | Internal                  |
| **Audience**       | Intern / Junior Developer |

Table of Contents

1\. Executive Summary

A Saudi healthcare customer has provided three reference datasets of standardized billing codes: SBS (procedures), GTIN (drugs), and GMDN (devices). They need a tool that takes a spreadsheet of \~1,906 free-text service descriptions and automatically finds the best matching standardized code for each row.

This document specifies a RAG (Retrieval-Augmented Generation) approach: embed all reference code descriptions into a vector database, then for each input service description, retrieve the top candidate codes and use an LLM to intelligently select the best match. This eliminates the need for hand-tuned NLP rules and leverages the LLM\'s medical knowledge for disambiguation.

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<tbody>
<tr class="odd">
<td><p><strong>WHY RAG OVER TRADITIONAL NLP</strong></p>
<p>1) Zero rule-writing: No regex, no abbreviation dictionaries, no fuzzy-threshold tuning. The LLM understands that 'Turbinectomy-Partial Excision' matches 'Partial turbinectomy' without any hand-coded logic. 2) Handles ambiguity: The LLM can reason about whether 'BEPANTHEN cream' is a drug (GTIN) or a procedure (SBS). 3) Scalable: Adding new code systems means adding new embeddings, not rewriting matching pipelines. 4) Explainable: The LLM provides reasoning for each match, not just a score.</p></td>
</tr>
</tbody>
</table>

2\. Scope & Data Model

2.1 What Goes In, What Comes Out

**INPUT:** An Excel file (Code Mapping Sheet) with columns **Service Code** and **Service Description**. Total: 1,906 rows across 3 sheets.

**OUTPUT:** The same Excel file with three new columns filled in: **Matched Code** (the SBS Code / GTIN Code / GMDN termCode), **Code System** (SBS / GTIN / GMDN), and **Confidence** (HIGH / MEDIUM / LOW / NONE).

2.2 Reference Data: Only the Fields That Matter

We strip each reference dataset down to the minimum fields needed for matching and output:

SBS (Procedures) - 2 match fields, 1 output field

| **Field**           | **Role**                                           | **Example**        |
|---------------------|----------------------------------------------------|--------------------|
| Short Description   | PRIMARY match target                               | Cisternal puncture |
| Long Description    | SECONDARY match target (when different from short) | Cisternal puncture |
| SBS Code Hyphenated | OUTPUT value (written to result)                   | 39003-00-00        |

GTIN (Drugs) - 3 match fields, 1 output field

| **Field**   | **Role**                                        | **Example**   |
|-------------|-------------------------------------------------|---------------|
| DISPLAY     | PRIMARY match target (brand/trade name)         | BEPANTHEN     |
| INGREDIENTS | SECONDARY match target (generic name)           | DEXPANTHENOL  |
| STRENGTH    | TERTIARY match target (disambiguates same drug) | 5 G/ 100 G    |
| CODE        | OUTPUT value (GTIN barcode)                     | 6285074000864 |

GMDN (Devices) - 2 match fields, 1 output field

| **Field**      | **Role**                                  | **Example**                               |
|----------------|-------------------------------------------|-------------------------------------------|
| termName       | PRIMARY match target                      | 1,25-Dihydroxy vitamin D3 IVD, calibrator |
| termDefinition | SECONDARY match target (full description) | A material which is used to establish\... |
| GMDN_termCode  | OUTPUT value                              | 38242                                     |

2.3 How Reference Data Becomes Vector Documents

Each row in each reference dataset is converted into a single text document for embedding. This is the most critical design decision:

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<tbody>
<tr class="odd">
<td><p><strong>DOCUMENT CONSTRUCTION FORMULA</strong></p>
<p>SBS: "[SBS] {Short Description}. {Long Description}"
GTIN: "[GTIN] {DISPLAY} - {INGREDIENTS} {STRENGTH}"
GMDN: "[GMDN] {termName}. {termDefinition}"
The [SBS], [GTIN], [GMDN] prefix tags help the embedding model and the LLM understand which code system each document belongs to.</p></td>
</tr>
</tbody>
</table>

Concrete examples from the actual data:

| **System** | **Constructed Document Text**                                                                                           | **Metadata (stored alongside)**               |
|------------|-------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------|
| SBS        | \[SBS\] Cisternal puncture. Cisternal puncture                                                                          | { code: \'39003-00-00\', system: \'SBS\' }    |
| SBS        | \[SBS\] Insertion of external ventricular drain. Insertion of external ventricular drain                                | { code: \'39015-00-00\', system: \'SBS\' }    |
| GTIN       | \[GTIN\] ZOVIRAX - ACYCLOVIR (ACICLOVIR) 5 G/ 100 G                                                                     | { code: \'6285096000200\', system: \'GTIN\' } |
| GTIN       | \[GTIN\] AIRFAST 4MG GRANULES - MONTELUKAST(AS SODIUM) 4 MG                                                             | { code: \'6285147000012\', system: \'GTIN\' } |
| GMDN       | \[GMDN\] 1,25-Dihydroxy vitamin D3 IVD, calibrator. A material which is used to establish known points of reference\... | { code: \'38242\', system: \'GMDN\' }         |

3\. RAG Architecture

3.1 System Overview

The system has exactly two phases: an Indexing Phase (runs once when reference data is loaded) and a Matching Phase (runs for every input row).

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<tbody>
<tr class="odd">
<td><p><strong>PHASE 1: INDEXING (One-Time Setup)</strong></p>
<p>[Reference Excel] ---&gt; [Document Builder] ---&gt; [Embedding Model] ---&gt; [ChromaDB]</p>
<p>SBS/GTIN/GMDN Build text docs OpenAI/Local Vector Store</p>
<p><strong>PHASE 2: MATCHING (Per Row)</strong></p>
<p>+----&gt; [Top-K Retrieval from ChromaDB]</p>
<p>[Service Desc] ---&gt;|</p>
<p>+----&gt; [LLM Prompt: "Which of these codes best matches?"]</p>
<p>|</p>
<p>v</p>
<p>[Structured JSON Response]</p>
<p>{ code, system, confidence, reasoning }</p></td>
</tr>
</tbody>
</table>

3.2 Technology Stack

| **Component**   | **Technology**                                            | **Why**                                                              |
|-----------------|-----------------------------------------------------------|----------------------------------------------------------------------|
| Language        | Python 3.11+                                              | Ecosystem: LangChain, ChromaDB, pandas all Python-native             |
| Vector Store    | ChromaDB                                                  | Zero-config, runs in-process, persistent storage, built-in embedding |
| Embedding Model | OpenAI text-embedding-3-small (or all-MiniLM-L6-v2 local) | OpenAI for best quality; local MiniLM as free fallback               |
| LLM             | OpenAI GPT-4o-mini (or Claude 3.5 Sonnet via API)         | Cheapest high-quality LLM; \$0.15/1M input tokens                    |
| Orchestration   | LangChain                                                 | Handles embedding, retrieval, prompt chaining out of the box         |
| Excel I/O       | openpyxl + pandas                                         | Read/write .xlsx natively                                            |
| UI              | Streamlit (single page)                                   | Upload file, click process, download result. 30 lines of code.       |

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<tbody>
<tr class="odd">
<td><p><strong>COST ESTIMATE FOR FULL RUN</strong></p>
<p>Reference data: 27 docs x ~100 tokens avg = 2,700 tokens to embed (one-time, ~$0.0001). Matching: 1,906 rows x ~500 tokens per prompt (top-5 candidates + instruction) = ~953K input tokens + ~100 tokens output each = ~1.1M tokens total. At GPT-4o-mini pricing ($0.15/1M in, $0.60/1M out): Total cost &lt; $0.30 for the entire batch. With local models: $0.00.</p></td>
</tr>
</tbody>
</table>

4\. Step-by-Step Implementation

4.1 Project Structure

> saudi-code-matcher/
>
> \|\-- .env \# API keys (OPENAI_API_KEY)
>
> \|\-- requirements.txt
>
> \|\-- config.py \# Thresholds, model names, paths
>
> \|\-- ingest.py \# Phase 1: Read Excel, build vector DB
>
> \|\-- matcher.py \# Phase 2: RAG matching engine
>
> \|\-- app.py \# Streamlit UI (upload/process/download)
>
> \|\-- prompts.py \# LLM prompt templates
>
> \|
>
> \|\-- data/
>
> \| \|\-- reference/ \# Saudi_Billing_Codes_Sample.xlsx
>
> \| \|\-- input/ \# Code_Mapping_Sheet_Sample.xlsx
>
> \| \|\-- output/ \# Matched results
>
> \|
>
> \|\-- chroma_db/ \# Persistent vector store (auto-created)
>
> \|
>
> \|\-- tests/
>
> \| \|\-- test_ingest.py
>
> \| \|\-- test_matcher.py
>
> \| \|\-- test_prompts.py

**That is it.** No complex module hierarchy. 5 Python files + 1 config. This is a POC.

4.2 requirements.txt

> \# Core
>
> langchain==0.3.0
>
> langchain-openai==0.2.0
>
> langchain-community==0.3.0
>
> chromadb==0.5.0
>
> \# Data
>
> pandas==2.2.0
>
> openpyxl==3.1.2
>
> \# UI
>
> streamlit==1.38.0
>
> \# (Optional) Local embedding fallback
>
> sentence-transformers==3.0.0
>
> \# Testing
>
> pytest==8.3.0
>
> python-dotenv==1.0.0

4.3 config.py

> import os
>
> from dotenv import load_dotenv
>
> load_dotenv()
>
> \# === LLM & Embedding ===
>
> OPENAI_API_KEY = os.getenv(\"OPENAI_API_KEY\")
>
> EMBEDDING_MODEL = \"text-embedding-3-small\" \# OpenAI
>
> \# EMBEDDING_MODEL = \"all-MiniLM-L6-v2\" \# Uncomment for local/free
>
> LLM_MODEL = \"gpt-4o-mini\" \# Cheap + smart
>
> LLM_TEMPERATURE = 0.0 \# Deterministic matching
>
> \# === Retrieval ===
>
> TOP_K = 10 \# Retrieve top 10 candidates from vector DB
>
> CHROMA_PERSIST_DIR = \"./chroma_db\"
>
> COLLECTION_NAME = \"saudi_billing_codes\"
>
> \# === Confidence Thresholds (set by LLM in response) ===
>
> \# HIGH = LLM is very sure this is the correct match
>
> \# MEDIUM = LLM thinks this is likely correct but not certain
>
> \# LOW = Best guess, needs human review
>
> \# NONE = No reasonable match found
>
> \# === Paths ===
>
> REFERENCE_FILE = \"./data/reference/Saudi_Billing_Codes_Sample.xlsx\"
>
> INPUT_DIR = \"./data/input/\"
>
> OUTPUT_DIR = \"./data/output/\"

4.4 ingest.py - Vector Database Builder

This file reads the reference Excel, constructs text documents, and stores them in ChromaDB. Run this ONCE whenever reference data changes.

> import pandas as pd
>
> import chromadb
>
> from langchain_openai import OpenAIEmbeddings
>
> from langchain_community.vectorstores import Chroma
>
> from langchain.schema import Document
>
> import config
>
> def build_documents(excel_path: str) -\> list\[Document\]:
>
> \"\"\"Read reference Excel and build LangChain Documents.\"\"\"
>
> xls = pd.ExcelFile(excel_path)
>
> docs = \[\]
>
> \# ── SBS Documents ──
>
> df_sbs = pd.read_excel(xls, sheet_name=\"SBS V3 Tabular List\")
>
> df_sbs = df_sbs.dropna(subset=\[\"SBS Code Hyphenated\"\])
>
> for \_, row in df_sbs.iterrows():
>
> short = str(row\[\"Short Description\"\]).strip()
>
> long = str(row\[\"Long Description\"\]).strip()
>
> text = f\"\[SBS\] {short}. {long}\" if short != long else f\"\[SBS\] {short}\"
>
> docs.append(Document(
>
> page_content=text,
>
> metadata={
>
> \"code\": str(row\[\"SBS Code Hyphenated\"\]),
>
> \"system\": \"SBS\",
>
> \"description\": short,
>
> }
>
> ))
>
> \# ── GTIN Documents ──
>
> df_gtin = pd.read_excel(xls, sheet_name=\"GTIN\")
>
> for \_, row in df_gtin.iterrows():
>
> display = str(row\[\"DISPLAY\"\]).strip()
>
> ingr = str(row\[\"INGREDIENTS\"\]).strip()
>
> stren = str(row\[\"STRENGTH\"\]).strip()
>
> text = f\"\[GTIN\] {display} - {ingr} {stren}\"
>
> docs.append(Document(
>
> page_content=text,
>
> metadata={
>
> \"code\": str(row\[\"CODE\"\]),
>
> \"system\": \"GTIN\",
>
> \"description\": display,
>
> }
>
> ))
>
> \# ── GMDN Documents ──
>
> df_gmdn = pd.read_excel(xls, sheet_name=\"GMDN\")
>
> for \_, row in df_gmdn.iterrows():
>
> name = str(row\[\"termName\"\]).strip()
>
> defn = str(row\[\"termDefinition\"\]).strip()
>
> \# Truncate long definitions to 300 chars for embedding efficiency
>
> defn_short = defn\[:300\] if len(defn) \> 300 else defn
>
> text = f\"\[GMDN\] {name}. {defn_short}\"
>
> docs.append(Document(
>
> page_content=text,
>
> metadata={
>
> \"code\": str(row\[\"GMDN_termCode\"\]),
>
> \"system\": \"GMDN\",
>
> \"description\": name,
>
> }
>
> ))
>
> return docs
>
> def create_vector_store(docs: list\[Document\]) -\> Chroma:
>
> \"\"\"Embed documents and persist to ChromaDB.\"\"\"
>
> embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
>
> vector_store = Chroma.from_documents(
>
> documents=docs,
>
> embedding=embeddings,
>
> collection_name=config.COLLECTION_NAME,
>
> persist_directory=config.CHROMA_PERSIST_DIR,
>
> )
>
> print(f\"Indexed {len(docs)} documents into ChromaDB\")
>
> return vector_store
>
> if \_\_name\_\_ == \"\_\_main\_\_\":
>
> docs = build_documents(config.REFERENCE_FILE)
>
> print(f\"Built {len(docs)} documents:\")
>
> for d in docs\[:3\]:
>
> print(f\" {d.page_content\[:80\]}\... \| code={d.metadata\[\'code\'\]}\")
>
> create_vector_store(docs)
>
> print(\"Done! Vector store saved to\", config.CHROMA_PERSIST_DIR)

4.5 prompts.py - LLM Prompt Templates

This is the brain of the system. The prompt must instruct the LLM to: analyze the input, compare it against retrieved candidates, pick the best match, and return structured JSON.

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<tbody>
<tr class="odd">
<td><p><strong>CRITICAL DESIGN DECISION: WHY THE LLM MATTERS</strong></p>
<p>Vector similarity alone will retrieve the top-10 candidates, but it cannot reason. For example, if the input is 'BEPANTHEN cream 30g' and the vector DB returns both BEPANTHEN Ointment (code A) and BEPANTHEN Cream (code B), only the LLM can read 'cream' in the input and pick code B. Similarly, 'Tap for subdural haemorrhage' is semantically close to both 'Ventricular puncture' and 'Cisternal puncture' - the LLM uses medical knowledge to disambiguate.</p></td>
</tr>
</tbody>
</table>

> MATCH_PROMPT = \"\"\"You are a Saudi healthcare billing code expert.
>
> Your job is to match a service description to the best standardized code.
>
> The three code systems are:
>
> \- SBS: Saudi Billing System for medical/surgical procedures
>
> \- GTIN: Global Trade Item Number for pharmaceutical drugs
>
> \- GMDN: Global Medical Device Nomenclature for devices/equipment
>
> \## Input Service Description
>
> {service_description}
>
> \## Candidate Codes (retrieved from database)
>
> {candidates}
>
> \## Instructions
>
> 1\. Analyze the service description carefully.
>
> 2\. Compare it against EACH candidate code above.
>
> 3\. Select the BEST match based on medical/clinical meaning, not just
>
> surface-level word overlap.
>
> 4\. If NONE of the candidates are a reasonable match, set matched_code
>
> to null.
>
> Respond with ONLY this JSON (no markdown, no explanation):
>
> {{
>
> \"matched_code\": \"\<the code value or null\>\",
>
> \"code_system\": \"\<SBS\|GTIN\|GMDN\|null\>\",
>
> \"matched_description\": \"\<official description from candidate\>\",
>
> \"confidence\": \"\<HIGH\|MEDIUM\|LOW\|NONE\>\",
>
> \"reasoning\": \"\<1 sentence explaining why this match was chosen\>\"
>
> }}
>
> \"\"\"
>
> def format_candidates(retrieved_docs) -\> str:
>
> \"\"\"Format retrieved documents into a numbered list for the prompt.\"\"\"
>
> lines = \[\]
>
> for i, doc in enumerate(retrieved_docs, 1):
>
> meta = doc.metadata
>
> lines.append(
>
> f\"{i}. \[{meta\[\'system\'\]}\] Code: {meta\[\'code\'\]} \| {doc.page_content}\"
>
> )
>
> return \"\n\".join(lines)

4.6 matcher.py - The RAG Matching Engine

This is the core orchestrator. For each service description, it: (1) retrieves top-K candidates from ChromaDB, (2) sends them to the LLM with the matching prompt, (3) parses the structured JSON response.

> import json
>
> import pandas as pd
>
> from langchain_openai import ChatOpenAI, OpenAIEmbeddings
>
> from langchain_community.vectorstores import Chroma
>
> import config
>
> from prompts import MATCH_PROMPT, format_candidates
>
> class CodeMatcher:
>
> def \_\_init\_\_(self):
>
> \"\"\"Load vector store and LLM.\"\"\"
>
> self.embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
>
> self.vector_store = Chroma(
>
> collection_name=config.COLLECTION_NAME,
>
> persist_directory=config.CHROMA_PERSIST_DIR,
>
> embedding_function=self.embeddings,
>
> )
>
> self.llm = ChatOpenAI(
>
> model=config.LLM_MODEL,
>
> temperature=config.LLM_TEMPERATURE,
>
> )
>
> def match_single(self, service_description: str) -\> dict:
>
> \"\"\"Match a single service description to a billing code.\"\"\"
>
> \# Step 1: Retrieve top-K candidates
>
> results = self.vector_store.similarity_search(
>
> service_description,
>
> k=config.TOP_K,
>
> )
>
> if not results:
>
> return {
>
> \"matched_code\": None,
>
> \"code_system\": None,
>
> \"matched_description\": None,
>
> \"confidence\": \"NONE\",
>
> \"reasoning\": \"No candidates found in vector database\",
>
> }
>
> \# Step 2: Build prompt with candidates
>
> candidates_text = format_candidates(results)
>
> prompt = MATCH_PROMPT.format(
>
> service_description=service_description,
>
> candidates=candidates_text,
>
> )
>
> \# Step 3: Call LLM
>
> response = self.llm.invoke(prompt)
>
> \# Step 4: Parse JSON response
>
> try:
>
> result = json.loads(response.content)
>
> except json.JSONDecodeError:
>
> \# Fallback: use top retrieval result
>
> top = results\[0\]
>
> result = {
>
> \"matched_code\": top.metadata\[\"code\"\],
>
> \"code_system\": top.metadata\[\"system\"\],
>
> \"matched_description\": top.metadata\[\"description\"\],
>
> \"confidence\": \"LOW\",
>
> \"reasoning\": \"LLM response parsing failed; using vector similarity\",
>
> }
>
> return result
>
> def match_batch(self, input_excel_path: str, output_excel_path: str):
>
> \"\"\"Process an entire mapping sheet.\"\"\"
>
> xls = pd.ExcelFile(input_excel_path)
>
> all_results = \[\]
>
> for sheet_name in xls.sheet_names:
>
> df = pd.read_excel(xls, sheet_name=sheet_name)
>
> print(f\"Processing sheet: {sheet_name} ({len(df)} rows)\")
>
> matched_codes = \[\]
>
> code_systems = \[\]
>
> confidences = \[\]
>
> descriptions = \[\]
>
> reasonings = \[\]
>
> for idx, row in df.iterrows():
>
> desc = str(row\[\"Service Description\"\])
>
> print(f\" \[{idx+1}/{len(df)}\] {desc\[:60\]}\...\")
>
> result = self.match_single(desc)
>
> matched_codes.append(result.get(\"matched_code\", \"\"))
>
> code_systems.append(result.get(\"code_system\", \"\"))
>
> confidences.append(result.get(\"confidence\", \"NONE\"))
>
> descriptions.append(result.get(\"matched_description\", \"\"))
>
> reasonings.append(result.get(\"reasoning\", \"\"))
>
> df\[\"Matched Code\"\] = matched_codes
>
> df\[\"Code System\"\] = code_systems
>
> df\[\"Matched Description\"\] = descriptions
>
> df\[\"Confidence\"\] = confidences
>
> df\[\"LLM Reasoning\"\] = reasonings
>
> all_results.append((sheet_name, df))
>
> \# Write all sheets to output Excel
>
> with pd.ExcelWriter(output_excel_path, engine=\"openpyxl\") as writer:
>
> for sheet_name, df in all_results:
>
> df.to_excel(writer, sheet_name=sheet_name, index=False)
>
> print(f\"Results saved to {output_excel_path}\")
>
> return all_results

4.7 app.py - Streamlit UI (Minimal)

The entire UI is a single Streamlit page: upload a file, click a button, download the result.

> import streamlit as st
>
> import os, tempfile
>
> from matcher import CodeMatcher
>
> from ingest import build_documents, create_vector_store
>
> import config
>
> st.set_page_config(page_title=\"Saudi Code Matcher\", layout=\"centered\")
>
> st.title(\"Saudi Billing Code Matcher\")
>
> st.caption(\"Upload a mapping sheet. Get matched codes back.\")
>
> \# ── Sidebar: Reference Data Upload ──
>
> with st.sidebar:
>
> st.header(\"Reference Data\")
>
> ref_file = st.file_uploader(\"Upload Saudi Billing Codes Excel\", type=\"xlsx\")
>
> if ref_file and st.button(\"Index Reference Data\"):
>
> with tempfile.NamedTemporaryFile(suffix=\".xlsx\", delete=False) as f:
>
> f.write(ref_file.read())
>
> ref_path = f.name
>
> with st.spinner(\"Building vector index\...\"):
>
> docs = build_documents(ref_path)
>
> create_vector_store(docs)
>
> st.success(f\"Indexed {len(docs)} codes!\")
>
> \# ── Main: Mapping Sheet Processing ──
>
> mapping_file = st.file_uploader(\"Upload Code Mapping Sheet\", type=\"xlsx\")
>
> if mapping_file and st.button(\"Match Codes\", type=\"primary\"):
>
> with tempfile.NamedTemporaryFile(suffix=\".xlsx\", delete=False) as f:
>
> f.write(mapping_file.read())
>
> input_path = f.name
>
> output_path = os.path.join(config.OUTPUT_DIR, \"matched_results.xlsx\")
>
> os.makedirs(config.OUTPUT_DIR, exist_ok=True)
>
> matcher = CodeMatcher()
>
> progress = st.progress(0, text=\"Matching\...\")
>
> \# Run matching (with progress updates)
>
> results = matcher.match_batch(input_path, output_path)
>
> progress.progress(100, text=\"Done!\")
>
> \# Show summary
>
> for sheet_name, df in results:
>
> st.subheader(f\"Sheet: {sheet_name}\")
>
> conf_counts = df\[\"Confidence\"\].value_counts()
>
> st.write(conf_counts)
>
> st.dataframe(df.head(10))
>
> \# Download button
>
> with open(output_path, \"rb\") as f:
>
> st.download_button(\"Download Matched Results\", f, \"matched_results.xlsx\")

5\. Output Format

The output Excel preserves the original input columns and adds exactly 5 new columns:

| **Column**                     | **Source**                               | **Example**                                                       |
|--------------------------------|------------------------------------------|-------------------------------------------------------------------|
| Service Code (original)        | Unchanged from input                     | FMORT-0706                                                        |
| Service Description (original) | Unchanged from input                     | COSTOPLASTY                                                       |
| Matched Code (NEW)             | LLM selected from candidates             | 39703-03-00                                                       |
| Code System (NEW)              | Which system the code belongs to         | SBS                                                               |
| Matched Description (NEW)      | Official description from reference data | Aspiration of brain cyst                                          |
| Confidence (NEW)               | LLM self-assessed confidence             | MEDIUM                                                            |
| LLM Reasoning (NEW)            | 1-line explanation                       | Costoplasty is a rib surgical procedure, closest SBS match is\... |

**Note:** The original NPHIES Code, Description, and Other Code Value columns from the input sheet are **dropped from the output** since they were empty. They are replaced by the more descriptive Matched Code, Code System, and Matched Description columns.

6\. RAG Flow: End-to-End Walkthrough

Let us trace exactly what happens for one real input row to make the entire flow concrete.

Example: Matching \'Cisternal puncture\'

Step 1: Input Row

| **Service Code** | **Service Description** |
|------------------|-------------------------|
| FMORT-0706       | Cisternal puncture      |

Step 2: Vector Retrieval (Top 5)

The service description \'Cisternal puncture\' is embedded and searched against ChromaDB. The top 5 nearest documents returned:

| **Rank** | **Score** | **Document**                                       | **Code**    | **System** |
|----------|-----------|----------------------------------------------------|-------------|------------|
| 1        | 0.96      | \[SBS\] Cisternal puncture                         | 39003-00-00 | SBS        |
| 2        | 0.82      | \[SBS\] Ventricular puncture                       | 39006-00-00 | SBS        |
| 3        | 0.78      | \[SBS\] Other cranial puncture                     | 90000-00-00 | SBS        |
| 4        | 0.71      | \[SBS\] Tap for subdural haemorrhage               | 39009-00-00 | SBS        |
| 5        | 0.45      | \[GMDN\] 1,25-Dihydroxy vitamin D3 IVD, calibrator | 38242       | GMDN       |

Step 3: LLM Prompt Sent

The 5 candidates above are formatted and injected into the MATCH_PROMPT template along with the input description. The LLM receives a complete context window.

Step 4: LLM Response

> {
>
> \"matched_code\": \"39003-00-00\",
>
> \"code_system\": \"SBS\",
>
> \"matched_description\": \"Cisternal puncture\",
>
> \"confidence\": \"HIGH\",
>
> \"reasoning\": \"Exact semantic match - cisternal puncture is a specific
>
> cranial procedure involving needle placement below the occipital bone.\"
>
> }

Step 5: Output Row Written

| **Service Code** | **Service Description** | **Matched Code** | **Code System** | **Confidence** |
|------------------|-------------------------|------------------|-----------------|----------------|
| FMORT-0706       | Cisternal puncture      | 39003-00-00      | SBS             | HIGH           |

Example: Matching \'BEPANTHEN cream 30g\'

Step 1-2: Retrieval returns GTIN candidates

| **Rank** | **Score** | **Document**                                            | **Code**      |
|----------|-----------|---------------------------------------------------------|---------------|
| 1        | 0.94      | \[GTIN\] BEPANTHEN - DEXPANTHENOL 5 G/ 100 G (Cream)    | 6285074000857 |
| 2        | 0.91      | \[GTIN\] BEPANTHEN - DEXPANTHENOL 5 G/ 100 G (Ointment) | 6285074000864 |
| 3        | 0.52      | \[GTIN\] ZOVIRAX - ACYCLOVIR 5 G/ 100 G                 | 6285096000200 |

Step 3: LLM Reasons and Disambiguates

> {
>
> \"matched_code\": \"6285074000857\",
>
> \"code_system\": \"GTIN\",
>
> \"matched_description\": \"BEPANTHEN\",
>
> \"confidence\": \"HIGH\",
>
> \"reasoning\": \"Input specifies BEPANTHEN cream. Code 6285074000857 is the cream
>
> formulation while 6285074000864 is the ointment. Selected cream variant.\"
>
> }

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<tbody>
<tr class="odd">
<td><p><strong>KEY INSIGHT: THIS IS WHY RAG &gt; PURE VECTOR SEARCH</strong></p>
<p>Pure vector similarity scored both BEPANTHEN Cream (0.94) and BEPANTHEN Ointment (0.91) nearly equally. The LLM read 'cream' in the input text and correctly selected the cream variant. No amount of fuzzy matching threshold tuning would achieve this reliably.</p></td>
</tr>
</tbody>
</table>

7\. Edge Cases & How RAG Handles Them

| **Edge Case**             | **Example Input**                     | **RAG Behavior**                                                         |
|---------------------------|---------------------------------------|--------------------------------------------------------------------------|
| Abbreviations             | ADD\'L FEE FOR COM.OR OVER 3HRS       | LLM understands medical abbreviations natively. No dictionary needed.    |
| Word order mismatch       | Knee arthrotomy vs Arthrotomy of knee | Embeddings capture semantic similarity regardless of word order.         |
| Brand vs generic drug     | Zovirax vs Acyclovir                  | Both terms appear in the same GTIN document, so retrieval finds it.      |
| Same drug, different form | BEPANTHEN cream vs ointment           | LLM reads dosage form from input and picks correct GTIN code.            |
| No match exists           | Proprietary internal procedure name   | LLM returns confidence=NONE with reasoning explaining no match found.    |
| Cross-system ambiguity    | I.M.Z. IMPLANT (device or procedure?) | Retrieval returns candidates from both SBS and GMDN; LLM picks best.     |
| Arabic text mixed in      | Partial Arabic service descriptions   | Embedding models handle multilingual text; LLM reasons across languages. |
| Very short input          | Neuroendoscopy                        | Exact or near-exact match in SBS, retrieved as rank 1.                   |
| Very long input           | CIFLOX 200MG-100ML I.V. INFUSION      | All tokens (brand, strength, form) match GTIN doc for Ciprofloxacin.     |

8\. Performance & Optimization

8.1 Bottleneck Analysis

| **Operation**               | **Time Per Row** | **Total (1,906 rows)** | **Bottleneck?** |
|-----------------------------|------------------|------------------------|-----------------|
| Vector retrieval (ChromaDB) | \~5ms            | \~10 seconds           | No              |
| LLM API call (GPT-4o-mini)  | \~500ms          | \~16 minutes           | YES             |
| Excel read/write            | One-time         | \~2 seconds            | No              |

**The LLM API call is the clear bottleneck.** Here are three strategies to optimize, in order of implementation priority:

8.2 Optimization Strategy 1: Parallel API Calls

Send multiple LLM requests concurrently using Python\'s asyncio. Most LLM APIs allow 50-100 concurrent requests.

> import asyncio
>
> from langchain_openai import ChatOpenAI
>
> async def match_batch_async(descriptions: list\[str\], candidates_list):
>
> \"\"\"Process all descriptions concurrently.\"\"\"
>
> llm = ChatOpenAI(model=\"gpt-4o-mini\", temperature=0)
>
> async def match_one(desc, candidates):
>
> prompt = build_prompt(desc, candidates)
>
> response = await llm.ainvoke(prompt)
>
> return json.loads(response.content)
>
> \# Run 20 concurrent requests at a time
>
> semaphore = asyncio.Semaphore(20)
>
> async def throttled_match(desc, cands):
>
> async with semaphore:
>
> return await match_one(desc, cands)
>
> tasks = \[throttled_match(d, c) for d, c in zip(descriptions, candidates_list)\]
>
> return await asyncio.gather(\*tasks)

**Impact:** With 20 concurrent requests, 1,906 rows completes in \~1.5 minutes instead of 16 minutes.

8.3 Optimization Strategy 2: Batch Multiple Rows Per LLM Call

Instead of sending 1 service description per API call, send 5-10 at once. This reduces the number of API calls by 5-10x.

> \# In the prompt, send multiple descriptions at once:
>
> BATCH_PROMPT = \"\"\"Match each service description to the best code.
>
> Service Descriptions:
>
> 1\. {desc_1}
>
> 2\. {desc_2}
>
> 3\. {desc_3}
>
> \...
>
> Candidate Codes (shared pool):
>
> {all_candidates}
>
> Return a JSON array with one result per description:
>
> \[
>
> {{ \"input_index\": 1, \"matched_code\": \"\...\", \... }},
>
> {{ \"input_index\": 2, \"matched_code\": \"\...\", \... }},
>
> \]\"\"\"

**Impact:** Batching 5 rows per call = 382 API calls instead of 1,906. Combined with parallelism = \~20 seconds total.

8.4 Optimization Strategy 3: Skip LLM for High-Similarity Hits

If ChromaDB returns a candidate with similarity \> 0.95, skip the LLM call entirely and auto-accept.

> def match_single(self, desc: str) -\> dict:
>
> results = self.vector_store.similarity_search_with_score(desc, k=config.TOP_K)
>
> top_doc, top_score = results\[0\]
>
> if top_score \> 0.95: \# Near-exact match
>
> return {
>
> \"matched_code\": top_doc.metadata\[\"code\"\],
>
> \"code_system\": top_doc.metadata\[\"system\"\],
>
> \"confidence\": \"HIGH\",
>
> \"reasoning\": f\"Auto-matched (similarity={top_score:.2f})\",
>
> }
>
> \# Otherwise, use LLM for disambiguation
>
> return self.\_llm_match(desc, results)

9\. Testing Plan

9.1 Unit Tests

| **Test File**   | **What It Tests**         | **Key Assertions**                                                                      |
|-----------------|---------------------------|-----------------------------------------------------------------------------------------|
| test_ingest.py  | Document builder          | Correct number of docs created; metadata contains code and system; no NaN rows ingested |
| test_ingest.py  | Vector store creation     | ChromaDB collection has correct count; documents are retrievable                        |
| test_matcher.py | Single match (mocked LLM) | Returns valid JSON structure; confidence is one of HIGH/MEDIUM/LOW/NONE                 |
| test_matcher.py | JSON parse fallback       | When LLM returns invalid JSON, falls back to top retrieval result                       |
| test_prompts.py | Candidate formatting      | Numbered list output matches expected format; all metadata present                      |

9.2 Integration Tests (Golden Set)

Create 15 manually-verified test cases from the actual sample data:

| **Input Service Description**           | **Expected Code**              | **Expected System** |
|-----------------------------------------|--------------------------------|---------------------|
| Cisternal puncture                      | 39003-00-00                    | SBS                 |
| Ventricular puncture                    | 39006-00-00                    | SBS                 |
| Neuroendoscopy                          | 40903-00-00                    | SBS                 |
| Tap for subdural haemorrhage            | 39009-00-00                    | SBS                 |
| Insertion of external ventricular drain | 39015-00-00                    | SBS                 |
| BEPANTHEN                               | 6285074000864 or 6285074000857 | GTIN                |
| ZOVIRAX cream                           | 6285096000200 or 6285096000194 | GTIN                |
| Montelukast 4mg                         | 6285147000012                  | GTIN                |
| Metronidazole 500mg vaginal             | 6221075150078                  | GTIN                |
| Salbutamol 2mg tablets                  | 6251159026081                  | GTIN                |
| Vitamin D3 IVD calibrator               | 38242                          | GMDN                |
| Vitamin D3 IVD control                  | 38243                          | GMDN                |
| Beta-D-glucan IVD calibrator            | 45702                          | GMDN                |

9.3 Success Criteria

| **Metric**                   | **Target**                             | **How to Measure**                                   |
|------------------------------|----------------------------------------|------------------------------------------------------|
| Golden set accuracy (top-1)  | \> 85% (11/13 correct)                 | Run golden set, compare output code to expected      |
| Batch completion             | All 1,906 rows processed without crash | Run full batch, check output row count matches input |
| Processing time (with async) | \< 5 minutes for full batch            | Time the batch_match function                        |
| Cost per run                 | \< \$1.00                              | Check OpenAI usage dashboard after full run          |
| LLM response parse rate      | \> 95% valid JSON                      | Count JSON parse failures in logs                    |

10\. Sprint Plan: 2-Week POC

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<tbody>
<tr class="odd">
<td><p><strong>POC TIMELINE: 2 WEEKS (10 WORKING DAYS)</strong></p>
<p>This is intentionally compressed. RAG eliminates the need for algorithm tuning, abbreviation dictionaries, and multi-strategy scoring that a traditional NLP approach requires. The LLM does the heavy lifting.</p></td>
</tr>
</tbody>
</table>

Week 1: Build & Verify

| **Day** | **Task**                                                                | **Deliverable**                                           | **Done When**                                |
|---------|-------------------------------------------------------------------------|-----------------------------------------------------------|----------------------------------------------|
| Day 1   | Setup: Python venv, install deps, .env with OpenAI key, project folders | Running \'python -c import langchain\' works              | pip freeze matches requirements.txt          |
| Day 2   | ingest.py: Read Excel, build documents, print them                      | Console shows all 27 documents correctly formatted        | SBS/GTIN/GMDN docs all have correct metadata |
| Day 3   | ingest.py: Create ChromaDB vector store, test retrieval                 | Query \'puncture\' returns SBS puncture codes             | Top-5 results make medical sense             |
| Day 4   | prompts.py + matcher.py: Single match with LLM                          | match_single(\'Cisternal puncture\') returns correct code | JSON response parses, code = 39003-00-00     |
| Day 5   | matcher.py: Batch processing of full mapping sheet                      | Output Excel generated with all 1,906 rows                | File opens in Excel, new columns present     |

Week 2: UI, Optimize & Deliver

| **Day** | **Task**                                                    | **Deliverable**                              | **Done When**                   |
|---------|-------------------------------------------------------------|----------------------------------------------|---------------------------------|
| Day 6   | app.py: Streamlit UI - upload, process, download            | Working web UI at localhost:8501             | Non-technical person can use it |
| Day 7   | Run golden test set, measure accuracy                       | Accuracy report with pass/fail per test case | Golden set accuracy \> 85%      |
| Day 8   | Add async parallel processing (optimization \#1)            | Batch runs in \< 5 minutes                   | Timed full batch run            |
| Day 9   | Add auto-accept for high-similarity hits (optimization \#3) | \~30% of rows skip LLM, faster + cheaper     | Log shows skipped LLM calls     |
| Day 10  | Final testing, README, demo to team                         | Complete POC deliverable package             | Team signs off on demo          |

11\. Running Without OpenAI (Free Local Alternative)

If the customer does not want to send data to OpenAI, the entire pipeline can run locally with zero API costs:

| **Component**   | **OpenAI Version**     | **Local Free Version**                       | **Trade-off**                                      |
|-----------------|------------------------|----------------------------------------------|----------------------------------------------------|
| Embedding Model | text-embedding-3-small | all-MiniLM-L6-v2 (via sentence-transformers) | Slightly lower quality, 80MB download, runs on CPU |
| LLM             | GPT-4o-mini            | Ollama + Llama 3.1 8B (or Mistral 7B)        | Requires 8GB+ RAM, slower, slightly lower accuracy |
| Vector Store    | ChromaDB (same)        | ChromaDB (same)                              | No change                                          |

> \# config.py changes for local mode:
>
> \# Embedding: use HuggingFace instead of OpenAI
>
> \# In ingest.py, replace:
>
> \# from langchain_openai import OpenAIEmbeddings
>
> \# With:
>
> from langchain_community.embeddings import HuggingFaceEmbeddings
>
> embeddings = HuggingFaceEmbeddings(model_name=\"all-MiniLM-L6-v2\")
>
> \# LLM: use Ollama instead of OpenAI
>
> \# Install: curl -fsSL https://ollama.ai/install.sh \| sh
>
> \# Pull: ollama pull llama3.1:8b
>
> \# In matcher.py, replace:
>
> \# from langchain_openai import ChatOpenAI
>
> \# With:
>
> from langchain_community.llms import Ollama
>
> llm = Ollama(model=\"llama3.1:8b\", temperature=0)

12\. From POC to Production (Future Scope)

This section is NOT part of the POC. It documents what would change if the customer wants to productionize:

| **Area**            | **POC State**         | **Production State**                                 |
|---------------------|-----------------------|------------------------------------------------------|
| Reference data size | 27 sample codes       | 10,000+ codes (full SBS + GTIN + GMDN catalogs)      |
| Vector store        | ChromaDB (in-process) | PostgreSQL + pgvector (or Pinecone/Weaviate hosted)  |
| LLM                 | OpenAI API            | Self-hosted model behind VPN for HIPAA compliance    |
| UI                  | Streamlit single page | React dashboard with auth, history, manual override  |
| Batch processing    | Sequential with async | Celery task queue with Redis broker                  |
| Caching             | None                  | Redis cache for repeated descriptions                |
| Monitoring          | Console prints        | Structured logging, OpenTelemetry, LLM cost tracking |
| Security            | Local .env file       | AWS Secrets Manager, VPC, audit trail                |
| Human review        | None                  | Review queue for MEDIUM/LOW confidence matches       |
| Re-indexing         | Manual script run     | Automated pipeline when reference data updates       |

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<tbody>
<tr class="odd">
<td><p><strong>FINAL NOTE TO THE INTERN</strong></p>
<p>Start with ingest.py. Get it printing documents correctly. Then get ChromaDB storing and retrieving them. Then add the LLM. Then the batch processor. Then the UI. Do not jump ahead. Each step depends on the previous one working. Ask questions early.</p></td>
</tr>
</tbody>
</table>
