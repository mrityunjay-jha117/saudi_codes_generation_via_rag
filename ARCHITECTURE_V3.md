# Saudi Billing Code Matcher - Architecture V3 (Pinecone Native)

## System Overview

The Saudi Billing Code Matcher is a specialized RAG (Retrieval-Augmented Generation) system designed to map free-text clinical service descriptions to standardized Saudi billing codes (SBS, GMDN, GTIN) with high precision.

Unlike V1 (ChromaDB + Local Embeddings) or V2 (Hybrid), **V3 operates on a Cloud-Native Architecture** leveraging Pinecone for both vector storage and embedding inference, and AWS Bedrock for intelligent reasoning.

---

## 🏗 Technology Stack

- **Orchestration**: Python 3.10+
- **Vector Database**: **Pinecone Serverless** (Index: `saudi-billing-codes`)
- **Embeddings**: **Pinecone Inference API** (Model: `multilingual-e5-large`)
- **LLM (Reasoning)**: **Claude 3.5 Sonnet** (AWS Bedrock)
- **LLM (Normalization)**: **Claude 3 Haiku** (AWS Bedrock) - _Speed Optimized_
- **Frameworks**: LangChain (minimal), Boto3, Streamlit (UI)

---

## 🧩 Core Architecture Components

### 1. The "Smart Router" Pipeline

The system moves away from a single "retrieve-then-generate" loop to a multi-stage routing pipeline.

#### **Stage 1: Normalization & Domain Prediction**

**Input:** Raw user text (e.g., _"Titanium Screw 3.5mm"_).
**Engine:** `Claude 3 Haiku` (configured via `NORMALIZATION_LLM_MODEL`).
**Action:**

1.  **Rewrite:** Standardize input into formal medical terminology (e.g., _"Orthopedic bone screw, titanium alloy, 3.5mm diameter"_).
2.  **Predict:** Determine the most likely coding system:
    - `sbs_v2`: Procedures/Surgeries
    - `gmdn_v2`: Medical Devices
    - `gtin_v2`: Commercial Products (Drugs/Consumables)

#### **Stage 2: Cloud-Native Embedding**

**Input:** Normalized Query.
**Engine:** `Pinecone Inference API`.
**Action:**

- The text is sent to Pinecone's `multilingual-e5-large` model hosted on their infrastructure.
- Returns a 1024-dimension vector.
- _Advantage:_ No local model weights, consistent Tokenizer with the database index.

#### **Stage 3: Namespace-Aware Retrieval ("Soft Routing")**

**Input:** Vector + Predicted Domain.
**Engine:** `Pinecone Index`.
**Strategy:**

- **Primary Search:** Query the _Predicted Namespace_ (e.g., `gmdn_v2`) for **Top-15** candidates.
- **Secondary Search:** Query _All Other Namespaces_ (`sbs_v2`, etc.) for **Top-3** candidates each.
- **Merge:** Deduplicate results.
  _Benefit:_ Prioritizes the likely domain but prevents "Hard Partition Failures" where a misclassification results in zero hits.

#### **Stage 4: Metadata-Aware LLM Selection**

**Input:** Original Input + Enriched Candidates (with full metadata).
**Engine:** `Claude 3.5 Sonnet`.
**Action:**

- The LLM evaluates candidates against the input using strict clinical rules.
- Checks: Side-laterality, Dosage, Material, Excludes/Includes fields.
- returns a JSON object with `matched_code`, `confidence`, and `reasoning`.

---

## 📂 Project Structure

| File         | Description                                                                          |
| :----------- | :----------------------------------------------------------------------------------- |
| `matcher.py` | Core engine. Handles Pinecone connection, Normalization loop, and LLM orchestration. |
| `config.py`  | Central configuration. Environment variables, Model IDs, Thresholds.                 |
| `prompts.py` | LLM Prompts (`MATCH_PROMPT` for selection, `NORMALIZATION_PROMPT` for routing).      |
| `app.py`     | Streamlit User Interface for batch processing and testing.                           |
| `.env`       | Secrets (`PINECONE_API_KEY`, `AWS_SECRET_ACCESS_KEY`, etc.).                         |

---

## ⚙️ Key Configuration (config.py)

| Setting                   | Value                  | Purpose                                    |
| :------------------------ | :--------------------- | :----------------------------------------- |
| `PINECONE_INDEX_NAME`     | `saudi-billing-codes`  | Target Vector DB Index                     |
| `AWS_REGION`              | `us-east-1`            | Bedrock Region                             |
| `LLM_MODEL`               | `claude-3-5-sonnet...` | Main Reasoner                              |
| `NORMALIZATION_LLM_MODEL` | `claude-3-haiku...`    | Fast Router/Normalizer                     |
| `TOP_K_GENERAL`           | `15`                   | Candidates to fetch from Primary Namespace |
| `MAX_CANDIDATES_TO_LLM`   | `10`                   | Final context window limit                 |

---

## 🚀 Usage Flow

1.  **Batch Processing**:
    - User uploads an Excel file via `app.py`.
    - `matcher.match_batch_async` processes rows in parallel (Semaphore limited to 1 for rate safety).
    - Results exported to Excel with highlighting.

2.  **Single Match**:
    - `matcher.match_single("Query")` executes the full pipeline for one item.

---

## ⚠️ Critical Dependencies

- **Pinecone Index**: Must be pre-created and populated with vectors in namespaces (`sbs_v2`, `gmdn_v2`, `gtin_v2`) using `multilingual-e5-large` compatible dimensions (1024).
- **Credentials**: Valid AWS Bedrock access and Pinecone API Key required.
