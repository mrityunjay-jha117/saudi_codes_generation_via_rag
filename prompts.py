"""
prompts.py - LLM prompt templates for code matching.

Contains the main matching prompt and helper functions for formatting candidates.
"""

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
