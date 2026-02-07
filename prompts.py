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
- DOMAIN: Surgical, Diagnostic/Imaging, Laboratory, Preventive, Therapeutic, Pharmaceutical, Device, Consultation
- SPECIFIC SERVICE: exact procedure/test/drug/device described
- PARAMETERS: counts (canals, surfaces, vessels, levels, units), dose/strength, size, duration, laterality
- QUALIFIERS: material, technique (open/laparoscopic/robotic/endoscopic/percutaneous), route (IV/oral/topical), complexity, timing (initial/revision/retreatment)

## STEP 2: EVALUATE EACH CANDIDATE — All 6 checks required

CHECK 1 — DOMAIN: Same broad clinical category?
  Reject if: imaging↔treatment, drug↔procedure, device↔procedure, lab↔imaging, preventive↔curative, diagnostic↔therapeutic

CHECK 2 — SERVICE TYPE: Same specific procedure/test/drug?
  Reject if: different procedure even within same domain. Watch for:
  treatment≠retreatment, partial≠total, excision≠drainage, insertion≠removal≠replacement,
  a single step≠the full procedure, a component≠the complete unit

CHECK 3 — NUMBERS: All quantitative parameters align?
  Reject if: count differs by >1 (canals, surfaces, vessels, levels, vertebrae, units, teeth, stents)
  Reject if: dose differs by >10%. Duration, size thresholds, and session counts must match.

CHECK 4 — QUALIFIERS: Material, technique, route, laterality compatible?
  Reject if: input specifies qualifier A, candidate specifies different qualifier B.
  Material A≠Material B, open≠laparoscopic≠robotic, IV≠oral, left≠right, tablet≠injection.

CHECK 5 — SCOPE: Same extent of work?
  Reject if: per-unit↔per-region, single↔multi-level, unilateral↔bilateral,
  simple↔complex, initial↔subsequent, with↔without modifier.

CHECK 6 — KEYWORD TRAP: Remove shared keywords — is there still a clinical connection?
  Reject if match depends entirely on shared words without shared clinical meaning.
  Examples: "bone graft"≠"bone scan", "3 surface"≠"3D imaging", "resin filling"≠"resin pontic",
  "cardiac catheter"≠"cardiac arrest", shared anatomy≠same service, shared number≠related procedure.

## STEP 3: DECIDE

- Candidate passes ALL 6 checks → MATCH (pick most specific if multiple pass)
- Passes checks 1-2 but minor issues on 3-5 → PARTIAL (MEDIUM/LOW confidence)
- NO candidate passes checks 1-2 → return null

## STEP 4: CONFIDENCE

HIGH — All 6 checks pass. Defensible in a payer audit.
MEDIUM — Clinically reasonable but one minor ambiguity: vague input, terminology variation, or unspecified qualifier.
LOW — Best available but has a parameter gap, scope mismatch, or needs human review.
NONE — Return null. No candidate shares both domain AND service type, or all matches are keyword-driven.

## MANDATORY REJECTIONS — return null if ANY is true

1. Best candidate is from a different clinical domain than input
2. Candidate describes a clinically different service despite shared terminology
3. A defining numerical parameter (count, dose, size) clearly mismatches
4. Input and candidate explicitly specify different qualifiers (material, technique, route)
5. Match depends entirely on shared keywords with no clinical connection
6. Drug input matched to procedure code or vice versa

## OUTPUT — Respond with ONLY this JSON, no markdown fences:

{{
  "input_analysis": {{
    "clinical_domain": "<domain>",
    "specific_service": "<procedure/test/drug/device>",
    "key_parameters": "<numbers, qualifiers extracted — or 'none specified'>"
  }},
  "matched_code": "<code or null>",
  "code_system": "<SBS|GTIN|GMDN|null>",
  "matched_description": "<candidate description or null>",
  "confidence": "<HIGH|MEDIUM|LOW|NONE>",
  "reasoning": "<1-2 sentences: which checks passed/failed, why selected or rejected>"
}}"""


def format_candidates(retrieved_docs) -> str:
    """
    Format retrieved documents into a numbered list for the LLM prompt.

    Args:
        retrieved_docs: List of LangChain Document objects from retrieval.

    Returns:
        A formatted string with numbered candidates.

    Example output:
        1. [SBS] Code: 39003-00-00 | Cisternal puncture
        2. [SBS] Code: 39006-00-00 | Ventricular puncture
        3. [GTIN] Code: 6285074000864 | BEPANTHEN - DEXPANTHENOL 5 G/ 100 G
    """
    lines = []
    for i, doc in enumerate(retrieved_docs, 1):
        meta = doc.metadata
        system = meta.get("system", "UNKNOWN")
        code = meta.get("code", "")

        # Strip the [SBS]/[GTIN]/[GMDN] prefix from page_content for cleaner display
        content = doc.page_content
        # Remove prefix pattern like "[SBS] ", "[GTIN] ", "[GMDN] "
        for prefix in ["[SBS] ", "[GTIN] ", "[GMDN] "]:
            if content.startswith(prefix):
                content = content[len(prefix):]
                break

        lines.append(f"{i}. [{system}] Code: {code} | {content}")

    return "\n".join(lines)
