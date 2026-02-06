"""
prompts.py - LLM prompt templates for code matching.

Contains the main matching prompt and helper functions for formatting candidates.
"""

MATCH_PROMPT = """You are a Saudi healthcare billing code expert specialized in NPHIES compliance.

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
{{"matched_code": "<code or null>", "code_system": "<SBS|GTIN|GMDN|null>", "matched_description": "<official description from candidate or null>", "confidence": "<HIGH|MEDIUM|LOW|NONE>", "reasoning": "<one sentence>"}}"""


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


def build_prompt(service_description: str, candidates_text: str) -> str:
    """
    Build the complete prompt for the LLM.

    Args:
        service_description: The input service description to match.
        candidates_text: Formatted string of candidate codes from format_candidates().

    Returns:
        The complete prompt string ready for LLM invocation.
    """
    return MATCH_PROMPT.format(
        service_description=service_description,
        candidates=candidates_text,
    )
