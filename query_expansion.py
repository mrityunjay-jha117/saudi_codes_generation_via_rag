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
    "r.c.t":             "root canal treatment endodontic",
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
