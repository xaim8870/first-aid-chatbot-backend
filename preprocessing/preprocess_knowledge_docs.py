import json
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent

RAW_FILE = ROOT_DIR / "data" / "clean" / "firstaid_docs.json"
CANONICAL_FILE = ROOT_DIR / "data" / "canonical" / "canonical_questions.json"
OUTPUT_FILE = ROOT_DIR / "data" / "clean" / "knowledge_docs.json"

MIN_TEXT_WORDS = 12


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    return text.strip()


def detect_language(text: str) -> str:
    return "ur" if re.search(r"[\u0600-\u06FF]", text or "") else "en"


def stable_hash(*parts: str) -> str:
    joined = "||".join((p or "").strip().lower() for p in parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def extract_instructions(raw_text: str) -> str:
    """
    Keep only the real medical instruction section.
    Remove title and question variation noise.
    """
    text = clean_text(raw_text)

    # Prefer content after "Instructions:"
    m = re.search(r"Instructions:\s*(.*)$", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        text = m.group(1).strip()

    # Remove leading title if still present
    text = re.sub(
        r"^First aid instructions for [\w_ -]+\.\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    # Remove any leftover "Question variations" block if present
    text = re.sub(
        r"Question variations:\s*.*?(?=Instructions:|$)",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()

    # Small typo cleanup from source patterns
    text = text.replace("handand", "hand and")
    text = text.replace("childen", "children")
    text = text.replace("it loses consciousness", "the infant loses consciousness")

    return clean_text(text)


def normalize_intent(category: str, text: str) -> str:
    """
    Map dataset category to your final chatbot intent names.
    """
    category = (category or "").strip().lower()
    text_l = (text or "").lower()

    direct_map = {
        "choking": "choking",
        "burns": "burns",
        "fracture": "fracture",
        "shock": "shock",
        "snake_bite": "snake_bite",
        "head_injury": "head_injury",
        "chest_pain": "chest_pain",
        "breathing_difficulty": "breathing_difficulty",
        "severe_wound": "severe_wound",
        "eye_injury": "eye_injury",
        "diabetic_emergency": "diabetic_emergency",
        "asthma_attack": "asthma_attack",
        "sprain_strain": "sprain_strain",
        "nosebleed": "nosebleed",
        "cuts_minor_wounds": "cuts_minor_wounds",
        "electric_shock": "electric_shock",
        "drowning_near_drowning": "drowning_near_drowning",
        "animal_bite": "animal_bite",
        "insect_sting": "insect_sting",
        "chemical_exposure": "chemical_exposure",
        "headache": "headache",
        "dental_injury": "dental_injury",
        "spinal_injury": "spinal_injury",
        "amputation_avulsion": "amputation_avulsion",
        "splinter_embedded_object": "splinter_embedded_object",
        "dehydration": "dehydration",
        "foreign_body_nose_ear": "foreign_body_nose_ear",
        "poisoning": "poisoning",
        "allergic_reaction": "allergic_reaction",
        "vertigo": "vertigo",
        "fainting": "fainting",
        "seizure": "seizure",
        "temperature_emergency": "temperature_emergency",
        "first_aid_kit_contents": "first_aid_kit_contents",
        "basic_first_aid_principles": "basic_first_aid_principles",
        "emergency_response_guidelines": "emergency_response_guidelines",
        "cpr_resuscitation": "cpr_resuscitation",
        "recovery_position": "recovery_position",
        "stroke_signs": "stroke_signs",
    }

    if category in direct_map:
        return direct_map[category]

    # Fix broad dataset category using content
    if category == "general_first_aid":
        if any(k in text_l for k in ["drowning", "drowned", "water", "rescue breaths"]):
            return "drowning_near_drowning"
        if any(k in text_l for k in ["cpr", "compressions", "not breathing", "unresponsive"]):
            return "cpr_resuscitation"
        return "emergency_response_guidelines"

    return category


AGE_PATTERNS = [
    ("self", r"(For yourself:)(.*?)(?=For adults?:|For children?:|For child(?:ren)?:|For kids?:|For infants?:|For babies?:|$)"),
    ("adult", r"(For adults?:)(.*?)(?=For children?:|For child(?:ren)?:|For kids?:|For infants?:|For babies?:|$)"),
    ("child", r"(For children?:|For child(?:ren)?:|For kids?:|For kids over 1-year old:)(.*?)(?=For infants?:|For babies?:|$)"),
    ("infant", r"(For infants?:|For babies?:)(.*?)(?=$)"),
]


def split_age_sections(text: str) -> Dict[str, str]:
    """
    Split adult/child/infant/self sections when present.
    If absent, return general.
    """
    sections: Dict[str, str] = {}

    for age_group, pattern in AGE_PATTERNS:
        m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            section_text = clean_text(m.group(2))
            if section_text:
                sections[age_group] = section_text

    if sections:
        return sections

    return {"general": clean_text(text)}


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def infer_keywords(intent: str, text: str) -> List[str]:
    base = {
        "choking": ["choking", "airway", "cough", "back blows", "abdominal thrusts"],
        "drowning_near_drowning": ["drowning", "breathing", "rescue breaths", "cpr"],
        "cpr_resuscitation": ["cpr", "compressions", "rescue breaths", "unresponsive"],
        "burns": ["burn", "cool water", "scald"],
        "fracture": ["fracture", "broken bone", "splint"],
        "bleeding": ["bleeding", "pressure", "blood loss"],
        "chest_pain": ["chest pain", "heart attack", "sweating"],
        "seizure": ["seizure", "convulsions"],
        "allergic_reaction": ["allergy", "swelling", "anaphylaxis"],
        "poisoning": ["poison", "swallowed", "toxic"],
    }

    kws = set(base.get(intent, []))
    text_l = text.lower()

    dynamic_terms = [
        "cpr", "compressions", "rescue breaths", "bleeding", "burn", "fracture",
        "poison", "eye", "chemical", "shock", "bite", "sting", "seizure",
        "breathing", "choking", "airway", "unconscious", "drowning"
    ]

    for term in dynamic_terms:
        if term in text_l:
            kws.add(term)

    return sorted(kws)


def main():
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Missing raw dataset file: {RAW_FILE}")

    if not CANONICAL_FILE.exists():
        raise FileNotFoundError(f"Missing canonical questions file: {CANONICAL_FILE}")

    raw_docs = load_json(RAW_FILE)
    canonical = load_json(CANONICAL_FILE)

    severity_map = {
        intent: payload.get("severity", "routine")
        for intent, payload in canonical.items()
    }

    cleaned_docs: List[Dict] = []
    seen = set()

    for row in raw_docs:
        raw_text = row.get("text", "")
        source = row.get("source", "unknown")
        category = row.get("category", "")

        instructions = extract_instructions(raw_text)
        if not instructions or word_count(instructions) < MIN_TEXT_WORDS:
            continue

        intent = normalize_intent(category, instructions)
        if not intent:
            continue

        severity = severity_map.get(intent, "routine")
        age_sections = split_age_sections(instructions)

        for age_group, section_text in age_sections.items():
            section_text = clean_text(section_text)
            if not section_text or word_count(section_text) < MIN_TEXT_WORDS:
                continue

            fp = stable_hash(intent, age_group, section_text)
            if fp in seen:
                continue
            seen.add(fp)

            cleaned_docs.append({
                "id": fp,
                "intent": intent,
                "severity": severity,
                "age_group": age_group,
                "language": detect_language(section_text),
                "source": source,
                "keywords": infer_keywords(intent, section_text),
                "text": section_text,
            })

    save_json(OUTPUT_FILE, cleaned_docs)
    print(f"✅ Wrote {len(cleaned_docs)} cleaned knowledge docs to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()