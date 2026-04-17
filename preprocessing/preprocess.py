import json
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent

RAW_FILE = ROOT_DIR / "data" / "clean" / "firstaid_docs.json"
CANONICAL_FILE = ROOT_DIR / "data" / "canonical" / "canonical_questions.json"
OUTPUT_FILE = ROOT_DIR / "data" / "clean" / "knowledge_docs.json"

MIN_TEXT_WORDS = 18
MIN_SECTION_WORDS = 16

BAD_TOPICS = [
    "pregnant", "pregnancy", "gynecologist", "thyroid",
    "weight loss", "ovulation", "period", "menstrual",
    "fertility", "contraception"
]

# only true leftover junk, not dataset wrapper phrases already stripped by extraction
NOISE_PATTERNS = [
    "<positive_smiley>",
    "this helped me personally",
    "i am not a doctor",
]

DEFINITION_PATTERNS = [
    r"^a [a-z -]+ is\b",
    r"^an [a-z -]+ is\b",
    r"^fracture is the breaking of a bone\b",
]

SYMPTOM_ONLY_PATTERNS = [
    r"^symptoms include\b",
    r"^signs include\b",
    r"^common symptoms\b",
]

GENERIC_WEAK_PATTERNS = [
    r"call (an )?ambulance\.?$",
    r"call emergency services immediately\.?$",
    r"seek medical help\.?$",
    r"consult a doctor\.?$",
]

MEDICINE_PATTERNS = [
    "which medicine to take",
    "doctor may prescribe",
    "prescribe stronger pain medication",
    "over-the-counter pain relievers",
    "acetaminophen",
    "ibuprofen",
]

AGE_PATTERNS = [
    ("self", r"(For yourself:)(.*?)(?=For adults?:|For children?:|For child(?:ren)?:|For kids?:|For infants?:|For babies?:|$)"),
    ("adult", r"(For adults?:)(.*?)(?=For children?:|For child(?:ren)?:|For kids?:|For infants?:|For babies?:|$)"),
    ("child", r"(For children?:|For child(?:ren)?:|For kids?:|For children over 1:|For kids over 1-year old:)(.*?)(?=For infants?:|For babies?:|$)"),
    ("infant", r"(For infants?:|For babies?:)(.*?)(?=$)"),
]

DIRECT_MAP = {
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
    "bleeding": "bleeding",
}

INTENT_PATTERNS = {
    "drowning_near_drowning": [
        "drowning", "drowned", "near drowning", "pulled from water",
        "water rescue", "rescued from water"
    ],
    "cpr_resuscitation": [
        "cpr", "compressions", "rescue breaths", "30 compressions",
        "2 rescue breaths", "not breathing", "unresponsive", "no pulse",
        "100-120 per minute", "chest compressions"
    ],
    "choking": [
        "choking", "foreign body", "back blows", "abdominal thrusts",
        "cannot breathe", "shoulder blades", "obstruction persists",
        "expel the foreign body"
    ],
    "bleeding": [
        "bleeding", "blood loss", "direct pressure", "heavy bleeding",
        "spurting blood", "apply pressure", "nose bleeding", "nosebleed"
    ],
    "burns": [
        "burn", "scald", "hot water", "hot oil", "steam burn",
        "cool the burn", "running water", "blister"
    ],
    "fracture": [
        "fracture", "broken bone", "splint", "bone", "broken toe",
        "dislocated bone", "dislocation", "joint immobilization"
    ],
    "seizure": [
        "seizure", "convulsion", "fit", "fits", "tonic-clonic"
    ],
    "poisoning": [
        "poison", "poisoning", "toxic", "swallowed poison",
        "chemical swallowed"
    ],
    "allergic_reaction": [
        "allergic", "allergy", "anaphylaxis", "swelling",
        "allergic reaction"
    ],
    "breathing_difficulty": [
        "difficulty breathing", "shortness of breath", "breathless",
        "hard to breathe", "trouble breathing"
    ],
    "asthma_attack": [
        "asthma", "wheezing", "inhaler", "asthma attack"
    ],
    "electric_shock": [
        "electric shock", "electrocution", "current", "live wire",
        "lightning", "electrical burn"
    ],
    "snake_bite": [
        "snake bite", "bitten by a snake", "venomous", "fang marks"
    ],
    "animal_bite": [
        "animal bite", "dog bite", "cat bite", "rabies"
    ],
    "chemical_exposure": [
        "chemical", "bleach", "acid", "fumes", "chemical burn",
        "chemical in eye", "chemical on skin", "flush with water"
    ],
    "eye_injury": [
        "eye injury", "something in the eye", "foreign object in eye",
        "vision", "do not rub the eye"
    ],
    "head_injury": [
        "head injury", "head trauma", "concussion", "hit on the head"
    ],
    "chest_pain": [
        "chest pain", "heart attack", "tightness in chest",
        "pain to arm", "pain to jaw", "sweating"
    ],
    "shock": [
        "shock", "clammy", "weak pulse", "circulatory shock"
    ],
    "dehydration": [
        "dehydration", "dry mouth", "sunken eyes", "water loss",
        "oral rehydration", "rehydration"
    ],
    "nosebleed": [
        "nosebleed", "blood from nose", "nose is bleeding"
    ],
    "sprain_strain": [
        "sprain", "strain", "twisted ankle", "pulled muscle",
        "elastic bandage", "ice covered in a towel", "bandage the joint",
        "joint immobilization", "rice", "rest"
    ],
    "cuts_minor_wounds": [
        "small cut", "minor wound", "scrape", "abrasion"
    ],
    "severe_wound": [
        "deep wound", "deep cut", "gaping wound", "severe wound"
    ],
    "spinal_injury": [
        "spinal injury", "spine injury", "neck injury", "spinal cord injury",
        "do not move the person", "stabilize their head and neck"
    ],
    "amputation_avulsion": [
        "cut off", "severed", "amputation", "body part detached",
        "amputated limb", "cut off limb"
    ],
    "foreign_body_nose_ear": [
        "object stuck in nose", "object in ear", "foreign body in nose",
        "foreign body in ear"
    ],
    "dental_injury": [
        "tooth", "knocked out tooth", "broken tooth"
    ],
    "insect_sting": [
        "bee sting", "wasp sting", "insect sting", "stinger"
    ],
    "temperature_emergency": [
        "heatstroke", "heat exhaustion", "hypothermia", "frostbite",
        "extreme heat", "extreme cold"
    ],
    "fainting": [
        "fainted", "fainting", "passed out", "blackout", "lost consciousness"
    ],
    "vertigo": [
        "vertigo", "dizziness", "room spinning"
    ],
    "headache": [
        "headache", "migraine", "head pain"
    ],
    "stroke_signs": [
        "stroke", "face drooping", "slurred speech", "one side weakness"
    ],
    "recovery_position": [
        "recovery position", "put on their side", "side position"
    ],
    "basic_first_aid_principles": [
        "basic first aid", "principles of first aid", "first aid rules"
    ],
    "first_aid_kit_contents": [
        "first aid kit", "kit contents", "what should be in a first aid kit"
    ],
    "emergency_response_guidelines": [
        "emergency response", "medical emergency steps", "what to do in an emergency"
    ],
}


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


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def contains_any(text: str, terms) -> bool:
    t = (text or "").lower()
    return any(term in t for term in terms)


def extract_instructions(raw_text: str) -> str:
    text = clean_text(raw_text)

    m = re.search(r"Instructions:\s*(.*)$", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        text = m.group(1).strip()

    text = re.sub(
        r"^First aid instructions for [\w_ -]+\.\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    text = re.sub(
        r"Question variations:\s*.*?(?=Instructions:|$)",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()

    text = text.replace("handand", "hand and")
    text = text.replace("childen", "children")
    text = text.replace("joinnt", "joint")
    text = text.replace("it loses consciousness", "the infant loses consciousness")

    return clean_text(text)


def is_bad_topic(text: str) -> bool:
    return contains_any(text, BAD_TOPICS)


def is_noisy(text: str) -> bool:
    return contains_any(text, NOISE_PATTERNS)


def is_definition_only(text: str) -> bool:
    t = clean_text(text).lower()
    return any(re.match(p, t) for p in DEFINITION_PATTERNS)


def is_symptom_only(text: str) -> bool:
    t = clean_text(text).lower()
    return any(re.match(p, t) for p in SYMPTOM_ONLY_PATTERNS)


def is_medicine_focused(text: str) -> bool:
    return contains_any(text, MEDICINE_PATTERNS)


def is_generic_weak_instruction(text: str) -> bool:
    t = clean_text(text).lower()
    if word_count(t) < MIN_TEXT_WORDS:
        return True
    for p in GENERIC_WEAK_PATTERNS:
        if re.fullmatch(p, t):
            return True
    return False


def split_age_sections(text: str) -> Dict[str, str]:
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


def score_intents_from_text(text: str) -> Dict[str, int]:
    text_l = (text or "").lower()
    scores = {}

    for intent, patterns in INTENT_PATTERNS.items():
        score = 0
        for p in patterns:
            if p in text_l:
                score += 1
        if score > 0:
            scores[intent] = score

    # strong action boosts
    if "ice covered in a towel" in text_l or "elastic bandage" in text_l or "joint immobilization" in text_l:
        scores["sprain_strain"] = scores.get("sprain_strain", 0) + 2
    if "30 compressions" in text_l or "2 rescue breaths" in text_l:
        scores["cpr_resuscitation"] = scores.get("cpr_resuscitation", 0) + 3
    if "back blows" in text_l or "abdominal thrusts" in text_l:
        scores["choking"] = scores.get("choking", 0) + 3
    if "flush with water" in text_l and ("chemical" in text_l or "eye" in text_l):
        scores["chemical_exposure"] = scores.get("chemical_exposure", 0) + 2
    if "immobilize" in text_l and ("splint" in text_l or "broken" in text_l or "fracture" in text_l):
        scores["fracture"] = scores.get("fracture", 0) + 2
    if "transport to hospital" in text_l and ("amputated" in text_l or "cut off limb" in text_l):
        scores["amputation_avulsion"] = scores.get("amputation_avulsion", 0) + 2
    if "stabilize their head and neck" in text_l or "do not move the person" in text_l:
        scores["spinal_injury"] = scores.get("spinal_injury", 0) + 2

    return scores


def choose_best_intent(category: str, text: str) -> Optional[str]:
    category = (category or "").strip().lower()
    scores = score_intents_from_text(text)

    if category in DIRECT_MAP and category != "general_first_aid":
        mapped = DIRECT_MAP[category]
        scores[mapped] = scores.get(mapped, 0) + 3

    if not scores:
        return None

    best_intent = max(scores, key=scores.get)
    best_score = scores[best_intent]

    # strict fallback handling
    if best_intent == "emergency_response_guidelines" and best_score < 2:
        return None

    if best_score < 2 and category == "general_first_aid":
        return None

    return best_intent


def infer_keywords(intent: str, text: str) -> List[str]:
    kws = set(INTENT_PATTERNS.get(intent, []))
    scores = score_intents_from_text(text)

    for matched_intent in scores:
        kws.add(matched_intent)

    text_l = text.lower()
    for term in [
        "cpr", "compressions", "rescue breaths", "bleeding", "burn", "fracture",
        "poison", "eye", "chemical", "shock", "bite", "sting", "seizure",
        "breathing", "choking", "airway", "unconscious", "drowning",
        "ice", "bandage", "swelling", "rest", "splint", "water",
        "immobilize", "hospital"
    ]:
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

    dropped = {
        "bad_topic": 0,
        "weak_or_short": 0,
        "empty": 0,
        "noise": 0,
        "definition_only": 0,
        "symptom_only": 0,
        "medicine_focused": 0,
        "no_intent": 0,
        "weak_generic": 0,
    }

    for row in tqdm(raw_docs, desc="Preprocessing first-aid docs"):
        raw_text = row.get("text", "")
        source = row.get("source", "unknown")
        category = row.get("category", "")

        instructions = extract_instructions(raw_text)

        if not instructions:
            dropped["empty"] += 1
            continue

        if is_noisy(instructions):
            dropped["noise"] += 1
            continue

        if is_bad_topic(raw_text) or is_bad_topic(instructions):
            dropped["bad_topic"] += 1
            continue

        if is_generic_weak_instruction(instructions):
            dropped["weak_or_short"] += 1
            continue

        # drop weak educational fragments unless category is strongly specific
        if is_definition_only(instructions) and category == "general_first_aid":
            dropped["definition_only"] += 1
            continue

        if is_symptom_only(instructions) and category == "general_first_aid":
            dropped["symptom_only"] += 1
            continue

        if is_medicine_focused(instructions):
            dropped["medicine_focused"] += 1
            continue

        chosen_intent = choose_best_intent(category, instructions)
        if not chosen_intent:
            dropped["no_intent"] += 1
            continue

        intent_scores = score_intents_from_text(instructions)
        top_score = max(intent_scores.values()) if intent_scores else 0

        if chosen_intent == "emergency_response_guidelines" and top_score < 2:
            dropped["weak_generic"] += 1
            continue

        severity = severity_map.get(chosen_intent, "routine")
        age_sections = split_age_sections(instructions)

        for age_group, section_text in age_sections.items():
            section_text = clean_text(section_text)

            if not section_text or word_count(section_text) < MIN_SECTION_WORDS:
                continue

            if is_bad_topic(section_text):
                continue

            section_scores = score_intents_from_text(section_text)
            section_top = max(section_scores.values()) if section_scores else 0

            if chosen_intent == "emergency_response_guidelines" and section_top < 2:
                continue

            fp = stable_hash(chosen_intent, age_group, section_text)
            if fp in seen:
                continue
            seen.add(fp)

            cleaned_docs.append({
                "id": fp,
                "intent": chosen_intent,
                "severity": severity,
                "age_group": age_group,
                "language": detect_language(section_text),
                "source": source,
                "keywords": infer_keywords(chosen_intent, section_text),
                "text": section_text,
            })

    save_json(OUTPUT_FILE, cleaned_docs)
    print(f"✅ Wrote {len(cleaned_docs)} cleaned knowledge docs to {OUTPUT_FILE}")
    print(f"Dropped stats: {dropped}")


if __name__ == "__main__":
    main()