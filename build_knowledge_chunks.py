import json
import re
import uuid
import hashlib
from pathlib import Path

# This file lives in backend/, so parent is correct.
# If you place it inside backend/preprocessing/, change to .parent.parent
ROOT_DIR = Path(__file__).resolve().parent

INPUT_FILE = ROOT_DIR / "data" / "clean" / "knowledge_docs.json"
CANONICAL_FILE = ROOT_DIR / "data" / "canonical" / "canonical_questions.json"
OUTPUT_FILE = ROOT_DIR / "data" / "knowledge" / "knowledge_chunks.json"

MIN_WORDS = 25
MAX_WORDS = 120

AGE_MARKERS = {
    "adult": [r"for adults?:", r"for yourself:"],
    "child": [r"for child(?:ren)?:", r"for kids?:", r"for children over 1:", r"for kids over 1-year old:"],
    "infant": [r"for infants?:", r"for babies?:", r"for infant:"]
}


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.lower().encode("utf-8")).hexdigest()


def sentence_split(text: str):
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [clean_text(p) for p in parts if clean_text(p)]


def pack_sentences(sentences):
    chunks = []
    current = []

    for sent in sentences:
        trial = " ".join(current + [sent]).strip()
        if len(trial.split()) <= MAX_WORDS:
            current.append(sent)
        else:
            if current:
                chunk = " ".join(current).strip()
                if len(chunk.split()) >= MIN_WORDS:
                    chunks.append(chunk)
            current = [sent]

    if current:
        chunk = " ".join(current).strip()
        if len(chunk.split()) >= MIN_WORDS:
            chunks.append(chunk)

    return chunks


def split_by_age(text: str):
    text_lower = text.lower()
    positions = []

    for age, patterns in AGE_MARKERS.items():
        for p in patterns:
            m = re.search(p, text_lower)
            if m:
                positions.append((age, m.start(), m.end()))

    if not positions:
        return {}

    positions.sort(key=lambda x: x[1])
    sections = {}

    for i, (age, start, end) in enumerate(positions):
        section_start = end
        section_end = positions[i + 1][1] if i + 1 < len(positions) else len(text)
        section_text = clean_text(text[section_start:section_end])
        if section_text:
            sections[age] = section_text

    return sections


def detect_language(text: str) -> str:
    if re.search(r"[\u0600-\u06FF]", text or ""):
        return "ur"
    return "en"


def infer_keywords(intent: str, text: str):
    base = {
        "headache": ["headache", "head pain", "migraine"],
        "chest_pain": ["chest pain", "heart attack", "sweating"],
        "burns": ["burn", "scald", "blister"],
        "choking": ["choking", "airway", "cannot breathe"],
        "chemical_exposure": ["chemical", "flush", "exposure"],
        "eye_injury": ["eye", "injury", "vision"],
        "bleeding": ["bleeding", "blood loss", "pressure"],
        "fracture": ["fracture", "broken bone", "splint"],
        "seizure": ["seizure", "convulsion"],
        "fainting": ["fainting", "passed out", "unconscious"],
        "poisoning": ["poison", "toxic", "swallowed"],
        "drowning_near_drowning": ["drowning", "water", "breathing", "cpr"],
        "cpr_resuscitation": ["cpr", "compressions", "rescue breaths", "unresponsive"],
        "breathing_difficulty": ["breathing", "shortness of breath", "airway"],
        "allergic_reaction": ["allergy", "swelling", "anaphylaxis"],
        "electric_shock": ["electric shock", "current", "electrocution"],
    }

    kws = set(base.get(intent, []))
    text_l = text.lower()
    for term in [
        "bleeding", "burn", "fracture", "headache", "chest pain", "poison",
        "eye", "chemical", "shock", "bite", "sting", "seizure", "cpr",
        "breathing", "choking", "drowning", "unconscious", "compressions"
    ]:
        if term in text_l:
            kws.add(term)

    return sorted(kws)


def is_contextless_fragment(text: str) -> bool:
    t = text.lower()
    return (
        t.startswith("after following the above steps")
        or t.startswith("the above steps")
        or t.startswith("also seek medical help if")
        or t.startswith("question variations")
    )


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing cleaned docs file: {INPUT_FILE}")

    severity_map = {}
    if CANONICAL_FILE.exists():
        with open(CANONICAL_FILE, "r", encoding="utf-8") as f:
            canonical = json.load(f)
        for intent, payload in canonical.items():
            severity_map[intent] = payload.get("severity", "routine")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        docs = json.load(f)

    final_chunks = []
    seen = set()

    for doc in docs:
        if isinstance(doc, str):
            raw_text = clean_text(doc)
            intent = ""
            source = "unknown"
            language = detect_language(raw_text)
            severity = "routine"
            doc_age_group = "general"
        else:
            raw_text = clean_text(doc.get("text", ""))
            intent = clean_text(doc.get("intent", ""))
            source = doc.get("source", "unknown")
            language = doc.get("language", detect_language(raw_text))
            severity = doc.get("severity", severity_map.get(intent, "routine"))
            doc_age_group = doc.get("age_group", "general")

        if not raw_text or not intent:
            continue

        # If preprocess already split by age, trust that first.
        # Only fallback to in-text splitting if doc is still general and contains age markers.
        if doc_age_group != "general":
            age_sections = {doc_age_group: raw_text}
        else:
            age_sections = split_by_age(raw_text) or {"general": raw_text}

        for age_group, section in age_sections.items():
            sentences = sentence_split(section)
            packed_chunks = pack_sentences(sentences)

            for chunk_text in packed_chunks:
                if is_contextless_fragment(chunk_text):
                    continue

                h = stable_hash(f"{intent}|{age_group}|{chunk_text}")
                if h in seen:
                    continue
                seen.add(h)

                final_chunks.append({
                    "id": str(uuid.uuid4()),
                    "intent": intent,
                    "severity": severity,
                    "age_group": age_group,
                    "language": language if language else detect_language(chunk_text),
                    "type": "procedure",
                    "keywords": infer_keywords(intent, chunk_text),
                    "text": chunk_text,
                    "source": source
                })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_chunks, f, indent=2, ensure_ascii=False)

    print(f"✅ Rebuilt {len(final_chunks)} clean knowledge chunks at {OUTPUT_FILE}")


if __name__ == "__main__":
    main()