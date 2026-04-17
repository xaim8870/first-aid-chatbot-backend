import json
import re
import uuid
import hashlib
from pathlib import Path
from collections import Counter, defaultdict

ROOT_DIR = Path(__file__).resolve().parent
INPUT_FILE = ROOT_DIR / "data" / "knowledge" / "knowledge_chunks.json"
OUTPUT_FILE = ROOT_DIR / "data" / "knowledge" / "knowledge_chunks_final.json"
CANONICAL_FILE = ROOT_DIR / "data" / "canonical" / "canonical_questions.json"

MIN_WORDS = 18
MAX_WORDS = 110

DROP_INTRO_PATTERNS = [
    r"^First aid instructions for [^.]+\.?\s*",
    r"Question variations:\s*.*?Instructions:\s*",
    r"^Instructions:\s*",
]

LEFTOVER_NOISE_PATTERNS = [
    r"question variations",
    r"how do you treat",
    r"what to do if i get",
    r"how to cure",
    r"<positive_smiley>",
    r"\bcrp\b",   # typo noise
]

BAD_TOPICS = [
    "pregnant", "pregnancy", "ovulation", "period", "gynecologist",
    "thyroid", "weight loss", "lose weight", "chest congestion",
    "tubes tied", "vaginal discharge", "contraceptives",
    "alarm go off", "dark chocolate", "track without",
]

BAD_ANECDOTAL = [
    "i am not a doctor",
    "this helped me personally",
    "my personal advice",
    "it is what i do personally",
    "i know from experience",
    "have a good day",
    "good luck",
    "you are your own drug",
]

DEFINITION_ONLY_PATTERNS = [
    r"^a fracture is the breaking of a bone\.?$",
    r"^a sprain is a partial or complete rupture.*$",
    r"^heatstroke is the rise in body temperature above.*$",
]

ACTION_WORDS = [
    "call", "seek", "start", "check", "place", "remove", "wash", "flush",
    "cool", "cover", "apply", "keep", "move", "rest", "loosen", "press",
    "encourage", "give", "do not", "don't", "tilt", "turn", "immobilize",
]

# Strong positive evidence for each intent
INTENT_KEYWORDS = {
    "bleeding": ["bleeding", "blood loss", "apply pressure", "direct pressure", "blood"],
    "allergic_reaction": ["allergic reaction", "anaphylaxis", "swelling", "epi pen", "epinephrine", "hives"],
    "poisoning": ["poison", "poisoned", "toxic", "swallowed", "ingested", "poison center", "chemical"],
    "vertigo": ["vertigo", "dizziness", "spinning", "lightheaded"],
    "fainting": ["faint", "fainted", "passed out", "syncope", "lost consciousness"],
    "seizure": ["seizure", "convulsion", "tonic-clonic", "fit", "febrile convulsion"],
    "burns": ["burn", "scald", "blister", "cool running water", "thermal burn", "burned skin"],
    "fracture": ["fracture", "broken bone", "splint", "immobilize", "deformity", "bone"],
    "shock": ["shock", "clammy", "pale", "lie down", "raise legs"],
    "choking": ["choking", "back blows", "abdominal thrusts", "foreign body", "cannot breathe", "can't breathe", "airway obstruction"],
    "temperature_emergency": ["heatstroke", "heat exhaustion", "hypothermia", "cold exposure", "overheating", "heat cramps"],
    "snake_bite": ["snake bite", "snakebite", "venom", "fang", "bitten by a snake"],
    "head_injury": ["head injury", "head trauma", "hit the head", "concussion"],
    "chest_pain": ["chest pain", "heart attack", "tightness", "pain in the chest", "pain spreads", "sweating", "radiates"],
    "breathing_difficulty": ["difficulty breathing", "shortness of breath", "trouble breathing", "breathless", "wheezing"],
    "severe_wound": ["deep wound", "severe wound", "gaping wound", "large cut", "open wound"],
    "eye_injury": ["eye injury", "injured eye", "foreign object in the eye", "vision", "eye"],
    "first_aid_kit_contents": ["first aid kit", "gauze", "bandage", "gloves", "kit contents"],
    "basic_first_aid_principles": ["first aid", "stay calm", "check response", "call for help", "danger response"],
    "emergency_response_guidelines": ["ambulance", "emergency services", "call 112", "call 911", "call emergency"],
    "cpr_resuscitation": ["cpr", "compressions", "rescue breaths", "unresponsive", "not breathing", "cardiopulmonary"],
    "recovery_position": ["recovery position", "on their side", "keep airway open"],
    "stroke_signs": ["stroke", "face drooping", "slurred speech", "one side weakness", "FAST"],
    "diabetic_emergency": ["low blood sugar", "high blood sugar", "diabetic", "glucose", "hypoglycemia", "hyperglycemia"],
    "asthma_attack": ["asthma", "inhaler", "wheeze", "asthma attack"],
    "sprain_strain": ["sprain", "strain", "twisted ankle", "pulled muscle", "RICE", "compression"],
    "nosebleed": ["nosebleed", "nose bleed", "pinch the nose"],
    "cuts_minor_wounds": ["minor cut", "scrape", "small wound", "clean the cut"],
    "electric_shock": ["electric shock", "electrocuted", "power source", "non-conductive", "electric current", "lightning"],
    "drowning_near_drowning": ["drowning", "near-drowning", "water rescue", "rescue breaths"],
    "animal_bite": ["animal bite", "dog bite", "cat bite", "rabies"],
    "insect_sting": ["bee sting", "wasp sting", "insect sting", "stinger"],
    "chemical_exposure": ["chemical exposure", "chemical burn", "flush the eye", "chemical in the eye", "caustic"],
    "headache": ["headache", "head pain", "migraine", "pain in the head"],
}

# Strong negative evidence: if these dominate, drop or reassign
NEGATIVE_KEYWORDS = {
    "headache": ["heatstroke", "heat exhaustion", "convulsions", "loss of consciousness"],
    "choking": ["seizure", "convulsion", "febrile convulsion", "fever", "pregnant", "chest congestion"],
    "burns": ["pregnant", "gynecologist", "tubes tied", "period", "ultrasound"],
    "bleeding": ["pregnant", "contraceptives", "period", "ovulation"],
    "cpr_resuscitation": ["heatstroke", "cool place", "ice packs", "fan"],  # mixed heat emergency text
}

# Some overlap is acceptable
RELATED_INTENTS = {
    "electric_shock": {"burns", "cpr_resuscitation"},
    "burns": {"electric_shock"},
    "cpr_resuscitation": {"drowning_near_drowning", "electric_shock", "choking"},
    "temperature_emergency": {"headache"},
    "headache": {"temperature_emergency"},
}

def clean_text(text: str) -> str:
    text = text or ""
    for pat in DROP_INTRO_PATTERNS:
        text = re.sub(pat, "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_text(text: str) -> str:
    text = clean_text(text)
    text = re.sub(r"\bCRP\b", "CPR", text, flags=re.IGNORECASE)
    return text.strip()

def stable_hash(text: str) -> str:
    return hashlib.sha1(text.lower().encode("utf-8")).hexdigest()

def sentence_split(text: str):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

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

def contains_any(text: str, patterns) -> bool:
    t = text.lower()
    return any(p in t for p in patterns)

def is_definition_only(text: str) -> bool:
    t = text.strip().lower()
    return any(re.match(p, t, flags=re.IGNORECASE) for p in DEFINITION_ONLY_PATTERNS)

def has_leftover_noise(text: str) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in LEFTOVER_NOISE_PATTERNS)

def has_action_signal(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in ACTION_WORDS)

def keyword_score(text: str, intent: str) -> int:
    t = text.lower()
    kws = INTENT_KEYWORDS.get(intent, [])
    return sum(1 for kw in kws if kw.lower() in t)

def negative_score(text: str, intent: str) -> int:
    t = text.lower()
    kws = NEGATIVE_KEYWORDS.get(intent, [])
    return sum(1 for kw in kws if kw.lower() in t)

def score_all_intents(text: str):
    scores = {}
    for intent in INTENT_KEYWORDS:
        s = keyword_score(text, intent)
        if s > 0:
            scores[intent] = s
    return scores

def best_intent_match(text: str):
    scores = score_all_intents(text)
    if not scores:
        return None, 0, {}

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_intent, best_score = ranked[0]
    return best_intent, best_score, scores

def is_mixed_topic(current_intent: str, best_intent: str, scores: dict) -> bool:
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if len(ranked) < 2:
        return False

    top_intent, top_score = ranked[0]
    second_intent, second_score = ranked[1]

    if second_score < 2:
        return False

    if second_intent in RELATED_INTENTS.get(top_intent, set()):
        return False

    return second_score >= top_score - 1

def should_drop(text: str, current_intent: str, source: str) -> bool:
    wc = len(text.split())
    if wc < MIN_WORDS:
        return True
    if is_definition_only(text):
        return True
    if has_leftover_noise(text):
        return True
    if contains_any(text, BAD_TOPICS):
        return True
    if contains_any(text, BAD_ANECDOTAL):
        return True

    best_intent, best_score, scores = best_intent_match(text)
    current_score = keyword_score(text, current_intent)
    current_neg = negative_score(text, current_intent)

    # For superdataset, be more strict
    if source.lower() == "superdataset.json":
        if not has_action_signal(text) and best_score < 2:
            return True

    # If current intent is contradicted by negative evidence and weak positive evidence
    if current_neg >= 1 and current_score <= 1:
        return True

    # No usable first-aid signal
    if best_score == 0:
        return True

    # Ambiguous mixed-topic blocks are poison for retrieval
    if best_intent and is_mixed_topic(current_intent, best_intent, scores):
        return True

    return False

def infer_better_intent(current_intent: str, text: str) -> str:
    best_intent, best_score, scores = best_intent_match(text)
    current_score = keyword_score(text, current_intent)
    current_neg = negative_score(text, current_intent)

    if not best_intent:
        return current_intent

    # Keep current if it has strong support and no strong contradiction
    if current_score >= 2 and current_neg == 0:
        return current_intent

    # Reassign only on strong positive evidence
    if best_score >= 2 and best_intent != current_intent:
        return best_intent

    return current_intent

def load_severity_map():
    if not CANONICAL_FILE.exists():
        return {}
    with open(CANONICAL_FILE, "r", encoding="utf-8") as f:
        canonical = json.load(f)

    severity_map = {}
    for intent, payload in canonical.items():
        severity_map[intent] = payload.get("severity", "routine")
    return severity_map

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    severity_map = load_severity_map()

    final_chunks = []
    seen = set()
    relabel_counts = Counter()
    drop_count = 0
    kept_by_intent = Counter()

    for row in chunks:
        original_intent = row.get("intent", "").strip()
        source = row.get("source", "unknown")
        language = row.get("language", "en")
        age_group = row.get("age_group", "general")
        text = normalize_text(row.get("text", ""))

        if not text or not original_intent:
            drop_count += 1
            continue

        packed = pack_sentences(sentence_split(text))

        for chunk_text in packed:
            new_intent = infer_better_intent(original_intent, chunk_text)

            if should_drop(chunk_text, new_intent, source):
                drop_count += 1
                continue

            severity = severity_map.get(new_intent, row.get("severity", "routine"))

            if new_intent != original_intent:
                relabel_counts[(original_intent, new_intent)] += 1

            h = stable_hash(f"{new_intent}|{age_group}|{chunk_text}")
            if h in seen:
                continue
            seen.add(h)

            final_chunks.append({
                "id": str(uuid.uuid4()),
                "intent": new_intent,
                "severity": severity,
                "age_group": age_group,
                "language": language,
                "type": "procedure",
                "text": chunk_text,
                "source": source
            })
            kept_by_intent[new_intent] += 1

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_chunks, f, indent=2, ensure_ascii=False)

    print(f"✅ Wrote {len(final_chunks)} final knowledge chunks to {OUTPUT_FILE}")
    print(f"🗑️ Dropped chunks: {drop_count}")
    print("📊 Kept by intent:")
    for intent, n in kept_by_intent.most_common():
        print(f"  {intent}: {n}")

    if relabel_counts:
        print("🔁 Relabeled chunks:")
        for (old_i, new_i), n in relabel_counts.most_common(25):
            print(f"  {old_i} -> {new_i}: {n}")

if __name__ == "__main__":
    main()