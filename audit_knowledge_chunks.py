#audit_knowledge_chunks.py
import json
import random
import re
from pathlib import Path
from collections import Counter, defaultdict

ROOT_DIR = Path(__file__).resolve().parent
INPUT_FILE = ROOT_DIR / "data" / "knowledge" / "knowledge_chunks.json"
REPORT_FILE = ROOT_DIR / "data" / "knowledge" / "knowledge_audit_report.json"

LEFTOVER_NOISE = [
    "question variations",
    "how do you treat",
    "what to do if i get",
    "how to cure",
    "<positive_smiley>",
    "i am not a doctor",
    "this helped me personally",
]

BAD_TOPICS = [
    "pregnant", "pregnancy", "gynecologist", "thyroid",
    "weight loss", "chest congestion", "ovulation", "period"
]

INTENT_KEYWORDS = {
    "fracture": ["fracture", "broken bone", "splint", "immobilize", "broken toe", "dislocated bone"],
    "sprain_strain": ["sprain", "strain", "elastic bandage", "ice", "joint immobilization", "rest"],
    "cpr_resuscitation": ["cpr", "compressions", "rescue breaths", "30 compressions", "2 rescue breaths"],
    "choking": ["choking", "back blows", "abdominal thrusts", "foreign body", "cannot breathe"],
    "bleeding": ["bleeding", "blood loss", "direct pressure", "apply pressure", "nosebleed"],
    "chemical_exposure": ["chemical", "flush with water", "chemical burn", "chemical in eye"],
    "spinal_injury": ["spinal injury", "neck injury", "stabilize head and neck", "do not move the person"],
    "amputation_avulsion": ["amputated limb", "cut off limb", "severed", "amputation"],

    "seizure": ["seizure", "convulsion", "fit", "fits", "protect from injury"],
    "burns": ["burn", "scald", "cool the burn", "running water", "blister"],
    "allergic_reaction": ["allergic", "allergy", "anaphylaxis", "swelling", "allergic reaction"],
    "chest_pain": ["chest pain", "heart attack", "tightness in chest", "pain to arm", "sweating"],
    "poisoning": ["poison", "poisoning", "toxic", "swallowed poison"],
    "headache": ["headache", "migraine", "head pain"],
    "vertigo": ["vertigo", "dizziness", "room spinning"],
    "asthma_attack": ["asthma", "wheezing", "inhaler", "asthma attack"],
    "breathing_difficulty": ["difficulty breathing", "shortness of breath", "breathless", "hard to breathe"],
    "dehydration": ["dehydration", "dry mouth", "oral rehydration", "rehydration"],
    "temperature_emergency": ["heatstroke", "heat exhaustion", "hypothermia", "frostbite"],
    "electric_shock": ["electric shock", "electrocution", "current", "live wire"],
    "nosebleed": ["nosebleed", "blood from nose", "nose is bleeding"],
    "animal_bite": ["animal bite", "dog bite", "cat bite", "rabies"],
    "snake_bite": ["snake bite", "bitten by a snake", "venomous"],
    "eye_injury": ["eye injury", "foreign object in eye", "something in the eye", "do not rub the eye"],
    "fainting": ["fainted", "fainting", "passed out", "lost consciousness"],
    "dehydration": ["dehydration", "dry mouth", "sunken eyes", "rehydration"],
}
def keyword_hits(intent, text):
    kws = INTENT_KEYWORDS.get(intent, [])
    t = text.lower()
    return sum(1 for kw in kws if kw in t)

def contains_any(text, terms):
    t = text.lower()
    return any(term in t for term in terms)

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    counts = Counter()
    suspicious = []
    by_intent = defaultdict(list)

    for row in chunks:
        intent = row.get("intent", "")
        text = row.get("text", "")
        source = row.get("source", "")

        counts["total"] += 1
        counts[f"intent::{intent}"] += 1
        counts[f"source::{source}"] += 1
        by_intent[intent].append(row)

        hit_count = keyword_hits(intent, text)

        if contains_any(text, LEFTOVER_NOISE):
            counts["leftover_noise"] += 1
            suspicious.append({"reason": "leftover_noise", "intent": intent, "text": text})

        if contains_any(text, BAD_TOPICS):
            counts["bad_topic"] += 1
            suspicious.append({"reason": "bad_topic", "intent": intent, "text": text})

        if hit_count == 0:
            counts["zero_intent_hits"] += 1
            suspicious.append({"reason": "zero_intent_hits", "intent": intent, "text": text})

        if len(text.split()) < 18:
            counts["too_short"] += 1
            suspicious.append({"reason": "too_short", "intent": intent, "text": text})

    sample_by_intent = {}
    random.seed(42)
    for intent, rows in by_intent.items():
        sample_by_intent[intent] = random.sample(rows, min(5, len(rows)))

    report = {
        "counts": dict(counts),
        "suspicious_examples": suspicious[:200],
        "sample_by_intent": sample_by_intent,
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"✅ Audit report written to {REPORT_FILE}")
    print("Key counts:")
    for k, v in counts.most_common(20):
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()