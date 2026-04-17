import json
import re
from collections import defaultdict

# =========================
# CONFIG
# =========================
INPUT_FILE = "data/clean/knowledge_docs.json"
OUTPUT_FILE = "data/analysis/sub_intent_counts.json"

# =========================
# MEDICAL KEYWORD MAP
# (initial, extendable)
# =========================
INTENT_KEYWORDS = {
    "vertigo": [
        "vertigo", "dizziness", "spinning", "room is spinning"
    ],
    "fainting": [
        "faint", "fainted", "fainting", "loss of consciousness"
    ],
    "heat_stroke": [
        "heat stroke", "heat exhaustion", "overheating", "high temperature"
    ],
    "hypothermia": [
        "hypothermia", "extreme cold", "freezing", "low body temperature"
    ],
    "seizure": [
        "seizure", "epileptic", "convulsion", "fits"
    ],
    "shock": [
        "shock", "traumatic shock", "circulatory failure"
    ],
    "allergic_reaction": [
        "allergic", "anaphylaxis", "allergy"
    ],
    "unconsciousness": [
        "unconscious", "unresponsive", "passed out"
    ],
    "poisoning": [
        "poison", "toxic", "overdose"
    ],
    "burns": [
        "burn", "scald", "thermal injury"
    ],
    "bleeding": [
        "bleeding", "hemorrhage", "blood loss"
    ],
    "fracture": [
        "fracture", "broken bone", "bone injury"
    ],
    "choking": [
        "choking", "airway obstruction", "cannot breathe"
    ],
    "snake_bite": [
        "snake bite", "snakebite", "venomous bite"
    ],
}

# =========================
# LOAD DATA
# =========================
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Loaded {len(data)} documents")

# =========================
# INTENT EXTRACTION
# =========================
intent_counts = defaultdict(int)
intent_examples = defaultdict(list)

for item in data:
    text = item["text"].lower()

    for intent, keywords in INTENT_KEYWORDS.items():
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", text):
                intent_counts[intent] += 1
                if len(intent_examples[intent]) < 3:
                    intent_examples[intent].append(item["text"][:200])
                break  # avoid double count per document

# =========================
# SAVE RESULTS
# =========================
output = {
    "intent_counts": dict(sorted(intent_counts.items(), key=lambda x: x[1], reverse=True)),
    "examples": intent_examples
}

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("Saved sub-intent analysis to:", OUTPUT_FILE)

# =========================
# PRINT SUMMARY
# =========================
print("\n=== SUB-INTENT COUNTS ===")
for intent, count in sorted(intent_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"{intent:20s} -> {count}")
