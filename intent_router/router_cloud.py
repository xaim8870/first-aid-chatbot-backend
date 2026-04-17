import json
import re
from pathlib import Path
from collections import defaultdict
from rapidfuzz import fuzz

ROOT_DIR = Path(__file__).resolve().parent.parent
CANONICAL_PATH = ROOT_DIR / "data" / "canonical" / "canonical_questions.json"


def normalize_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def lexical_overlap_score(a: str, b: str) -> float:
    a_tokens = set(re.findall(r"\w+", normalize_text(a)))
    b_tokens = set(re.findall(r"\w+", normalize_text(b)))
    if not a_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / max(len(a_tokens), 1)


with open(CANONICAL_PATH, "r", encoding="utf-8") as f:
    CANONICAL = json.load(f)

# flatten examples once at import time
CANONICAL_ROWS = []
for intent, payload in CANONICAL.items():
    severity = payload.get("severity", "routine")
    examples = payload.get("examples", [])
    for ex in examples:
        if ex and ex.strip():
            CANONICAL_ROWS.append({
                "intent": intent,
                "severity": severity,
                "text": ex.strip(),
            })


def detect_intent(text: str, top_k: int = 3):
    query = normalize_text(text)
    buckets = defaultdict(lambda: {
        "intent": None,
        "severity": "routine",
        "scores": [],
    })

    for row in CANONICAL_ROWS:
        example = normalize_text(row["text"])

        fuzz_score = fuzz.token_set_ratio(query, example) / 100.0
        overlap = lexical_overlap_score(query, example)

        final_score = (0.75 * fuzz_score) + (0.25 * overlap)

        if final_score < 0.35:
            continue

        intent = row["intent"]
        buckets[intent]["intent"] = intent
        buckets[intent]["severity"] = row["severity"]
        buckets[intent]["scores"].append(final_score)

    results = []
    for intent, row in buckets.items():
        vals = row["scores"]
        results.append({
            "intent": intent,
            "severity": row["severity"],
            "confidence": round(max(vals), 4),
            "avg_confidence": round(sum(vals) / len(vals), 4),
            "support": len(vals),
        })

    results.sort(
        key=lambda x: (x["confidence"], x["avg_confidence"], x["support"]),
        reverse=True
    )

    return results[:top_k]