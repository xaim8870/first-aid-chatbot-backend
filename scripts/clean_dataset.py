import json
import re
import hashlib
from pathlib import Path
from collections import defaultdict

# ---------------- PATHS ----------------
BASE_DIR = Path(__file__).resolve().parents[1]

RAW_PATH = BASE_DIR / "data" / "raw" / "superdataset.json"
OUT_DIR = BASE_DIR / "data" / "clean"

KNOWLEDGE_PATH = OUT_DIR / "knowledge_docs.json"
QUESTION_MAP_PATH = OUT_DIR / "question_map.json"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- HELPERS ----------------
def is_valid_text(x):
    return isinstance(x, str) and x.strip() != ""

def clean_text(t: str) -> str:
    t = t.replace("\\n", "\n")
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def infer_category(text: str) -> str:
    t = text.lower()
    if "chok" in t:
        return "choking"
    if "burn" in t:
        return "burns"
    if "bleed" in t:
        return "bleeding"
    if "fracture" in t or "broken bone" in t:
        return "fracture"
    if "snake" in t:
        return "snake_bite"
    if "cpr" in t:
        return "cpr"
    return "general_first_aid"

def hash_answer(answer: str) -> str:
    return hashlib.sha1(answer.encode("utf-8")).hexdigest()

# ---------------- MAIN ----------------
def main():
    total = 0
    removed = 0

    answer_store = {}
    question_map = []

    with open(RAW_PATH, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            item = json.loads(line)

            q = item.get("question")
            a = item.get("answer")

            # STRICT RULE: both must be valid
            if not is_valid_text(q) or not is_valid_text(a):
                removed += 1
                continue

            q = clean_text(q)
            a = clean_text(a)

            answer_id = hash_answer(a)

            # Store unique medical knowledge
            if answer_id not in answer_store:
                answer_store[answer_id] = {
                    "id": answer_id,
                    "text": a,
                    "category": infer_category(q),
                    "source": "superdataset.json"
                }

            # Store ALL question variants
            question_map.append({
                "question": q,
                "answer_id": answer_id
            })

    # Save outputs
    with open(KNOWLEDGE_PATH, "w", encoding="utf-8") as f:
        json.dump(list(answer_store.values()), f, indent=2, ensure_ascii=False)

    with open(QUESTION_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(question_map, f, indent=2, ensure_ascii=False)

    # Report
    print("\n✅ FINAL DATASET CLEANING COMPLETE")
    print("=" * 50)
    print(f"Total original records     : {total}")
    print(f"Removed (null/empty only)  : {removed}")
    print(f"Knowledge documents        : {len(answer_store)}")
    print(f"Question variants retained: {len(question_map)}")
    print("\n📁 Outputs:")
    print(f" - {KNOWLEDGE_PATH}")
    print(f" - {QUESTION_MAP_PATH}")

if __name__ == "__main__":
    main()
