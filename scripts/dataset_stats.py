import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "raw" / "superdataset.json"
OUT_JSON = BASE_DIR / "data" / "stats" / "dataset_stats.json"
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

def is_valid_text(x):
    return isinstance(x, str) and x.strip() != ""

def main():
    stats = {
        "total_records": 0,
        "question": {
            "valid": 0,
            "null": 0,
            "empty": 0,
            "missing": 0
        },
        "answer": {
            "valid": 0,
            "null": 0,
            "empty": 0,
            "missing": 0
        },
        "combined": {
            "both_valid": 0,
            "only_question_valid": 0,
            "only_answer_valid": 0,
            "neither_valid": 0
        }
    }

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            stats["total_records"] += 1
            item = json.loads(line)

            q_exists = "question" in item
            a_exists = "answer" in item

            q = item.get("question")
            a = item.get("answer")

            # Question stats
            if not q_exists:
                stats["question"]["missing"] += 1
            elif q is None:
                stats["question"]["null"] += 1
            elif isinstance(q, str) and q.strip() == "":
                stats["question"]["empty"] += 1
            else:
                stats["question"]["valid"] += 1

            # Answer stats
            if not a_exists:
                stats["answer"]["missing"] += 1
            elif a is None:
                stats["answer"]["null"] += 1
            elif isinstance(a, str) and a.strip() == "":
                stats["answer"]["empty"] += 1
            else:
                stats["answer"]["valid"] += 1

            # Combined
            q_ok = is_valid_text(q)
            a_ok = is_valid_text(a)

            if q_ok and a_ok:
                stats["combined"]["both_valid"] += 1
            elif q_ok and not a_ok:
                stats["combined"]["only_question_valid"] += 1
            elif a_ok and not q_ok:
                stats["combined"]["only_answer_valid"] += 1
            else:
                stats["combined"]["neither_valid"] += 1

    # Save JSON
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("✅ Dataset statistics saved")
    print(f"📁 {OUT_JSON}")

if __name__ == "__main__":
    main()
