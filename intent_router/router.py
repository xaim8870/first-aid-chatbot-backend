#intent_router/router.py
import faiss
import pickle
from collections import defaultdict
from common.embeddings import embed_texts

INDEX_PATH = "data/intent_router/intent_index.faiss"
MAP_PATH = "data/intent_router/intent_map.pkl"

index = faiss.read_index(INDEX_PATH)

with open(MAP_PATH, "rb") as f:
    intent_map = pickle.load(f)


def normalize_faiss_score(raw_score: float, metric_type: int) -> float:
    if metric_type == faiss.METRIC_INNER_PRODUCT:
        return float(raw_score)
    if metric_type == faiss.METRIC_L2:
        return 1.0 / (1.0 + float(raw_score))
    return float(raw_score)


def detect_intent(text: str, top_k: int = 8):
    emb = embed_texts([text])
    scores, idxs = index.search(emb, top_k)

    metric_type = getattr(index, "metric_type", faiss.METRIC_INNER_PRODUCT)

    buckets = defaultdict(lambda: {
        "intent": None,
        "severity": "routine",
        "scores": []
    })

    for raw_score, idx in zip(scores[0], idxs[0]):
        if idx < 0:
            continue

        row = intent_map[idx]

        if isinstance(row, dict):
            intent = row["intent"]
            severity = row.get("severity", "routine")
        else:
            intent = row
            severity = "routine"

        sim = normalize_faiss_score(raw_score, metric_type)

        buckets[intent]["intent"] = intent
        buckets[intent]["severity"] = severity
        buckets[intent]["scores"].append(sim)

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
    return results[:3]