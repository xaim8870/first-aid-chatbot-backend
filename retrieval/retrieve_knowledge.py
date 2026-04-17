import faiss
import pickle
import re
from typing import Optional, List, Dict, Any
from common.embeddings import embed_texts

INDEX_PATH = "data/knowledge_faiss/knowledge.index"
META_PATH = "data/knowledge_faiss/knowledge_meta.pkl"

index = faiss.read_index(INDEX_PATH)

with open(META_PATH, "rb") as f:
    metadata = pickle.load(f)


def normalize_faiss_score(raw_score: float, metric_type: int) -> float:
    if metric_type == faiss.METRIC_INNER_PRODUCT:
        return float(raw_score)
    if metric_type == faiss.METRIC_L2:
        return 1.0 / (1.0 + float(raw_score))
    return float(raw_score)


def lexical_overlap_score(query: str, text: str) -> float:
    q = set(re.findall(r"\w+", (query or "").lower()))
    t = set(re.findall(r"\w+", (text or "").lower()))
    if not q:
        return 0.0
    return len(q & t) / max(len(q), 1)


def detect_language(text: str) -> str:
    if re.search(r"[\u0600-\u06FF]", text or ""):
        return "ur"
    return "en"


def normalize_text(text: str) -> str:
    return " ".join((text or "").lower().split())


def _score_candidates(
    query: str,
    scores,
    idxs,
    metric_type: int,
    intent_priors: Dict[str, float],
    age_group: Optional[str] = None,
    require_age_match: bool = True,
) -> List[Dict[str, Any]]:
    candidates = []
    seen_texts = set()
    query_language = detect_language(query)

    top_intents = set(intent_priors.keys())
    router_confident = any(v >= 0.55 for v in intent_priors.values())

    for raw_score, idx in zip(scores[0], idxs[0]):
        if idx < 0 or idx >= len(metadata):
            continue

        doc = metadata[idx]
        doc_text = (doc.get("text") or "").strip()
        doc_intent = doc.get("intent", "general")
        doc_age = doc.get("age_group", "general")
        doc_source = doc.get("source")
        doc_language = doc.get("language", "en")

        if not doc_text:
            continue

        # Age filtering
        if require_age_match and age_group:
            if doc_age not in ("general", age_group):
                continue

        # If router is reasonably confident, suppress unrelated chunks
        if router_confident and top_intents:
            if doc_intent not in top_intents and doc_intent not in ("general", "general_first_aid"):
                continue

        dedupe_key = normalize_text(doc_text)
        if dedupe_key in seen_texts:
            continue
        seen_texts.add(dedupe_key)

        dense_score = normalize_faiss_score(raw_score, metric_type)

        intent_bonus = 0.0
        if doc_intent in intent_priors:
            intent_bonus += 0.20 * intent_priors[doc_intent]

        lexical_bonus = 0.10 * lexical_overlap_score(query, doc_text)

        language_bonus = 0.0
        if doc_language == query_language:
            language_bonus += 0.03

        age_bonus = 0.0
        if age_group and doc_age == age_group:
            age_bonus += 0.04
        elif doc_age == "general":
            age_bonus += 0.01

        generic_penalty = 0.0
        if router_confident and doc_intent in ("general", "general_first_aid"):
            generic_penalty -= 0.08

        final_score = dense_score + intent_bonus + lexical_bonus + language_bonus + age_bonus + generic_penalty

        candidates.append({
            "score": round(final_score, 4),
            "dense_score": round(dense_score, 4),
            "intent_bonus": round(intent_bonus, 4),
            "lexical_bonus": round(lexical_bonus, 4),
            "language_bonus": round(language_bonus, 4),
            "age_bonus": round(age_bonus, 4),
            "generic_penalty": round(generic_penalty, 4),
            "intent": doc_intent,
            "text": doc_text,
            "age_group": doc_age,
            "language": doc_language,
            "source": doc_source,
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates


def retrieve_knowledge(
    query: str,
    intent_results: List[Dict[str, Any]],
    age_group: Optional[str] = None,
    candidate_k: int = 40,
    top_k: int = 6,
    min_score: float = 0.20,
) -> List[Dict[str, Any]]:
    if not query or not query.strip():
        return []

    if index.ntotal == 0:
        return []

    safe_candidate_k = min(max(candidate_k, top_k), index.ntotal)

    query_emb = embed_texts([query])
    scores, idxs = index.search(query_emb, safe_candidate_k)

    metric_type = getattr(index, "metric_type", faiss.METRIC_INNER_PRODUCT)

    intent_priors = {}
    for item in (intent_results or [])[:3]:
        intent = item.get("intent")
        confidence = float(item.get("confidence", 0.0))
        if intent:
            intent_priors[intent] = confidence

    candidates = _score_candidates(
        query=query,
        scores=scores,
        idxs=idxs,
        metric_type=metric_type,
        intent_priors=intent_priors,
        age_group=age_group,
        require_age_match=True,
    )

    # Fallback: relax age filter if needed
    if not candidates and age_group:
        candidates = _score_candidates(
            query=query,
            scores=scores,
            idxs=idxs,
            metric_type=metric_type,
            intent_priors=intent_priors,
            age_group=age_group,
            require_age_match=False,
        )

    candidates = [c for c in candidates if c["score"] >= min_score]

    return candidates[:top_k]