import json
import re
from pathlib import Path
from typing import Optional, List, Dict, Any
from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz

ROOT_DIR = Path(__file__).resolve().parent.parent
CHUNKS_PATH = ROOT_DIR / "data" / "knowledge" / "knowledge_chunks.json"


def normalize_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def detect_language(text: str) -> str:
    if re.search(r"[\u0600-\u06FF]", text or ""):
        return "ur"
    return "en"


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", normalize_text(text))


def lexical_overlap_score(query: str, text: str) -> float:
    q = set(tokenize(query))
    t = set(tokenize(text))
    if not q:
        return 0.0
    return len(q & t) / max(len(q), 1)


with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

TOKENIZED_CORPUS = [tokenize(row.get("text", "")) for row in metadata]
bm25 = BM25Okapi(TOKENIZED_CORPUS)


def _score_candidates(
    query: str,
    intent_results: List[Dict[str, Any]],
    age_group: Optional[str] = None,
    candidate_k: int = 40,
    require_age_match: bool = True,
) -> List[Dict[str, Any]]:
    query_tokens = tokenize(query)
    query_lang = detect_language(query)
    top_intents = {x["intent"]: float(x.get("confidence", 0.0)) for x in (intent_results or [])[:3]}
    router_confident = any(v >= 0.55 for v in top_intents.values())

    raw_scores = bm25.get_scores(query_tokens)
    ranked_indices = sorted(range(len(raw_scores)), key=lambda i: raw_scores[i], reverse=True)[:candidate_k]

    candidates = []
    seen = set()

    for idx in ranked_indices:
        doc = metadata[idx]
        doc_text = (doc.get("text") or "").strip()
        if not doc_text:
            continue

        doc_intent = doc.get("intent", "general")
        doc_age = doc.get("age_group", "general")
        doc_lang = doc.get("language", "en")
        doc_source = doc.get("source")

        if require_age_match and age_group:
            if doc_age not in ("general", age_group):
                continue

        if router_confident and top_intents:
            if doc_intent not in top_intents:
                continue

        dedupe_key = normalize_text(doc_text)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        bm25_score = float(raw_scores[idx])
        fuzz_score = fuzz.token_set_ratio(normalize_text(query), normalize_text(doc_text)) / 100.0
        lexical = lexical_overlap_score(query, doc_text)

        intent_bonus = 0.0
        if doc_intent in top_intents:
            intent_bonus += 0.18 * top_intents[doc_intent]

        age_bonus = 0.0
        if age_group and doc_age == age_group:
            age_bonus += 0.05
        elif doc_age == "general":
            age_bonus += 0.01

        language_bonus = 0.03 if doc_lang == query_lang else 0.0

        # normalize BM25 roughly
        bm25_norm = min(bm25_score / 10.0, 1.0)

        final_score = (
            0.50 * bm25_norm +
            0.20 * fuzz_score +
            0.10 * lexical +
            intent_bonus +
            age_bonus +
            language_bonus
        )

        candidates.append({
            "score": round(final_score, 4),
            "bm25_score": round(bm25_score, 4),
            "fuzz_score": round(fuzz_score, 4),
            "lexical_bonus": round(lexical, 4),
            "intent_bonus": round(intent_bonus, 4),
            "age_bonus": round(age_bonus, 4),
            "language_bonus": round(language_bonus, 4),
            "intent": doc_intent,
            "text": doc_text,
            "age_group": doc_age,
            "language": doc_lang,
            "source": doc_source,
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates


def retrieve_knowledge(
    query: str,
    intent_results: List[Dict[str, Any]],
    age_group: Optional[str] = None,
    candidate_k: int = 40,
    top_k: int = 5,
    min_score: float = 0.20,
) -> List[Dict[str, Any]]:
    if not query or not query.strip():
        return []

    candidates = _score_candidates(
        query=query,
        intent_results=intent_results,
        age_group=age_group,
        candidate_k=candidate_k,
        require_age_match=True,
    )

    if not candidates and age_group:
        candidates = _score_candidates(
            query=query,
            intent_results=intent_results,
            age_group=age_group,
            candidate_k=candidate_k,
            require_age_match=False,
        )

    candidates = [c for c in candidates if c["score"] >= min_score]
    return candidates[:top_k]