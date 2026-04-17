from fastapi import APIRouter, HTTPException
from schemas.chat import ChatRequest, ChatResponse
from prompts.summarize import build_messages
from prompts.emergency_messages import (
    EMERGENCY_CRITICAL,
    EMERGENCY_URGENT
)
from utils.memory import add_to_memory, get_memory
from utils.llm import generate_llm_response
import os

# ---------------- MODE SWITCH ----------------
APP_MODE = os.getenv("APP_MODE", "local").lower()

if APP_MODE == "cloud":
    from intent_router.router_cloud import detect_intent
    from retrieval.retrieve_cloud import retrieve_knowledge
else:
    from intent_router.router import detect_intent
    from retrieval.retrieve_knowledge import retrieve_knowledge

router = APIRouter()

# ---------------- CONSTANTS ----------------
CRITICAL_INTENTS = {
    "chest_pain",
    "cpr_resuscitation",
    "stroke_signs",
    "drowning_near_drowning",
    "electric_shock",
}

URGENT_INTENTS = {
    "snake_bite",
    "head_injury",
    "severe_wound",
    "allergic_reaction",
    "breathing_difficulty",
    "seizure",
    "poisoning",
    "bleeding",
    "burns",
}

# ---------------- HELPERS ----------------
def normalize_text(text: str) -> str:
    return (text or "").strip().lower()


def has_all(text: str, terms: list[str]) -> bool:
    return all(term in text for term in terms)


def has_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


# ---------------- EMERGENCY DETECTION ----------------
def detect_query_emergency(query: str, intent_results: list[dict]) -> str | None:
    q = normalize_text(query)

    critical_patterns = [
        ["chest pain", "sweating"],
        ["not breathing"],
        ["unconscious"],
        ["severe bleeding"],
        ["possible stroke"],
        ["electric shock"],
        ["drowning"],
    ]

    for pattern in critical_patterns:
        if has_all(q, pattern):
            return "critical"

    critical_any = [
        "heart attack",
        "ambulance",
        "cpr",
        "call 1122",
    ]

    if has_any(q, critical_any):
        return "critical"

    if intent_results:
        top_intents = [x["intent"] for x in intent_results[:3]]

        if any(i in CRITICAL_INTENTS for i in top_intents):
            return "critical"
        if any(i in URGENT_INTENTS for i in top_intents):
            return "urgent"

    return None


# ---------------- TOPIC GUARD ----------------
def answer_topic_mismatch(intent: str, answer: str) -> bool:
    text = answer.lower()

    mismatch_terms = {
        "burns": ["heimlich", "abdominal thrust"],
        "choking": ["cool running water", "blister"],
        "chest_pain": ["heimlich"],
    }

    if intent in mismatch_terms:
        return any(term in text for term in mismatch_terms[intent])

    return False


def build_emergency_prefix(language: str, emergency_type: str | None) -> str:
    if emergency_type == "critical":
        return EMERGENCY_CRITICAL[language] + "\n\n"
    if emergency_type == "urgent":
        return EMERGENCY_URGENT[language] + "\n\n"
    return ""


# ---------------- MAIN ROUTE ----------------
@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):

    # 1️⃣ Intent detection
    intent_results = detect_intent(req.query)

    if not intent_results:
        raise HTTPException(status_code=400, detail="Unable to determine intent.")

    primary_intent = intent_results[0]["intent"]
    primary_severity = intent_results[0].get("severity", "routine")

    # 2️⃣ Emergency detection
    emergency_type = detect_query_emergency(req.query, intent_results)

    effective_severity = primary_severity
    if emergency_type == "critical":
        effective_severity = "critical"
    elif emergency_type == "urgent":
        effective_severity = "urgent"

    # 3️⃣ Retrieval
    retrieved = retrieve_knowledge(
        query=req.query,
        intent_results=intent_results,
        age_group=req.age_group,
        candidate_k=40,
        top_k=5,
        min_score=0.20,
    )

    # 4️⃣ Fallback
    if not retrieved:
        fallback = (
            "I could not find enough reliable first-aid information for that query. "
            "Please add more detail."
            if req.language == "en"
            else "براہِ کرم مزید تفصیل فراہم کریں۔"
        )

        answer = build_emergency_prefix(req.language, emergency_type) + fallback

        add_to_memory(req.session_id, "user", req.query)
        add_to_memory(req.session_id, "assistant", answer)

        return ChatResponse(intent=primary_intent, answer=answer)

    # 5️⃣ Prompt building
    prompt_messages = build_messages(
        user_query=req.query,
        retrieved=retrieved,
        language=req.language,
    )

    history = get_memory(req.session_id)[-4:]
    messages = [prompt_messages[0]] + history + [prompt_messages[1]]

    # 6️⃣ LLM generation
    answer = generate_llm_response(messages=messages, temperature=0.2)

    # 7️⃣ Safety guard
    if answer_topic_mismatch(primary_intent, answer):
        raise HTTPException(status_code=500, detail="Answer-topic mismatch")

    # 8️⃣ Emergency prefix
    answer = build_emergency_prefix(req.language, emergency_type) + answer

    # 9️⃣ Save memory
    add_to_memory(req.session_id, "user", req.query)
    add_to_memory(req.session_id, "assistant", answer)

    return ChatResponse(
        intent=primary_intent,
        answer=answer
    )