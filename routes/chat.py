from fastapi import APIRouter, HTTPException
from schemas.chat import ChatRequest, ChatResponse
from retrieval.retrieve_knowledge import retrieve_knowledge
from intent_router.router import detect_intent
from prompts.summarize import build_messages
from prompts.emergency_messages import (
    EMERGENCY_CRITICAL,
    EMERGENCY_URGENT
)
from utils.memory import add_to_memory, get_memory
from utils.llm import generate_llm_response

router = APIRouter()

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


def normalize_text(text: str) -> str:
    return (text or "").strip().lower()


def has_all(text: str, terms: list[str]) -> bool:
    return all(term in text for term in terms)


def has_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


def detect_query_emergency(query: str, intent_results: list[dict]) -> str | None:
    q = normalize_text(query)

    critical_patterns = [
        ["chest pain", "sweating"],
        ["severe chest pain"],
        ["not breathing"],
        ["stopped breathing"],
        ["unconscious"],
        ["unresponsive"],
        ["severe bleeding"],
        ["bleeding heavily"],
        ["possible stroke"],
        ["electric shock"],
        ["drowning"],

        ["سینے", "درد"],
        ["سانس", "نہیں"],
        ["سانس", "بند"],
        ["بے ہوش"],
        ["شدید", "خون"],
        ["فالج"],
        ["کرنٹ"],
        ["ڈوب"],
    ]

    for pattern in critical_patterns:
        if has_all(q, pattern):
            return "critical"

    critical_any = [
        "heart attack",
        "ambulance",
        "cpr",
        "call 1122",
        "call ambulance",
        "possible heart attack",
        "possible stroke",
        "دل کا دورہ",
        "ایمبولینس",
        "سی پی آر",
        "1122",
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


def answer_topic_mismatch(intent: str, answer: str) -> bool:
    text = answer.lower()

    mismatch_terms = {
        "burns": ["abdominal thrust", "heimlich", "xiphoid", "navel"],
        "choking": ["cool running water", "blister", "chemical burn"],
        "chest_pain": ["abdominal thrust", "heimlich", "cool running water"],
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


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    print("\n" + "=" * 80)
    print("📩 NEW CHAT REQUEST")
    print(f"QUERY: {req.query}")
    print(f"LANGUAGE: {req.language}")
    print(f"AGE GROUP: {req.age_group}")
    print(f"SESSION ID: {req.session_id}")

    # 1) Intent routing
    intent_results = detect_intent(req.query)
    print(f"INTENT RESULTS: {intent_results}")

    if not intent_results:
        raise HTTPException(status_code=400, detail="Unable to determine intent.")

    primary_intent = intent_results[0]["intent"]
    primary_severity = intent_results[0].get("severity", "routine")
    print(f"✅ PRIMARY INTENT: {primary_intent}")
    print(f"✅ PRIMARY SEVERITY: {primary_severity}")

    # 2) Emergency detection
    emergency_type = detect_query_emergency(req.query, intent_results)
    print(f"🚨 QUERY EMERGENCY TYPE: {emergency_type}")

    # If router severity is higher, respect that too
    effective_severity = primary_severity
    if emergency_type == "critical":
        effective_severity = "critical"
    elif emergency_type == "urgent" and effective_severity != "critical":
        effective_severity = "urgent"

    print(f"🩺 EFFECTIVE SEVERITY: {effective_severity}")

    # 3) Retrieve grounded knowledge
    retrieved = retrieve_knowledge(
        query=req.query,
        intent_results=intent_results,
        age_group=req.age_group,
        candidate_k=40,
        top_k=6,
        min_score=0.20,
    )

    print(f"RETRIEVED COUNT: {len(retrieved)}")
    if retrieved:
        for i, chunk in enumerate(retrieved, start=1):
            preview = chunk["text"][:180].replace("\n", " ")
            print(
                f"[{i}] score={chunk['score']:.4f} "
                f"intent={chunk.get('intent')} "
                f"age_group={chunk.get('age_group')} "
                f"source={chunk.get('source')} "
                f"text={preview}..."
            )

    # 4) Fallback if retrieval fails
    if not retrieved:
        if req.language == "ur":
            fallback = (
                "مجھے اس سوال کے لیے کافی قابلِ اعتماد ابتدائی طبی معلومات نہیں مل سکیں۔ "
                "براہِ کرم علامات، جسم کے حصے، یا کیا ہوا ہے اس کی مزید تفصیل لکھیں۔"
            )
        else:
            fallback = (
                "I could not find enough reliable first-aid information for that query. "
                "Please add more detail, such as what happened, the symptoms, or the body part involved."
            )

        answer = build_emergency_prefix(req.language, emergency_type) + fallback

        add_to_memory(req.session_id, "user", req.query)
        add_to_memory(req.session_id, "assistant", answer)

        print("↩️ Returning fallback response")
        print("=" * 80)

        return ChatResponse(intent=primary_intent, answer=answer)

    # 5) Build grounded prompt
    prompt_messages = build_messages(
        user_query=req.query,
        retrieved=retrieved,
        language=req.language,
        severity=effective_severity,
    )

    print("🧾 PROMPT PREVIEW:")
    print(prompt_messages[-1]["content"][:800] + ("..." if len(prompt_messages[-1]["content"]) > 800 else ""))

    # 6) Short memory
    history = get_memory(req.session_id)[-4:]
    print(f"🧠 MEMORY ITEMS USED: {len(history)}")

    # Keep system first, then short history, then grounded user prompt
    messages = [prompt_messages[0]] + history + [prompt_messages[1]]

    # 7) Generate answer
    answer = generate_llm_response(
        messages=messages,
        temperature=0.2
    )

    print("🤖 ANSWER PREVIEW:")
    print(answer[:800] + ("..." if len(answer) > 800 else ""))

    # 8) Safety/topic guard
    if answer_topic_mismatch(primary_intent, answer):
        raise HTTPException(
            status_code=500,
            detail=f"Generated answer-topic mismatch for intent '{primary_intent}'."
        )

    # 9) Prepend emergency notice, do not append
    answer = build_emergency_prefix(req.language, emergency_type) + answer

    # 10) Save memory
    add_to_memory(req.session_id, "user", req.query)
    add_to_memory(req.session_id, "assistant", answer)

    print("✅ RESPONSE READY")
    print("=" * 80)

    return ChatResponse(
        intent=primary_intent,
        answer=answer
    )