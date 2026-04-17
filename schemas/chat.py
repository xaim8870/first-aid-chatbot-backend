#backend/schemas/chat.py
from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    query: str
    language: str = "en"        # en | ur
    age_group: str = "general" # adult | child | infant | general
    session_id: str            # REQUIRED for memory


class ChatResponse(BaseModel):
    intent: str
    answer: str
