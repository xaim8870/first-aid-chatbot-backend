from pydantic import BaseModel
from typing import Literal

class ChatRequest(BaseModel):
    query: str
    language: Literal["en", "ur"] = "en"
    age_group: Literal["adult", "child", "infant", "general"] = "general"
    session_id: str

class ChatResponse(BaseModel):
    intent: str
    answer: str