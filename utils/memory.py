#utils/memory.py
from collections import defaultdict

SESSION_MEMORY = defaultdict(list)
MAX_TURNS = 6  # last 6 messages only

def add_to_memory(session_id: str, role: str, content: str):
    SESSION_MEMORY[session_id].append({
        "role": role,
        "content": content
    })
    SESSION_MEMORY[session_id] = SESSION_MEMORY[session_id][-MAX_TURNS:]


def get_memory(session_id: str):
    return SESSION_MEMORY.get(session_id, [])
