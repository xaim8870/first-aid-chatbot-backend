#prompts/summarize.py
def build_messages(user_query, retrieved, language="en", severity="routine"):
    context = "\n\n".join(
        f"[{i+1}] source={c.get('source', 'unknown')} intent={c.get('intent', 'unknown')} age_group={c.get('age_group', 'general')}\n{c['text']}"
        for i, c in enumerate(retrieved)
    )

    if language == "ur":
        system = (
            "آپ ایک محفوظ، قابلِ اعتماد ابتدائی طبی امداد کے معاون ہیں۔ "
            "صرف فراہم کردہ معلومات کی بنیاد پر جواب دیں۔ "
            "اگر معلومات ناکافی ہوں تو صاف بتائیں۔ "
            "غیر متعلقہ بیماریوں، غیر ضروری دواؤں، یا من گھڑت مشورے شامل نہ کریں۔ "
            "ایک ہی بات کو بار بار مختلف الفاظ میں نہ دہرائیں۔ "
            "جواب مختصر، عملی اور واضح رکھیں۔"
        )

        severity_extra = ""
        if severity == "critical":
            severity_extra = "- جواب کا آغاز فوری ہنگامی تنبیہ سے کریں\n"
        elif severity == "urgent":
            severity_extra = "- واضح کریں کہ اگر حالت بگڑے تو فوراً طبی مدد لیں\n"

        user = f"""
صارف کا سوال:
{user_query}

فراہم کردہ طبی معلومات:
{context}

ہدایات:
- سوال کا سیدھا جواب دیں
- 3 سے 6 عملی اقدامات دیں
- صرف فراہم کردہ معلومات استعمال کریں
- غیر متعلقہ حالتیں شامل نہ کریں
- آخر میں بتائیں کب فوری طبی مدد لینی ہے
- اگر معلومات ناکافی ہوں تو واضح طور پر بتائیں
{severity_extra}
"""
    else:
        system = (
            "You are a safe, reliable first-aid assistant. "
            "Answer using only the provided medical context. "
            "Do not mention diseases, diagnoses, or conditions that are not explicitly supported by the provided context. "
            "Do not add extra symptom lists beyond the retrieved context. "
            "Do NOT include words like Answer: in the response"
            "Do not invent medication advice, causes, or differential diagnoses. "
            "Focus ONLY on immediate first-aid actions.\n"
            "Keep the answer short, clear, and practical.\n"
            "If the context is insufficient, say so briefly. "
            "Do not repeat yourself. "
            "Keep the answer practical, concise, and directly useful."
            "You are a safe, reliable first-aid assistant.\n"
            "Answer ONLY using the provided medical context.\n"
            "DO NOT add diseases, diagnoses, or explanations not present in the context.\n"
            "DO NOT speculate (e.g., fractures, nerve damage, internal injury) unless explicitly mentioned.\n"
            "DO NOT mention medical terms like tendinitis, bursitis, etc., unless they appear in context.\n"
            "Keep the answer practical, short, and actionable.\n"
            "Avoid repetition.\n"
        )

        severity_extra = ""
        if severity == "critical":
            severity_extra = "- Start with a short emergency warning\n"
        elif severity == "urgent":
            severity_extra = "- Clearly mention that prompt medical care may be needed if symptoms worsen\n"

        user = f"""
User question:
{user_query}

Provided medical context:
{context}

Instructions:
- Answer the actual question directly
- Give 3 to 6 practical steps
- Use only the provided context
- Do not include unrelated conditions
- End with when to seek urgent medical help
- If the context is insufficient, say that plainly
- Do not mention unrelated causes or conditions
- Do not expand into broader diagnosis lists
- Do NOT include words like "Answer:" in the response
- If this is an emergency, focus on immediate first-aid steps only
- If the query is about an infant/baby, answer only for infants unless the context is insufficient
- Do not include child or adult procedures unless explicitly labeled as a fallback
- Do not repeat the same step in different words
{severity_extra}
"""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]