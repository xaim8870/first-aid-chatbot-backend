# prompts/emergency_messages.py
# Emergency message templates (bilingual)

EMERGENCY_CRITICAL = {
    "en": (
        "🚨 Emergency Notice\n"
        "If the person is unconscious, not breathing, bleeding heavily, having severe chest pain, or symptoms are rapidly worsening:\n\n"
        "• Pakistan: Call **1122** immediately\n"
        "• Common global numbers: **112** (many countries), **911** (US/Canada), **999/112** (UK)\n"
        "• If you cannot call, ask someone nearby to call while you start first aid.\n\n"
        "This guidance supports first aid only and does not replace professional medical care."
    ),
    "ur": (
        "🚨 ہنگامی اطلاع\n"
        "اگر مریض بے ہوش ہو جائے، سانس بند ہو جائے، بہت زیادہ خون بہہ رہا ہو، سینے میں شدید درد ہو، یا حالت تیزی سے بگڑ رہی ہو تو:\n\n"
        "• پاکستان: فوراً **1122** پر کال کریں\n"
        "• عام عالمی نمبر: **112** (کئی ممالک)، **911** (امریکہ/کینیڈا)، **999/112** (برطانیہ)\n"
        "• اگر آپ کال نہ کر سکیں تو کسی قریب موجود شخص سے کہیں کہ وہ کال کرے جبکہ آپ ابتدائی طبی امداد شروع کریں۔\n\n"
        "یہ معلومات ابتدائی طبی امداد کے لیے ہیں اور پیشہ ور طبی علاج کا متبادل نہیں۔"
    ),
}

EMERGENCY_URGENT = {
    "en": (
        "⚠️ Medical Advice\n"
        "If symptoms do not improve, return, or worsen, seek medical care as soon as possible.\n\n"
        "For emergencies:\n"
        "• Pakistan: **1122**\n"
        "• Elsewhere: your local emergency number (often **112** or **911**)"
    ),
    "ur": (
        "⚠️ طبی مشورہ\n"
        "اگر علامات بہتر نہ ہوں، دوبارہ ہو جائیں، یا بگڑ جائیں تو جلد از جلد طبی مدد حاصل کریں۔\n\n"
        "ہنگامی صورت میں:\n"
        "• پاکستان: **1122**\n"
        "• دیگر ممالک: مقامی ایمرجنسی نمبر (اکثر **112** یا **911**)"
    ),
}
