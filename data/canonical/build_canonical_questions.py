import json
import os

SEEDS_FILE = "data/canonical/intent_seeds.json"
OUT_FILE = "data/canonical/canonical_questions.json"

EN_SUBJECTS = [
    "I have",
    "my father has",
    "my mother has",
    "my child has",
    "my baby has",
    "someone has"
]

UR_SUBJECTS = [
    "مجھے",
    "میرے والد کو",
    "میری والدہ کو",
    "میرے بچے کو",
    "میرے بچے کو",
    "کسی کو"
]

MIXED_SUBJECTS = [
    "mujhe",
    "mere father ko",
    "meri mother ko",
    "mere child ko",
    "mere baby ko",
    "someone ko"
]

EN_TEMPLATES = [
    "What should I do for {phrase}?",
    "What first aid should be given for {phrase}?",
    "How do I help someone with {phrase}?",
    "{subject} {phrase}, what should I do?",
    "{subject} {phrase}. What first aid is needed?",
    "Emergency help for {phrase}",
    "First aid for {phrase}",
    "How to treat {phrase} at home?"
]

UR_TEMPLATES = [
    "{phrase} کی صورت میں کیا کرنا چاہیے؟",
    "{phrase} میں ابتدائی طبی امداد کیا ہے؟",
    "{phrase} ہو تو کیا کریں؟",
    "{subject} {phrase} ہے، کیا کرنا چاہیے؟",
    "{subject} {phrase} ہو گیا ہے، کیا کریں؟",
    "{phrase} کا فرسٹ ایڈ علاج کیا ہے؟",
    "{phrase} میں فوری کیا کرنا چاہیے؟"
]

MIXED_TEMPLATES = [
    "{phrase}, what should I do?",
    "{subject} {phrase}, what first aid is needed?",
    "{phrase} mein first aid kya hai?",
    "{phrase} ke liye first aid steps batao",
    "{subject} {phrase} hai, kya karna chahiye?"
]

MAX_TOTAL = 60
MAX_EN = 20
MAX_UR = 20
MAX_MIXED = 20


def dedupe_keep_order(items):
    seen = set()
    out = []
    for item in items:
        norm = " ".join(item.strip().split()).lower()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(item.strip())
    return out


def expand_phrases(phrases, templates, subjects):
    examples = []
    for phrase in phrases:
        phrase = (phrase or "").strip()
        if not phrase:
            continue
        examples.append(phrase)
        for template in templates:
            if "{subject}" in template:
                for subject in subjects:
                    examples.append(template.format(subject=subject, phrase=phrase))
            else:
                examples.append(template.format(phrase=phrase))
    return examples


def take_balanced(en_list, ur_list, mixed_list,
                  max_en=MAX_EN, max_ur=MAX_UR, max_mixed=MAX_MIXED, max_total=MAX_TOTAL):
    en_list = dedupe_keep_order(en_list)[:max_en]
    ur_list = dedupe_keep_order(ur_list)[:max_ur]
    mixed_list = dedupe_keep_order(mixed_list)[:max_mixed]

    combined = []
    max_len = max(len(en_list), len(ur_list), len(mixed_list))

    for i in range(max_len):
        if i < len(en_list):
            combined.append(en_list[i])
        if i < len(ur_list):
            combined.append(ur_list[i])
        if i < len(mixed_list):
            combined.append(mixed_list[i])

    return dedupe_keep_order(combined)[:max_total]


def main():
    with open(SEEDS_FILE, "r", encoding="utf-8") as f:
        seeds = json.load(f)

    canonical = {}

    for intent, cfg in seeds.items():
        phrases_en = list(cfg.get("phrases_en", []))
        phrases_ur = list(cfg.get("phrases_ur", []))
        phrases_mixed = list(cfg.get("phrases_mixed", []))

        # Use aliases as additional English coverage
        aliases = cfg.get("aliases", [])
        if aliases:
            phrases_en.extend(aliases)

        en_examples = expand_phrases(phrases_en, EN_TEMPLATES, EN_SUBJECTS)
        ur_examples = expand_phrases(phrases_ur, UR_TEMPLATES, UR_SUBJECTS)
        mixed_examples = expand_phrases(phrases_mixed, MIXED_TEMPLATES, MIXED_SUBJECTS)

        canonical[intent] = {
            "severity": cfg.get("severity", "routine"),
            "age_focus": cfg.get("age_focus", ["general"]),
            "examples": take_balanced(en_examples, ur_examples, mixed_examples)
        }

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(canonical, f, ensure_ascii=False, indent=2)

    total = sum(len(v["examples"]) for v in canonical.values())
    print(f"✅ Wrote {len(canonical)} intents and {total} examples to {OUT_FILE}")


if __name__ == "__main__":
    main()