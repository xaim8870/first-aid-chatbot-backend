import json
import torch
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm
import os

# ---------------- CONFIG ----------------
INPUT_FILE = "data/clean/knowledge_docs.json"
OUTPUT_FILE = "data/urdu/knowledge_docs_ur.json"
MODEL_NAME = "Helsinki-NLP/opus-mt-en-ur"
BATCH_SIZE = 8   # safe for GPU/CPU

os.makedirs("data/urdu", exist_ok=True)

# ---------------- LOAD MODEL ----------------
print("🔄 Loading MarianMT model...")
tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model = MarianMTModel.from_pretrained(MODEL_NAME)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"✅ Using device: {device}")

# ---------------- LOAD DATA ----------------
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"📄 Loaded {len(data)} documents")

# ---------------- TRANSLATION FUNCTION ----------------
def translate_batch(texts):
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=512)

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

# ---------------- TRANSLATE ----------------
texts_en = [item["text"] for item in data]
texts_ur = []

print("🌍 Translating instructional text...")
for i in tqdm(range(0, len(texts_en), BATCH_SIZE)):
    batch = texts_en[i:i+BATCH_SIZE]
    texts_ur.extend(translate_batch(batch))

# ---------------- BUILD OUTPUT ----------------
translated = []
for i, item in enumerate(data):
    translated.append({
        "id": item.get("id"),
        "category": item.get("category"),
        "source": item.get("source"),
        "text_en": item["text"],
        "text_ur": texts_ur[i]
    })

# ---------------- SAVE ----------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(translated, f, ensure_ascii=False, indent=2)

print(f"✅ Saved Urdu dataset to: {OUTPUT_FILE}")
print("✅ Translation complete.")
