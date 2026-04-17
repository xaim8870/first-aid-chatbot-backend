import json
import faiss
import numpy as np
import os
import pickle
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from common.embeddings import embed_texts

CANONICAL_FILE = ROOT_DIR / "data" / "canonical" / "canonical_questions.json"
OUT_DIR = ROOT_DIR / "data" / "intent_router"

os.makedirs(OUT_DIR, exist_ok=True)

with open(CANONICAL_FILE, "r", encoding="utf-8") as f:
    canonical = json.load(f)

texts = []
intent_map = []

for intent, payload in canonical.items():
    for ex in payload.get("examples", []):
        ex = (ex or "").strip()
        if not ex:
            continue
        texts.append(ex)
        intent_map.append({
            "intent": intent,
            "severity": payload.get("severity", "routine"),
            "age_focus": payload.get("age_focus", ["general"])
        })

if not texts:
    raise ValueError("No canonical examples found. Cannot build intent router index.")

print(f"Loaded {len(texts)} canonical examples")

embeddings = embed_texts(texts)
embeddings = np.asarray(embeddings, dtype="float32")

dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)

faiss.write_index(index, str(OUT_DIR / "intent_index.faiss"))

with open(OUT_DIR / "intent_map.pkl", "wb") as f:
    pickle.dump(intent_map, f)

with open(OUT_DIR / "intent_texts.pkl", "wb") as f:
    pickle.dump(texts, f)

with open(OUT_DIR / "intent_index_meta.json", "w", encoding="utf-8") as f:
    json.dump({
        "total_examples": len(texts),
        "embedding_dim": int(dim)
    }, f, indent=2)

print("✅ Intent router index built")