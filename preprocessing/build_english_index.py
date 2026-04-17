# build_english_index.py
import os
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# -----------------------------
# Config
# -----------------------------
INDEX_DIR = "data/indexes"
os.makedirs(INDEX_DIR, exist_ok=True)

INDEX_PATH = os.path.join(INDEX_DIR, "english_faiss.index")
TEXTS_PATH = os.path.join(INDEX_DIR, "english_texts.pkl")

# -----------------------------
# Load dataset
# -----------------------------
print("📥 Loading English dataset...")
ds = load_dataset("lextale/FirstAidInstructionsDataset")
eng_texts = [item["Instruction"] for item in ds["train"]]

print(f"✅ Loaded {len(eng_texts)} English first-aid instructions.")

# -----------------------------
# Embeddings
# -----------------------------
print("🔍 Generating embeddings...")
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
eng_embeddings = model.encode(eng_texts, convert_to_numpy=True, show_progress_bar=True)

# -----------------------------
# FAISS Index
# -----------------------------
print("⚡ Building FAISS index...")
dimension = eng_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(eng_embeddings)

# -----------------------------
# Save to disk
# -----------------------------
faiss.write_index(index, INDEX_PATH)

with open(TEXTS_PATH, "wb") as f:
    pickle.dump(eng_texts, f)

print(f"✅ English FAISS index saved to {INDEX_PATH}")
print(f"✅ Texts saved to {TEXTS_PATH}")
