import json
import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

INPUT_FILE = "data/knowledge/knowledge_chunks.json"
OUT_DIR = "data/knowledge_faiss"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

os.makedirs(OUT_DIR, exist_ok=True)


def normalize_text(text: str) -> str:
    return " ".join((text or "").lower().split())


def load_chunks():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return chunks


def dedupe_chunks(chunks):
    unique_chunks = []
    seen = set()

    for c in chunks:
        norm = normalize_text(c.get("text", ""))
        key = (
            c.get("intent", ""),
            c.get("age_group", "general"),
            norm,
        )
        if not norm or key in seen:
            continue
        seen.add(key)
        unique_chunks.append(c)

    return unique_chunks


def main():
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} knowledge chunks")

    chunks = dedupe_chunks(chunks)
    print(f"After exact dedupe: {len(chunks)} knowledge chunks")

    texts = [c["text"] for c in chunks]

    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, os.path.join(OUT_DIR, "knowledge.index"))

    with open(os.path.join(OUT_DIR, "knowledge_meta.pkl"), "wb") as f:
        pickle.dump(chunks, f)

    print("✅ FAISS knowledge index built successfully")


if __name__ == "__main__":
    main()