import faiss
import pickle
import os

BASE_DIR = "data/knowledge_faiss"

INDEX_PATH = os.path.join(BASE_DIR, "knowledge.index")
META_PATH = os.path.join(BASE_DIR, "knowledge_meta.pkl")

# ---------------- LOAD FAISS INDEX ----------------
index = faiss.read_index(INDEX_PATH)

# ---------------- LOAD METADATA -------------------
with open(META_PATH, "rb") as f:
    chunks = pickle.load(f)

# ---------------- VERIFICATION --------------------
print("🔍 Verifying FAISS ↔ metadata alignment...")

print("FAISS vectors:", index.ntotal)
print("Metadata chunks:", len(chunks))

assert len(chunks) == index.ntotal, (
    f"❌ Mismatch: {len(chunks)} metadata chunks vs "
    f"{index.ntotal} FAISS vectors"
)

print("✅ Alignment verified: FAISS and metadata are in sync")
