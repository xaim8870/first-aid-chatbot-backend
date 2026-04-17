import os, re, fitz
import pytesseract
from PIL import Image
from sentence_transformers import SentenceTransformer
import faiss, numpy as np, pickle

# Path to tesseract.exe (Windows only)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

PDF_PATH = "data/urdu/dataset.pdf"
INDEX_DIR = "data/indexes"
os.makedirs(INDEX_DIR, exist_ok=True)

INDEX_PATH = os.path.join(INDEX_DIR, "urdu_faiss.index")
TEXTS_PATH = os.path.join(INDEX_DIR, "urdu_texts.pkl")

print("📖 Extracting Urdu text from PDF...")
doc = fitz.open(PDF_PATH)
text = ""

for page_num, page in enumerate(doc, start=1):
    page_text = page.get_text("text")
    if page_text.strip():
        text += page_text
    else:
        # If no text, run OCR on image
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        page_text = pytesseract.image_to_string(img, lang="urd")  # use Urdu OCR
        text += page_text

# Clean
text = re.sub(r"\s+", " ", text)
print(f"✅ Extracted {len(text)} characters of Urdu text.")

# Chunking
def chunk_text(text, max_chars=300):
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

urdu_chunks = chunk_text(text, max_chars=300)
print(f"✂️ Split into {len(urdu_chunks)} chunks.")

# Embeddings
if len(urdu_chunks) > 0:
    print("🔍 Generating embeddings...")
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    urdu_embeddings = model.encode(urdu_chunks, convert_to_numpy=True, show_progress_bar=True)

    # FAISS Index
    print("⚡ Building FAISS index...")
    dimension = urdu_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(urdu_embeddings)

    faiss.write_index(index, INDEX_PATH)
    with open(TEXTS_PATH, "wb") as f:
        pickle.dump(urdu_chunks, f)

    print(f"✅ Urdu FAISS index saved to {INDEX_PATH}")
    print(f"✅ Urdu chunks saved to {TEXTS_PATH}")
else:
    print("❌ No Urdu text found in the PDF, even after OCR.")
