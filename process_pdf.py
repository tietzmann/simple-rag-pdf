import pdfplumber
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
import nltk

# -----------------------------
# 1. Load PDF
# -----------------------------
def load_pdf(pdf_path):
    text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text.append(t)
    return "\n".join(text)


# -----------------------------
# 2. Split into chunks
# -----------------------------
def split_text(text, chunk_size=800, overlap=150):
    sentences = nltk.sent_tokenize(text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # if adding sentence keeps chunk under limit
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())

            # create overlap (keep last part of previous chunk)
            current_chunk = current_chunk[-overlap:] + " " + sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# -----------------------------
# 3. Create embeddings
# -----------------------------
def create_embeddings(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, normalize_embeddings=True, show_progress_bar=True)
    return embeddings


# -----------------------------
# 4. Store in FAISS
# -----------------------------
def build_faiss(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(np.array(embeddings, dtype=np.float32))
    return index


# -----------------------------
# 5. Save index + chunks
# -----------------------------
def save_index(index, chunks):
    faiss.write_index(index, "faiss_index.bin")
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    pdf_path = "Trading-Volatility-Colin-Bennet.pdf"

    print("📖 Reading PDF...")
    text = load_pdf(pdf_path)

    print("✂️ Splitting into chunks...")
    chunks = split_text(text)
    print(f"Total chunks: {len(chunks)}")

    print("🧠 Creating embeddings...")
    embeddings = create_embeddings(chunks)

    print("📦 Building FAISS index...")
    index = build_faiss(embeddings)

    print("💾 Saving index...")
    save_index(index, chunks)

    print("✅ Done! Index saved as faiss_index.bin")