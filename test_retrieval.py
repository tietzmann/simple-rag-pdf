import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# -----------------------------
# 1. Load FAISS index
# -----------------------------
index = faiss.read_index("faiss_index.bin")

# -----------------------------
# 2. Load chunks
# -----------------------------
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

print(f"Loaded {len(chunks)} chunks")

# -----------------------------
# 3. Load embedding model
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# 4. Ask a question
# -----------------------------
query = input("Ask a question: ").strip()

if not query:
    print("❌ Please enter a valid question.")
    exit()

# Convert query to embedding
query_embedding = model.encode([query], normalize_embeddings=True)

# -----------------------------
# 5. Search in FAISS
# -----------------------------
k = 5  # number of results
D, I = index.search(np.array(query_embedding, dtype=np.float32), k)

# -----------------------------
# 6. Show results
# -----------------------------
print("\n🔎 Top results:\n")
    
for i, idx in enumerate(I[0]):
    if idx == -1:
        continue
    print(f"Score: {D[0][i]:.4f}")
    print(chunks[idx])
    print("-" * 50)
