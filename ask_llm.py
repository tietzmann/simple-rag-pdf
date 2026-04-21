import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import sys
from groq import Groq

# -----------------------------
# 1. Load Groq API key
# -----------------------------
with open("groq.txt", "r") as f:
    GROQ_API_KEY = f.read().strip()

client = Groq(api_key=GROQ_API_KEY)

# -----------------------------
# 2. Load FAISS index
# -----------------------------
index = faiss.read_index("faiss_index.bin")

# -----------------------------
# 3. Load chunks
# -----------------------------
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

print(f"Loaded {len(chunks)} chunks")

# -----------------------------
# 4. Load embedding model
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# 5. Ask question
# -----------------------------
query = input("Ask a question: ").strip()

if not query:
    print("❌ Please enter a valid question.")
    sys.exit()

# -----------------------------
# 6. Retrieve relevant chunks
# -----------------------------
query_embedding = model.encode([query], normalize_embeddings=True)

k = 5
D, I = index.search(np.array(query_embedding, dtype=np.float32), k)

retrieved_chunks = [chunks[i] for i in I[0]]

context = "\n\n".join(retrieved_chunks)

# -----------------------------
# 7. Send to Groq
# -----------------------------
prompt = f"""
You are a helpful assistant answering questions based on a book.

Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {query}

Give a detailed, well-explained answer with examples if possible.
"""

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "user", "content": prompt}
    ],
    temperature=0.2
)

# -----------------------------
# 8. Output answer
# -----------------------------
answer = response.choices[0].message.content

print("\n🤖 Answer:\n")
print(answer)