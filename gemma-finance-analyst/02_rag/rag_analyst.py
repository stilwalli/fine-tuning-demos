import chromadb
import ollama
from sentence_transformers import SentenceTransformer

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to ChromaDB
client = chromadb.PersistentClient(path="02_rag/db")
collection = client.get_collection("earnings")

# Question
question = "What happened to the share buyback program?"

# Convert question to embedding and search
question_embedding = embedder.encode(question).tolist()
results = collection.query(
    query_embeddings=[question_embedding],
    n_results=3
)

# Build context from retrieved transcripts
context = "\n\n---\n\n".join(results["documents"][0])

# Call Gemma with context
response = ollama.chat(
    model="gemma2:9b",
    messages=[
        {
            "role": "system",
            "content": "You are a senior financial analyst. Answer questions based only on the provided context."
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }
    ]
)

print(response["message"]["content"])