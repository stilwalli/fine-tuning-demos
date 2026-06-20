import chromadb
from sentence_transformers import SentenceTransformer
import os

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Set up ChromaDB
client = chromadb.PersistentClient(path="02_rag/db")
collection = client.get_or_create_collection("earnings")

# Load and store each transcript
for filename in os.listdir("data"):
    if filename.endswith(".txt"):
        with open(f"data/{filename}", "r") as f:
            text = f.read()
        
        embedding = embedder.encode(text).tolist()
        
        collection.add(
            documents=[text],
            embeddings=[embedding],
            ids=[filename]
        )
        print(f"Added {filename}")

print("Done!")  