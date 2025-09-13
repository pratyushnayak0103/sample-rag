from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load documents
print("Index started")
with open('documents.txt', 'r') as f:
    documents = [line.strip() for line in f.readlines() if line.strip()]

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Embed documents
doc_embeddings = model.encode(documents, convert_to_numpy=True)

# Create FAISS index (vector search index)
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)

# Save index and documents for retrieval
faiss.write_index(index, 'faiss_index.bin')
import pickle
with open('documents.pkl', 'wb') as f:
    pickle.dump(documents, f)

print("Index created and saved!")
