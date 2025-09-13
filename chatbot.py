from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import torch

# Load documents and FAISS index
with open('documents.pkl', 'rb') as f:
    documents = pickle.load(f)

index = faiss.read_index('faiss_index.bin')

# Load embedding model for query
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Load a small seq2seq model for generation (local)
model_name = 't5-small'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

def retrieve(query, top_k=2):
    query_vec = embedder.encode([query])
    D, I = index.search(query_vec, top_k)
    return [documents[i] for i in I[0]]

def generate_answer(query, retrieved_docs):
    context = " ".join(retrieved_docs)
    input_text = f"question: {query} context: {context}"
    inputs = tokenizer(input_text, return_tensors='pt', truncation=True).to(device)
    outputs = model.generate(**inputs, max_length=100)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

if __name__ == "__main__":
    print("Welcome to RAG Chatbot (local)!")
    while True:
        query = input("\nYour question: ")
        if query.lower() in ['exit', 'quit']:
            break
        retrieved_docs = retrieve(query)
        print("\nRetrieved docs:")
        for d in retrieved_docs:
            print("-", d)
        answer = generate_answer(query, retrieved_docs)
        print("\nAnswer:", answer)
