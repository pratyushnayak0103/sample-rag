# sample-rag

# Retrieval-Augmented Generation (RAG) Architecture

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline that enables a chatbot to answer questions using your own documents as context.

---

## 🔄 Architecture Flow

```mermaid
flowchart TD
    A[💬 User Query<br/>("Where is the Eiffel Tower?")] --> B[🧠 Query Embedding<br/>(SentenceTransformer)]
    B --> C[📚 Vector Store (FAISS)<br/>Search Top-K Chunks]
    C --> D[📑 Retrieved Chunks<br/>("The Eiffel Tower is in Paris.")]
    D --> E[📝 Prompt Builder<br/>Context + Question]
    E --> F[🤖 LLM Generation<br/>(GPT-3.5/4 or local model)]
    F --> G[✅ Chatbot Response<br/>"The Eiffel Tower is located in Paris, France."]

    subgraph Indexing Phase (One-time)
        H[📂 Document Loader<br/>PDF/Text/Web] --> I[✂️ Chunking<br/>Split Documents]
        I --> J[🧠 Document Embedding<br/>(SentenceTransformer)]
        J --> C
    end
