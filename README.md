✅ What is RAG?
Retrieval-Augmented Generation (RAG) is a method that improves LLMs by letting them retrieve relevant documents from a knowledge base before generating a response.

It combines:

Dense Retrieval (e.g., FAISS): to find relevant documents

Generative Models (e.g., OpenAI GPT, HuggingFace models): to generate answers based on retrieved documents

🔧 Setup & Step-by-Step Guide
✅ Step 0: Install Required Packages
pip install transformers faiss-cpu sentence-transformers 

✅ Step 1: Prepare Knowledge Base (Documents)
Let's simulate a small knowledge base.

            documents = [
                "The Eiffel Tower is located in Paris.",
                "The Great Wall of China is visible from space.",
                "The capital of France is Paris.",
                "Python is a popular programming language for AI.",
                "Transformers are deep learning models used in NLP."
]
✅ Step 2: Create a python file create_index.py which will be used for embedding the document knowledge source to vector database.

✅ Step 3: Create a chatbot.py pipeline which act as prompt receiver and provide output to user.
