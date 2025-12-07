# RAG Chatbot (Streamlit + LangChain + ChromaDB)

Minimal, plug-and-play retrieval-augmented chatbot that:
- Loads PDFs, indexes them with Chroma (sentence-transformer embeddings)
- Uses a small open-source LLM (default: google/flan-t5-small)
- Provides a Streamlit chat UI with file upload and source citations
- Includes a FastAPI inference endpoint (optional)

Quick start:

1. Create venv and install:

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

2. Run the app:

streamlit run app.py

Notes:
- Chroma persists to ./chroma_db
