from typing import List
import os
from PyPDF2 import PdfReader
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from tqdm import tqdm

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def extract_text_from_pdf(path: str) -> str:
    try:
        reader = PdfReader(path)
    except Exception:
        return ""
    texts = []
    for p in reader.pages:
        try:
            t = p.extract_text()
            if t:
                texts.append(t)
        except Exception:
            continue
    return "\n".join(texts)

def simple_text_split(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    if not text:
        return []
    words = text.split()
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

class ChromaIndexer:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)
        self.client = chromadb.Client(Settings(persist_directory=self.persist_directory))
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL_NAME)
        try:
            self.collection = self.client.get_collection(name="documents")
        except Exception:
            self.collection = self.client.create_collection(name="documents", embedding_function=self.embedding_fn)

    def add_pdf(self, pdf_path: str, doc_id_prefix: str = None):
        text = extract_text_from_pdf(pdf_path)
        chunks = simple_text_split(text)
        ids, metadatas, docs = [], [], []
        for i, c in enumerate(tqdm(chunks, desc=f"Indexing {os.path.basename(pdf_path)}")):
            did = f"{os.path.basename(pdf_path)}_{i}" if doc_id_prefix is None else f"{doc_id_prefix}_{i}"
            ids.append(did)
            metadatas.append({"source": os.path.basename(pdf_path), "chunk_index": i})
            docs.append(c)
        if docs:
            self.collection.add(ids=ids, metadatas=metadatas, documents=docs)
        return len(docs)

    def query(self, query_text: str, n_results: int = 4):
        return self.collection.query(query_texts=[query_text], n_results=n_results)
