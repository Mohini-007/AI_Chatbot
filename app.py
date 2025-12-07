import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from utils import ChromaIndexer
from rag_chain import build_qa_chain

# Load env
load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_KEY:
    st.warning("Set OPENAI_API_KEY in your environment (.env). Some features will not work without it.")

# FIX: Inject the key explicitly
client = OpenAI(api_key=OPENAI_KEY)


# Config
PERSIST_DIR = "./chroma_db"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
USE_GPU = False

st.set_page_config(page_title="RAG Voice Chatbot", layout="wide")
st.markdown("<h1>🎧 RAG Chatbot — Streamlit + Whisper API + TTS</h1>", unsafe_allow_html=True)

# Init indexer & vector store (cached)
@st.cache_resource
def init_vectorstore():
    indexer = ChromaIndexer(persist_directory=PERSIST_DIR)
    emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    vector_store = Chroma(persist_directory=PERSIST_DIR, embedding_function=emb)
    return indexer, vector_store

indexer, vector_store = init_vectorstore()
retriever = vector_store.as_retriever(search_kwargs={"k": 4})

# Build QA chain once (cached)
@st.cache_resource
def get_qa_chain():
    return build_qa_chain(retriever, use_gpu=USE_GPU)

qa_chain = get_qa_chain()

# Session history
if "history" not in st.session_state:
    st.session_state.history = []  # each item: {"role":"user"/"bot", "text": "..."} 

def add_history(role, text):
    st.session_state.history.append({"role": role, "text": text})

# --- Whisper API transcription (server-side)
def transcribe_audio_api(path: str) -> str:
    try:
        with open(path, "rb") as f:
            resp = client.audio.transcriptions.create(model="whisper-1", file=f)
        return resp.text
    except Exception as e:
        return f"Transcription failed: {e}"

# --- OpenAI TTS (returns local file path)
def text_to_speech(text: str) -> str:
    try:
        # model name may vary; adjust if your API requires different call
        resp = client.audio.speech.create(model="gpt-4o-mini-tts", voice="alloy", input=text)
        out = "tts_output.mp3"
        resp.stream_to_file(out)
        return out
    except Exception as e:
        st.error(f"TTS failed: {e}")
        return None

# --- RAG query helper
def answer_query(question: str) -> str:
    try:
        result = qa_chain({"query": question})
        answer = result.get("result") or str(result)
        srcs = result.get("source_documents", [])
    except Exception as e:
        return f"Error from QA chain: {e}"
    sources = [f"[{d.metadata.get('source')}] chunk:{d.metadata.get('chunk_index')}" for d in srcs]
    src_text = ("\n\nSources: " + ", ".join(sources)) if sources else ""
    return answer + src_text

# --- UI layout
with st.sidebar:
    st.header("📄 Document Indexing")
    uploaded = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if st.button("Index uploaded PDFs"):
        if not uploaded:
            st.warning("Please select PDF(s) to index.")
        else:
            total = 0
            with st.spinner("Indexing..."):
                for f in uploaded:
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                    tmp.write(f.read())
                    tmp.flush()
                    total += indexer.add_pdf(tmp.name)
                    tmp.close()
            st.success(f"Indexed {total} chunks.")

    st.markdown("---")
    st.header("⚙️ Settings")
    st.write(f"Vector DB: {PERSIST_DIR}")
    st.write(f"Embedding model: {EMBED_MODEL_NAME}")

st.subheader("💬 Ask (text or upload voice)")

col1, col2 = st.columns([3,1])
with col1:
    user_text = st.text_input("Type your question here:", value="")
    audio_upload = st.file_uploader("Or upload voice (wav/mp3):", type=["wav", "mp3"])

    if st.button("Send"):
        final_q = None
        if user_text and user_text.strip():
            final_q = user_text.strip()
            add_history("user", final_q)
        elif audio_upload is not None:
            tmp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tmp_audio.write(audio_upload.read())
            tmp_audio.flush()
            tmp_audio.close()
            st.info("Transcribing audio...")
            trans = transcribe_audio_api(tmp_audio.name)
            add_history("user", f"🎤 {trans}")
            final_q = trans
        else:
            st.warning("Please type a question or upload an audio file.")
            final_q = None

        if final_q:
            with st.spinner("Querying knowledge base..."):
                out = answer_query(final_q)
                add_history("bot", out)
                st.markdown("**Answer:**")
                st.write(out)
                # TTS
                tts_path = text_to_speech(out)
                if tts_path:
                    st.audio(tts_path)

with col2:
    st.markdown("### Conversation History")
    for entry in reversed(st.session_state.history[-40:]):  # show last 40 messages
        role = entry["role"]
        text = entry["text"]
        if role == "user":
            st.markdown(f"<div style='background:#4a61dd;padding:8px;border-radius:8px;color:white;margin:4px'>{text}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background:#2d333b;padding:8px;border-radius:8px;color:white;margin:4px'>{text}</div>", unsafe_allow_html=True)

st.markdown("---")
st.caption("Powered by OpenAI (Whisper + GPT) and ChromaDB")
