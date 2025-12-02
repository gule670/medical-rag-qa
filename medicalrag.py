# app.py
import streamlit as st
import json
import faiss
import numpy as np

BASE_PATH = os.path.dirname(__file__)
@st.cache_resource
def load_faiss():
    index_path = os.path.join(BASE_PATH, "medical_faiss.index")
    return faiss.read_index(index_path)

@st.cache_resource
def load_chunks():
    chunks_path = os.path.join(BASE_PATH, "processed_medical_chunks.json")
    with open(chunks_path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource
def load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

chunks = load_chunks()
index = load_faiss()
model = load_model()


from groq import Groq
import streamlit as st
client = Groq(api_key=st.secrets["GROQ_API_KEY"])
def ask_llm(question,context):
    prompt = f"""
    Answer the question based on the context below. Only include information directly relevant.

    Question: {question}

    Context:
    {" ".join(context)}

    Answer:
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="Medical RAG Assistant",
    page_icon="🩺",
    layout="centered"
)

# ---- CUSTOM CSS FOR COVER PAGE ----
cover_css = """
<style>
body {
    background: linear-gradient(135deg, #eef2f3 0%, #cfd9df 100%);
}
.cover-card {
    background: white;
    padding: 3rem;
    border-radius: 20px;
    box-shadow: 0px 5px 20px rgba(0,0,0,0.15);
    text-align: center;
}
.title {
    font-size: 3rem;
    font-weight: 800;
    color: #1A5678;
}
.subtitle {
    font-size: 1.2rem;
    color: #444;
    margin-top: 0.7rem;
    margin-bottom: 2rem;
}
.logo {
    width: 140px;
    margin-bottom: 1.5rem;
}
.footer-text {
    margin-top: 2rem;
    font-size: 0.9rem;
    color: #777;
}
button[kind="primary"] {
    background: #1A5678 !important;
    border-radius: 10px !important;
}
</style>
"""

st.markdown(cover_css, unsafe_allow_html=True)

# ---- COVER CARD ----
with st.container():
    #st.markdown('<div class="cover-card">', unsafe_allow_html=True)
    st.markdown('<div class="title">Medical RAG Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">AI-powered Retrieval-Augmented Medical Knowledge Explorer</div>', unsafe_allow_html=True)

    st.write(
        """
        This system uses **Retrieval-Augmented Generation (RAG)** to provide  
        grounded, citation-aware answers from a curated dataset of **medical transcriptions**.

        It performs:
        - Semantic retrieval using embeddings + FAISS  
        - Context-aware medical reasoning with Gorq  
        - Safe, grounded clinical summarization  
        """
    )

    st.write("\n")


    st.markdown('</div>', unsafe_allow_html=True)

# ---- REDIRECT TO MAIN RAG UI ----
if "start" in st.session_state:
    st.switch_page("rag_app.py")   # create this file for the RAG interface


user_input = st.text_area("Please enter your query:")
if st.button("Enter"):
    if not user_input.strip():
        st.warning("⚠️ Please enter a sentence.")
    else:
        q_emb = model.encode([user_input])

        D, I = index.search(q_emb, k=3)
        retrieved_texts = [chunks[idx]["chunk_text"][:500] for idx in I[0]]
        st.write(ask_llm(user_input,retrieved_texts))

