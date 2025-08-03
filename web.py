import os
import pickle
import faiss
import streamlit as st
import docx
import PyPDF2
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# ==============================
# LOAD GEMINI API KEY FROM .env
# ==============================
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("⚠️ No GEMINI_API_KEY found in .env file!")

genai.configure(api_key=GEMINI_API_KEY)

# ==============================
# CONFIG
# ==============================
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNKS_FILE = "chunks.pkl"
INDEX_FILE = "faiss_index.idx"
FILES_TRACKER = "indexed_files.pkl"
DOCUMENTS_FOLDER = "documents"

os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)

# ==============================
# LOAD EMBEDDING MODEL
# ==============================
@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBEDDING_MODEL)

embedder = load_embedder()

# ==============================
# RESET DATA FUNCTION
# ==============================
def reset_data():
    for file in [INDEX_FILE, CHUNKS_FILE, FILES_TRACKER]:
        if os.path.exists(file):
            os.remove(file)
    for f in os.listdir(DOCUMENTS_FOLDER):
        os.remove(os.path.join(DOCUMENTS_FOLDER, f))
    st.success("✅ All data has been reset! Start fresh by uploading new files.")

# ==============================
# FILE READING FUNCTIONS
# ==============================
def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_pdf(file_path):
    text = ""
    with open(file_path, "rb") as f:
        pdf_reader = PyPDF2.PdfReader(f)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def read_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

# ==============================
# CHUNKING
# ==============================
def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks

# ==============================
# FAISS FUNCTIONS
# ==============================
def create_faiss_index(chunks):
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def add_to_faiss_index(index, new_chunks):
    embeddings = embedder.encode(new_chunks, convert_to_numpy=True)
    index.add(embeddings)
    return index

# ==============================
# SAVE/LOAD INDEX
# ==============================
def save_data(index, chunks, indexed_files):
    faiss.write_index(index, INDEX_FILE)
    pickle.dump(chunks, open(CHUNKS_FILE, "wb"))
    pickle.dump(indexed_files, open(FILES_TRACKER, "wb"))

def load_data():
    if not os.path.exists(INDEX_FILE) or not os.path.exists(CHUNKS_FILE) or not os.path.exists(FILES_TRACKER):
        return None, [], set()
    index = faiss.read_index(INDEX_FILE)
    chunks = pickle.load(open(CHUNKS_FILE, "rb"))
    indexed_files = pickle.load(open(FILES_TRACKER, "rb"))
    return index, chunks, indexed_files

# ==============================
# RETRIEVAL
# ==============================
def retrieve_relevant_chunks(query, index, chunks, top_k=3):
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]

# ==============================
# GEMINI QA
# ==============================
def answer_question_gemini(question, index, chunks):
    relevant_chunks = retrieve_relevant_chunks(question, index, chunks)
    context = " ".join(relevant_chunks)

    prompt = f"""
    You are an advanced AI assistant specialized in Retrieval-Augmented Generation (RAG).

    You are given the following CONTEXT extracted from documents:
    ---
    {context}
    ---

    Answer the QUESTION based on the context above.
    If the context does not have the answer, clearly state: 
    ⚠️ I could not find this information in the provided documents.

    When answering:
    - Be accurate and use only relevant context.
    - Provide a clear, structured answer.
    - Bold the main headings (no asterisks before/after the heading name).
    - Use bullet points or numbered lists for details.
    - Keep sentences short and break lines every 12–15 words for better readability.
    - If explanation is long, split into paragraphs.

    QUESTION:
    {question}

    Now give the answer in a well-formatted, human-friendly style:
    """

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text

# ==============================
# STREAMLIT UI
# ==============================
st.set_page_config(page_title="📄 Document Answerer", layout="wide")
st.title("📄 Document Answerer")
st.write("Upload documents, then ask questions.")

# Sidebar reset button
if st.sidebar.button("🔄 Reset Data"):
    reset_data()
    st.stop()

# File uploader
uploaded_files = st.file_uploader(
    "📂 Upload your documents (.pdf, .docx, .txt)", 
    type=["pdf", "docx", "txt"], 
    accept_multiple_files=True
)

# Load existing index or create new
index, all_chunks, indexed_files = load_data()
if index is None:
    index = faiss.IndexFlatL2(embedder.get_sentence_embedding_dimension())
    all_chunks = []
    indexed_files = set()

# Handle uploads
if uploaded_files:
    new_chunks = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(DOCUMENTS_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        if uploaded_file.name not in indexed_files:
            if uploaded_file.name.lower().endswith(".txt"):
                content = read_txt(file_path)
            elif uploaded_file.name.lower().endswith(".pdf"):
                content = read_pdf(file_path)
            elif uploaded_file.name.lower().endswith(".docx"):
                content = read_docx(file_path)
            else:
                continue

            chunks = chunk_text(content)
            new_chunks.extend(chunks)
            all_chunks.extend(chunks)
            indexed_files.add(uploaded_file.name)

    if new_chunks:
        index = add_to_faiss_index(index, new_chunks)
        save_data(index, all_chunks, indexed_files)
        st.success(f"Indexed {len(new_chunks)} new chunks from {len(uploaded_files)} files.")

# Question input
question = st.text_input("❓ Ask a question about your documents:")
if question:
    if not all_chunks:
        st.error("Please upload and index documents first.")
    else:
        answer = answer_question_gemini(question, index, all_chunks)
        st.markdown(f"### 💡 Answer:\n{answer}")
