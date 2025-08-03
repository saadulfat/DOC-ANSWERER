import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import docx
import PyPDF2
from dotenv import load_dotenv

# ==============================
# LOAD GEMINI API KEY FROM .env
# ==============================
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("⚠️ No GEMINI_API_KEY found in .env file!")

import google.generativeai as genai
genai.configure(api_key=GEMINI_API_KEY)

# ==============================
# CONFIGURATION
# ==============================
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNKS_FILE = "chunks.pkl"
INDEX_FILE = "faiss_index.idx"
FILES_TRACKER = "indexed_files.pkl"
DOCUMENTS_FOLDER = "documents"

print("📌 Loading embedding model...")
embedder = SentenceTransformer(EMBEDDING_MODEL)

# ==============================
# RESET DATA
# ==============================
def reset_data():
    """Deletes index, chunks, tracker, and clears documents folder."""
    for file in [INDEX_FILE, CHUNKS_FILE, FILES_TRACKER]:
        if os.path.exists(file):
            os.remove(file)
    if os.path.exists(DOCUMENTS_FOLDER):
        for f in os.listdir(DOCUMENTS_FOLDER):
            os.remove(os.path.join(DOCUMENTS_FOLDER, f))
    print("✅ All data reset! You can now start fresh.")

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
# DOCUMENT LOADING
# ==============================
def load_documents(folder_path):
    print(f"📂 Checking documents in {folder_path}...")
    all_files = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if file.lower().endswith(".txt"):
            all_files.append((file, read_txt(file_path)))
        elif file.lower().endswith(".pdf"):
            all_files.append((file, read_pdf(file_path)))
        elif file.lower().endswith(".docx"):
            all_files.append((file, read_docx(file_path)))
    print(f"📄 Found {len(all_files)} documents.")
    return all_files

# ==============================
# TEXT CHUNKING
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
    if not new_chunks:
        return index
    embeddings = embedder.encode(new_chunks, convert_to_numpy=True)
    index.add(embeddings)
    return index

# ==============================
# SAVE / LOAD FUNCTIONS
# ==============================
def save_data(index, chunks, indexed_files):
    faiss.write_index(index, INDEX_FILE)
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)
    with open(FILES_TRACKER, "wb") as f:
        pickle.dump(indexed_files, f)
    print("✅ Index, chunks, and file tracker saved.")

def load_data():
    if not os.path.exists(INDEX_FILE) or not os.path.exists(CHUNKS_FILE) or not os.path.exists(FILES_TRACKER):
        return None, [], set()
    index = faiss.read_index(INDEX_FILE)
    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)
    with open(FILES_TRACKER, "rb") as f:
        indexed_files = pickle.load(f)
    print("✅ Existing index and tracker loaded.")
    return index, chunks, indexed_files

# ==============================
# GEMINI QA
# ==============================
def retrieve_relevant_chunks(query, index, chunks, top_k=3):
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]

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
- Keep sentences short and break lines every 12–15 words 
  for better readability.
- If explanation is long, split into paragraphs.

QUESTION:
{question}

Now give the answer in a well-formatted, human-friendly style:
"""

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)

    print("📌 Type 'reset' to clear all indexed data and start fresh.")
    print("📌 Type 'exit' to quit.\n")

    index, all_chunks, indexed_files = load_data()

    # If no index yet, create it
    if index is None:
        print("📌 No existing index found. Creating new one...")
        index = faiss.IndexFlatL2(embedder.get_sentence_embedding_dimension())
        indexed_files = set()

    # Load current documents
    documents = load_documents(DOCUMENTS_FOLDER)

    new_chunks = []
    for file_name, content in documents:
        if file_name not in indexed_files and content.strip():
            print(f"➕ Indexing new file: {file_name}")
            chunks = chunk_text(content)
            new_chunks.extend(chunks)
            all_chunks.extend(chunks)
            indexed_files.add(file_name)

    # If new files found, update FAISS
    if new_chunks:
        index = add_to_faiss_index(index, new_chunks)
        save_data(index, all_chunks, indexed_files)
    else:
        print("✅ No new files to index.")

    # Start QA loop
    while True:
        user_input = input("\n❓ Question: ").strip()
        if user_input.lower() == "exit":
            break
        elif user_input.lower() == "reset":
            reset_data()
            break
        elif not user_input:
            continue
        else:
            answer = answer_question_gemini(user_input, index, all_chunks)
            print(f"\n💡 Answer: {answer}\n")
