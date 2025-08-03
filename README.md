📖 About the Project

Document Answerer is an AI-powered document question answering system based on Retrieval-Augmented Generation (RAG).
It enables users to upload documents in .pdf, .docx, or .txt formats, index them using FAISS vector search, and retrieve relevant context for accurate natural language answers powered by Google Gemini AI.

The system provides:
Terminal mode for quick command-line interaction
Streamlit web app for a user-friendly interface
Secure API handling via .env
Automatic re-indexing for newly added files
Multi-format document support and fast semantic search
This makes it ideal for research, report analysis, study material summarization, and knowledge management.

🚀 Features:

📂 Supports multiple formats: PDF, DOCX, TXT

⚡ Fast document search using FAISS vector database

🤖 Accurate answers powered by Gemini AI

🖥 Two interfaces:

Terminal version for quick usage

Streamlit web app for an interactive UI

🔄 Automatic re-indexing when new files are added

🗑 One-click reset to clear all indexed data

🔒 Secure API key storage with .env file

📦 Installation:

1️⃣ Clone the repository
bash
Copy
Edit
git clone https://github.com/YOUR_USERNAME/document-answerer.git
cd document-answerer

2️⃣ Create a virtual environment (recommended)
bash
Copy
Edit
python -m venv rag_env
rag_env\Scripts\activate   # Windows
source rag_env/bin/activate # Mac/Linux

3️⃣ Install dependencies
bash
Copy
Edit
pip install -r requirements.txt

4️⃣ Add your Gemini API key
Create a .env file in the project root:

ini
Copy
Edit
GEMINI_API_KEY=your_api_key_here
🖥 Usage
Option 1: Terminal Version
Place your documents inside the documents folder.

Run:

bash
Copy
Edit
python doc_answer_terminal.py
Type your question:

yaml
Copy
Edit
❓ Question: What is ProRLearn?
💡 Answer: [AI-generated answer]
Type reset to clear all data or exit to quit.

Option 2: Streamlit Web App
Run:

bash
Copy
Edit
streamlit run doc_answer_streamlit.py
Open the provided local URL (usually http://localhost:8501).

Upload your documents via the interface.

Type your question in the text box.

View the AI-generated answer instantly.


⚠️ Notes
Never upload .env to GitHub (contains your API key).
For large documents, the first indexing may take some time.
Make sure you have Python 3.8+ installed.

📜 License
This project is licensed under the MIT License.
