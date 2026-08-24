# 📄 DocuMind

DocuMind is an AI-powered document summary and question-answering assistant. It supports PDF and image uploads, OCR, intelligent summarization, key-point extraction, semantic search, and RAG-based question answering with relevant page citations. The application uses Streamlit, Ollama, Llama 3.2, embeddings, and FAISS.

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Kusuma-Sure/documind-document-summary-assistant.git
cd documind-document-summary-assistant
```

### 2. Create and activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and start Ollama

Install Ollama from:

https://ollama.com/

Then download the required model:

```bash
ollama pull llama3.2
```

### 5. Run DocuMind

```bash
streamlit run app.py
```

Open the local URL shown by Streamlit in your browser.
