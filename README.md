# 🌍 Multilingual PDF RAG Pipeline

A production-ready **Retrieval-Augmented Generation** system for querying 200+ multilingual PDF documents across any domain.

## Supported Languages
English · French · Hindi · Kannada · Tamil · Telugu · German · Spanish · Portuguese · Italian · Arabic · Chinese · Japanese · Korean · Russian — and 35+ more via the multilingual embedding model.

---

## Architecture

```
pdfs/
  ├─ doc1.pdf (English)
  ├─ doc2.pdf (Hindi)
  └─ doc3.pdf (Kannada)
        │
        ▼
  [pdf_parser.py]          PyMuPDF → text + tables (markdown)
        │                  • Skip scanned-image pages (warning logged)
        │                  • Skip empty pages silently
        │                  • MD5 deduplication
        │
        ▼
  [chunker.py]             Multilingual sentence-aware chunking
        │                  • NLTK sent_tokenize for European langs
        │                  • Punctuation splits for Indic/CJK scripts
        │                  • Sliding-window overlap
        │
        ▼
  [embedder.py]            paraphrase-multilingual-mpnet-base-v2
        │                  • 768-dim, 50+ languages, L2-normalised
        │
        ▼
  [ChromaDB]               Persistent vector store (cosine similarity)
        │
        ▼
  [retriever.py]           Query → embed → top-K chunks (cross-language)
        │                  → Groq LLM (llama-3.1-8b-instant)
        │
        ▼
  [app.py]                 Streamlit chat UI + source citations
```

---

## Quick Start

### 1 — Prerequisites
- Python 3.10+
- A [Groq API key](https://console.groq.com) (free tier available)

### 2 — Clone / download the project
```bash
git clone <your-repo>
cd rag_pipeline
```

### 3 — One-shot setup
```bash
bash setup.sh
```
This will:
- Create a virtual environment
- Install all dependencies (CPU-only PyTorch)
- Download NLTK tokenizer data
- Pre-download the multilingual embedding model (~420 MB)
- Create `.env` from `.env.example`

### 4 — Configure your API key
```bash
# Open .env and set:
GROQ_API_KEY=gsk_your_key_here
```

### 5 — Add your PDFs
```bash
cp /path/to/your/pdfs/*.pdf ./pdfs/
# Or set PDF_FOLDER in .env to point to an existing folder
```

### 6 — Run ingestion
```bash
source venv/bin/activate
python ingest.py
```

Options:
```bash
python ingest.py --pdf-folder /custom/path/to/pdfs
python ingest.py --reset   # wipe index and re-ingest everything
```

Ingestion is **resumable** — re-running only processes new/changed files.

### 7 — Launch the UI
```bash
streamlit run app.py
```
Then open [http://localhost:8501](http://localhost:8501).

---

## Manual Setup (without setup.sh)

```bash
# 1. Create and activate virtualenv
python3 -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# 4. Create directories
mkdir -p pdfs chroma_db

# 5. Configure environment
cp .env.example .env
# Edit .env and set GROQ_API_KEY

# 6. Ingest PDFs
python ingest.py

# 7. Run UI
streamlit run app.py
```

---

## Project Structure

```
rag_pipeline/
├── app.py                # Streamlit UI
├── ingest.py             # Ingestion pipeline (run once)
├── retriever.py          # Query + generation pipeline
├── config.py             # All configuration constants
├── utils/
│   ├── __init__.py
│   ├── pdf_parser.py     # PyMuPDF-based PDF → text extractor
│   ├── chunker.py        # Multilingual sentence-aware chunker
│   └── embedder.py       # SentenceTransformer wrapper + ChromaDB EF
├── requirements.txt
├── .env.example
├── setup.sh
└── README.md

# Created at runtime:
├── .env                  # Your secrets (not committed)
├── pdfs/                 # Input PDF files
├── chroma_db/            # Persistent ChromaDB vector store
├── ingested_hashes.json  # MD5 hashes of processed files (dedup)
└── ingest.log            # Detailed ingestion log
```

---

## Edge Cases Handled

| Scenario | Handling |
|---|---|
| Scanned PDF (no extractable text) | Detected by low char-count + image presence; skipped with `WARNING` log |
| Duplicate PDFs | MD5 hash check before parsing; skipped instantly |
| Empty pages | Silently skipped inside the parser |
| 100+ page PDFs | Processed page-by-page (no memory accumulation) |
| Very long sentences | Hard character-window fallback with overlap |
| Kannada / Hindi text | Punctuation-based sentence splitting (।, ?, !, .) |
| Ingestion interrupted | Hashes saved after each file; re-run resumes from where it stopped |
| Tables | Extracted as Markdown tables and embedded in chunk text |
| Cross-language queries | Multilingual embedding maps all languages to shared space |

---

## Configuration Reference (`.env`)

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | Your Groq API key |
| `PDF_FOLDER` | `./pdfs` | Folder containing PDF files |
| `CHROMA_DB_PATH` | `./chroma_db` | ChromaDB persistence directory |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model ID |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-mpnet-base-v2` | HuggingFace model ID |
| `TOP_K` | `5` | Number of chunks to retrieve |
| `INGEST_LOG` | `./ingest.log` | Ingestion log file path |

### Groq model options
- `llama-3.1-8b-instant` — fastest, great for most queries
- `llama3-70b-8192` — highest quality, slower
- `mixtral-8x7b-32768` — large context window (32K tokens)

---

## Troubleshooting

**`RuntimeError: Vector store 'multilingual_rag' not found`**
→ Run `python ingest.py` first.

**`GROQ_API_KEY is not set`**
→ Copy `.env.example` to `.env` and add your key.

**Ingestion is slow**
→ Normal for the first run (model download ~420 MB + encoding). Subsequent runs are faster (cached model, skips already-ingested files).

**Kannada/Hindi answers are in English**
→ The LLM is prompted to respond in the query language. Very short queries may be mis-detected — try a longer question.

**`CUDA out of memory`**
→ The CPU-only torch build is specified in `requirements.txt`. If you installed the GPU version separately, force CPU: `CUDA_VISIBLE_DEVICES="" python ingest.py`
