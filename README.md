# 📊 Finance RAG — Hybrid Retrieval-Augmented Generation System

A production-style RAG (Retrieval-Augmented Generation) pipeline built for **financial document Q&A**.
It answers natural language questions by reasoning across structured CSV data (invoices, payments, vendors, expenses) and unstructured policy documents (PDF/TXT), using a hybrid dense + sparse retriever with entity-aware expansion.

---

## 📁 Project Structure

```
RAG/
│
├── data/                          # Your knowledge base (all documents live here)
│   ├── csv/                       # Structured tabular data
│   │   ├── invoices.csv
│   │   ├── payments.csv
│   │   ├── vendors.csv
│   │   └── expenses.csv
│   ├── txt/                       # Policy documents (plain text)
│   │   ├── procurement_policy.txt
│   │   ├── reimbursement_policy.txt
│   │   └── vendor_payment_policy.txt
│   └── pdf/                       # Optional PDF documents
│
├── src/                           # Core source code
│   ├── config.py                  # Central configuration (paths, model names, hyperparameters)
│   ├── data_loader.py             # Loads CSV, TXT, PDF files into LangChain Documents
│   ├── chunking.py                # Splits documents into smaller overlapping chunks
│   ├── embeddings.py              # Loads the HuggingFace sentence-transformer model
│   ├── vector_store.py            # Builds FAISS vector index from chunks
│   ├── bm25_retriever.py          # Keyword-based BM25 sparse retriever
│   ├── hybrid_retriever.py        # Combines dense + sparse + entity expansion
│   └── rag_pipeline.py            # Builds prompt and calls Groq LLM for final answer
│
├── main.py                        # Entry point — orchestrates the full pipeline
├── requirements.txt               # Python dependencies
└── .env                           # API keys (never commit this)
```

---

## ⚙️ Configuration — `src/config.py`

The single source of truth for all tunable parameters.

```python
DATA_PATH     = "./data"           # Root folder scanned recursively for documents
CHUNK_SIZE    = 1000               # Max characters per chunk
CHUNK_OVERLAP = 100                # Characters shared between adjacent chunks
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K         = 10                 # Number of results from dense + sparse retrieval each
BM25_K        = 10
HYBRID_ALPHA  = 0.6                # Reserved for future weighted score fusion
```

**Why these values?**
- `CHUNK_SIZE=1000` — keeps full CSV rows intact. 500 was too small and split rows mid-way, destroying the row's meaning.
- `TOP_K=10` — wider net needed for multi-table questions where relevant data is spread across 3 files.
- `CHUNK_OVERLAP=100` — prevents context loss at chunk boundaries for policy paragraphs.

---

## 📂 Data Loading — `src/data_loader.py`

Converts raw files into **LangChain `Document` objects** — the standard unit throughout the pipeline. Each `Document` has two fields: `page_content` (the text) and `metadata` (source info).

Uses `os.walk()` for **recursive directory scanning**, so CSVs and TXTs inside any subfolder of `./data` are automatically discovered.

### CSV Loading — Two strategies per file

For every CSV file, two types of documents are created:

**1. Row-level documents** (one per row)
```
invoice_id: INV003
vendor_id: V001
amount: 15000
date: 2024-01-15
category: Software
```
> Best for: filtering queries — "show me invoice INV003", "which invoices are from vendor V001"

**2. Table-level document** (one per file, the entire table as a string)
```
Table: invoices.csv
Columns: invoice_id, vendor_id, amount, date, category
Data:
 INV001  V002   4800  2024-01-10    Travel
 INV002  V003   1800  2024-01-12      Food
 ...
```
> Best for: aggregation queries — "total amount of all pending invoices", "how many invoices exist"

### TXT Loading
Reads entire `.txt` files as single documents. Policy files are small enough that one document per file is appropriate.

### PDF Loading
Uses LangChain's `PyPDFLoader` which extracts text page-by-page, creating one document per PDF page.

---

## ✂️ Chunking — `src/chunking.py`

Large documents are split into smaller overlapping chunks using LangChain's `RecursiveCharacterTextSplitter`.

**How it splits (priority order):**
1. First tries to split on paragraph breaks (`\n\n`)
2. Then on line breaks (`\n`)
3. Then on sentences (`. `)
4. Finally on characters as a last resort

**Why chunking matters:**
- Embedding models have a token limit (~512 tokens for MiniLM)
- Smaller chunks = more precise retrieval (the retrieved chunk is more likely to be about exactly what was asked)
- Overlap ensures that a sentence split across a boundary isn't lost

```
[ chunk 1: chars 0–1000   ]
                [chunk 2: chars 900–1900  ]  ← 100 char overlap
                              [chunk 3: chars 1800–2800]
```

---

## 🔢 Embeddings — `src/embeddings.py`

Loads the `sentence-transformers/all-MiniLM-L6-v2` model via LangChain's `HuggingFaceEmbeddings` wrapper.

**What embeddings do:**
Each chunk of text is converted into a 384-dimensional vector (a list of 384 numbers). Semantically similar texts produce vectors that are geometrically close to each other.

```
"invoice from Infosys" → [0.21, -0.83, 0.14, ... ]  (384 numbers)
"bill raised by Infosys" → [0.19, -0.80, 0.17, ... ]  (very close)
"monsoon rainfall data" → [-0.72, 0.34, -0.55, ... ]  (very far)
```

The model is loaded from HuggingFace using your `HF_TOKEN` from `.env`. It runs **locally on your machine** — no API call needed for embeddings.

---

## 🗄️ Vector Store — `src/vector_store.py`

Builds a **FAISS** (Facebook AI Similarity Search) index from all chunk embeddings.

**What FAISS does:**
- Stores all 384-dimensional vectors in an optimized index
- At query time, converts the query to a vector and finds the `k` nearest vectors using L2 (Euclidean) distance
- Extremely fast even with millions of vectors

```
Query: "Which vendor raised INV003?"
         ↓  embed
Query Vector: [0.33, -0.71, ...]
         ↓  FAISS nearest-neighbour search
Top 10 closest chunks retrieved
```

The index is created **once at startup** and lives in memory for the duration of the session.

---

## 🔍 BM25 Retriever — `src/bm25_retriever.py`

A **keyword-based sparse retriever** using the BM25 Okapi algorithm — the same algorithm used by search engines like Elasticsearch.

**How BM25 works:**
- Tokenizes every chunk into individual words at startup
- For a given query, scores every chunk based on:
  - Term frequency (TF) — how often does the query word appear in this chunk?
  - Inverse document frequency (IDF) — how rare is this word across all chunks?
  - Document length normalization — prevents long chunks from winning just because they're long

**Why BM25 alongside embeddings?**
Embeddings excel at semantic similarity but can miss exact keyword matches. BM25 is the opposite — it's terrible at synonyms but perfect for exact IDs.

```
Query: "INV003"
BM25 → directly finds every chunk containing "INV003"  ✅
Embeddings → may retrieve semantically similar invoice chunks but miss INV003 specifically  ⚠️
```

---

## 🔀 Hybrid Retriever — `src/hybrid_retriever.py`

The most important component. Combines dense (FAISS) and sparse (BM25) retrieval, then adds **entity-aware expansion** for multi-hop questions.

### Step 1: Standard Hybrid Retrieval
```
Query → FAISS → Top 10 dense docs
Query → BM25  → Top 10 sparse docs
Combined → deduplicated by page_content → ~15–20 unique chunks
```

### Step 2: Entity ID Extraction
The retriever scans both the **query** and the **already retrieved chunks** for entity IDs using a regex pattern:

```python
ID_PATTERN = re.compile(r'\b(INV\d+|V\d+|PAY\d+|EXP\d+)\b')
```

Example — for the query *"What vendor raised INV003 and has it been paid?"*:
- From query: `{INV003}`
- From retrieved invoices chunk: `{INV003, V001}`
- From retrieved payments chunk: `{PAY003, INV003}`
- **All IDs found: `{INV003, V001, PAY003}`**

### Step 3: Expansion Lookups
For each discovered ID, BM25 fires a targeted lookup:
```
BM25.retrieve("V001", k=3)   → pulls Infosys row from vendors.csv
BM25.retrieve("PAY003", k=3) → pulls Pending status from payments.csv
BM25.retrieve("INV003", k=3) → confirms invoice details
```

### Step 4: Final Merge
All initial + expansion docs are merged and deduplicated. The LLM now has **all related rows from all tables** in its context.

**Without expansion:**
```
Context has: INV003 → V001, amount 15000
Missing: V001 → Infosys (vendors.csv) and PAY003 → Pending (payments.csv)
Result: "Not found in knowledge base" ❌
```

**With expansion:**
```
Context has: INV003 details + V001 → Infosys + PAY003 → Pending
Result: "INV003 was raised by Infosys. Payment is Pending." ✅
```

---

## 🤖 RAG Pipeline — `src/rag_pipeline.py`

Takes the retrieved chunks and calls the **Groq LLM** (Llama 3.3 70B) to generate the final answer.

### Context Assembly
Each chunk is labelled with its source file and type before being sent to the LLM:

```
[1] Source: invoices.csv (table_row)
invoice_id: INV003
vendor_id: V001
amount: 15000
...

---

[2] Source: vendors.csv (table_row)
vendor_id: V001
vendor_name: Infosys
...

---

[3] Source: payments.csv (table_row)
payment_id: PAY003
invoice_id: INV003
status: Pending
```

### Prompt Design
The prompt explicitly instructs the LLM to:
- Join data across tables using matching IDs
- Show step-by-step reasoning before the final answer
- Compute aggregations (sums, counts) from the data in context
- Only use provided context — no hallucination

### LLM — Groq + Llama 3.3 70B
- **Groq** provides ultra-fast inference via their Language Processing Unit (LPU) hardware
- **Llama 3.3 70B** is a powerful open-source model that handles multi-step reasoning well
- `temperature=0` ensures deterministic, factual answers (no creativity)

---

## 🚀 Main Entry Point — `main.py`

Orchestrates the entire pipeline in sequence:

```
1. Load documents (CSV + TXT + PDF)     → data_loader.py
2. Chunk documents                       → chunking.py
3. Load embedding model                  → embeddings.py
4. Build FAISS vector index              → vector_store.py
5. Build BM25 index                      → bm25_retriever.py
6. Create HybridRetriever                → hybrid_retriever.py
7. Start query loop
   └── For each query:
       a. HybridRetriever.retrieve()     → hybrid_retriever.py
       b. generate_answer()              → rag_pipeline.py
       c. Print answer
```

Steps 1–6 run **once at startup**. Step 7 loops until the user types `exit`.

---

## 🔄 End-to-End Workflow

```
User types a question
        │
        ▼
┌─────────────────────┐
│   HybridRetriever   │
│                     │
│  1. FAISS search    │──── Top 10 semantic matches
│  2. BM25 search     │──── Top 10 keyword matches
│  3. Merge + dedup   │
│  4. Extract IDs     │──── INV003, V001, PAY003 ...
│  5. Expand via BM25 │──── Cross-table rows fetched
│  6. Final merge     │──── ~20–30 relevant chunks
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│    RAG Pipeline     │
│                     │
│  1. Label chunks    │──── [1] Source: invoices.csv ...
│  2. Build prompt    │──── Context + rules + question
│  3. Call Groq API   │──── Llama 3.3 70B (temp=0)
│  4. Return answer   │
└────────┬────────────┘
         │
         ▼
   Answer printed
```

---

## 🧠 Why Hybrid Retrieval?

| Method | Strength | Weakness |
|---|---|---|
| **Dense (FAISS)** | Finds semantically related content even with different wording | Can miss exact ID matches like `INV003` |
| **Sparse (BM25)** | Perfect for exact keyword/ID matches | Fails on synonyms, paraphrasing |
| **Hybrid** | Gets the best of both worlds | Slightly more complexity |
| **+ Entity Expansion** | Chains across multiple tables automatically | Required for any multi-hop question |

---

## 🗂️ Data Schema Reference

**invoices.csv** — `invoice_id, vendor_id, amount, date, category`
**payments.csv** — `payment_id, invoice_id, status, paid_date`
**vendors.csv** — `vendor_id, vendor_name, category, location`
**expenses.csv** — `expense_id, employee, category, amount, date`

**Key joins:**
- `invoices.vendor_id` → `vendors.vendor_id` (which company?)
- `payments.invoice_id` → `invoices.invoice_id` (paid or not?)
- `expenses.employee` → `reimbursement_policy.txt` (within limits?)
- `invoices.amount` → `procurement_policy.txt` (needs approval?)

---

## 🔧 Setup & Installation

### 1. Clone and create virtual environment
```bash
git clone <your-repo>
cd RAG
python -m venv myenv
myenv\Scripts\activate        # Windows
# source myenv/bin/activate   # Mac/Linux
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment variables
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_huggingface_token_here
```
Get your Groq API key at: https://console.groq.com
Get your HuggingFace token at: https://huggingface.co/settings/tokens

### 4. Add your data
Place files inside the `./data` folder. Subfolders are supported:
```
data/
├── csv/your_file.csv
├── txt/your_policy.txt
└── pdf/your_document.pdf
```

### 5. Run
```bash
python main.py
```

---

## 📦 Dependencies — `requirements.txt`

| Package | Purpose |
|---|---|
| `langchain` | Core framework for chaining LLM components |
| `langchain-community` | Community integrations (FAISS, PyPDFLoader) |
| `langchain-huggingface` | HuggingFace embeddings wrapper |
| `faiss-cpu` | Fast vector similarity search (CPU version) |
| `sentence-transformers` | Loads the MiniLM embedding model locally |
| `rank-bm25` | BM25 sparse retrieval algorithm |
| `pandas` | CSV reading and DataFrame operations |
| `pypdf` | PDF text extraction |
| `groq` | Groq API client for Llama inference |
| `python-dotenv` | Loads `.env` file into environment variables |
| `tqdm` | Progress bars during model loading |

---

## 🔐 Environment Variables — `.env`

```
GROQ_API_KEY=   # Required — used in rag_pipeline.py to call Llama 3.3
HF_TOKEN=       # Required — used in embeddings.py to download MiniLM model
```

> ⚠️ Never commit `.env` to version control. Add it to `.gitignore`.

---

## 🧪 Sample Questions to Test

### Single-source (baseline)
```
What is the travel reimbursement limit per trip?
What are the standard vendor payment terms?
What is the monthly procurement budget cap?
```

### Multi-hop (cross-table joins)
```
What is the name of the vendor who raised INV003, and has it been paid?
Which invoices from Bangalore-based vendors are still pending?
INV001 was raised by which vendor, and does it qualify for early payment discount?
```

### Aggregation
```
What is the total amount of all pending invoices?
What is the total travel spend across invoices and employee expenses?
Which invoices exceed the director approval threshold?
```

### Policy + Data
```
Priya submitted a travel expense of 5200 INR. Is this within policy limits?
INV003 is 15000 INR for Software. Does it require director approval?
```

---

## 🛠️ Common Issues & Fixes

| Error | Cause | Fix |
|---|---|---|
| `documents loaded: 0` | Files are in subfolders, `os.listdir` doesn't recurse | Use `os.walk()` — already fixed in `data_loader.py` |
| `IndexError: list index out of range` in FAISS | No documents were chunked (0 chunks = empty embedding list) | Ensure documents load correctly first |
| `Not found in knowledge base` on valid questions | Retrieved chunks missing cross-table data | Entity expansion in `hybrid_retriever.py` fixes this |
| `UNEXPECTED: embeddings.position_ids` | Model architecture mismatch warning from HuggingFace | Harmless — safely ignore |
| Slow first startup | MiniLM model downloading from HuggingFace | One-time download, cached locally after first run |