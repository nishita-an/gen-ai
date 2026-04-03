# Persistent Memory System for Long-Context LLM Study Assistant

A production-grade context management system that enables an LLM-based chatbot to maintain coherent memory across **50+ conversation turns** without exceeding the model's context window.

---

## Architecture Overview

```
User Message
     │
     ▼
┌────────────────────────────────────────────────────────┐
│                    MemoryAgent                         │
│  ┌──────────────┐  ┌────────────────┐  ┌───────────┐  │
│  │ ShortTerm    │  │   Episodic     │  │ Semantic  │  │
│  │ Memory (STM) │→ │   Memory       │→ │ Memory    │  │
│  │ (deque, N=10)│  │   (SQLite)     │  │ (FAISS)   │  │
│  └──────────────┘  └────────────────┘  └───────────┘  │
│                                                        │
│  ┌──────────────────────────────────────────────────┐  │
│  │          User Profile Memory (JSON)              │  │
│  └──────────────────────────────────────────────────┘  │
│                                                        │
│  ┌──────────────┐  ┌────────────────┐                 │
│  │  Summarizer  │  │ FactExtractor  │                 │
│  │  (LLM-based) │  │  (LLM-based)   │                 │
│  └──────────────┘  └────────────────┘                 │
│                                                        │
│  ┌────────────────────────────────────────────────┐   │
│  │           ContextAssembler                     │   │
│  │   Profile + Summaries + Facts + STM → Prompt  │   │
│  └────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────┘
     │
     ▼
  Grok API
     │
     ▼
 Response + Memory Updates
```

---

## Memory Layers

| Layer | Storage | Contents | Discarded? |
|---|---|---|---|
| **Short-Term** | In-memory deque | Last 10 turns verbatim | Yes (FIFO) |
| **Episodic** | SQLite | Compressed chunk/session summaries | No (decays) |
| **Semantic** | FAISS index | Extracted factual embeddings | No (decays) |
| **User Profile** | JSON file | Long-term user facts & goals | Never |

---

## Project Structure

```
memory_agent/
├── config/
│   └── settings.py           ← All tuneable parameters
├── memory/
│   ├── short_term.py         ← FIFO deque working memory
│   ├── episodic_memory.py    ← SQLite hierarchical summaries
│   ├── semantic_memory.py    ← FAISS vector store
│   └── user_profile_memory.py← Persistent JSON profile
├── retrieval/
│   ├── embedder.py           ← Sentence-transformer wrapper
│   └── vector_store.py       ← FAISS retrieval + topic detection
├── llm/
│   └── grok_client.py        ← Grok/xAI API client (pluggable)
├── services/
│   ├── memory_agent.py       ← Main orchestrator (7-step loop)
│   ├── summarizer.py         ← LLM-based chunk summarisation
│   ├── fact_extractor.py     ← LLM-based fact extraction
│   ├── context_assembler.py  ← Structured prompt builder
│   └── importance_scorer.py  ← Composite importance scoring
├── api/
│   └── main.py               ← FastAPI server
├── tests/
│   └── test_memory.py        ← Full test suite (no API key required)
├── data/                     ← Auto-created: SQLite, FAISS index, JSON
├── example_usage.py          ← Demo script
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# 1. Clone / unzip the project
cd memory_agent

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your Grok API key
export GROK_API_KEY="xai-..."    # Windows: set GROK_API_KEY=xai-...
```

> **Note:** The embedding model (`all-MiniLM-L6-v2`, ~90 MB) is downloaded
> automatically on first run via sentence-transformers.

---

## Running the API Server

```bash
cd memory_agent
python -m uvicorn api.main:app --reload
```

Interactive API docs: http://localhost:8000/docs

---

## Testing the API

### Send a chat message

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain backpropagation through time", "session_id": "study1"}'
```

Response:
```json
{
  "response": "BPTT works by...",
  "session_id": "study1",
  "turn_count": 1,
  "token_estimate": 142,
  "compression_triggered": false,
  "latency_ms": 1240.5
}
```

### Check memory status

```bash
curl http://localhost:8000/status/study1
```

### View user profile

```bash
curl http://localhost:8000/profile/study1
```

### Manually update profile

```bash
curl -X POST http://localhost:8000/profile/study1 \
  -H "Content-Type: application/json" \
  -d '{"name": "Alice", "goal": "Master NLP for thesis", "interest": "transformers"}'
```

### Reset short-term memory (keep episodic/semantic)

```bash
curl -X DELETE http://localhost:8000/session/study1
```

---

## Running the Demo Script

```bash
cd memory_agent
python example_usage.py
```

Without a `GROK_API_KEY`, it runs in **mock mode** — demonstrating that the full
memory pipeline (STM → compression → episodic → semantic → context assembly) works
correctly, just with placeholder LLM responses.

---

## Running Tests

```bash
cd memory_agent
python -m pytest tests/ -v
```

All tests run without a live API key. The test suite covers:
- ShortTermMemory FIFO eviction, chunking, token estimation
- EpisodicMemory SQLite persistence, list field serialisation, decay
- SemanticMemory FAISS add/search/persist
- UserProfileMemory CRUD, persistence, LLM-merge
- ImportanceScorer range, keyword, recency weighting
- Token counting utilities
- Integration: STM → chunk pop → ready for summarisation

---

## Configuration

Edit `config/settings.py` to tune:

| Parameter | Default | Description |
|---|---|---|
| `short_term_max_turns` | 10 | Working memory buffer size |
| `episodic_chunk_size` | 6 | Turns per compressed chunk |
| `semantic_top_k` | 5 | FAISS retrieval results |
| `summary_top_k` | 3 | Summaries injected per prompt |
| `context_window` | 8192 | Model token limit |
| `summarisation_threshold` | 0.75 | Compress at 75% of window |
| `embedding_model` | `all-MiniLM-L6-v2` | Sentence transformer |

---

## Switching LLM Providers

The client is fully pluggable. To use OpenAI instead of Grok:

1. Create `llm/openai_client.py` subclassing `BaseLLMClient`
2. Change `settings.llm.provider = "openai"` in `config/settings.py`
3. Add `"openai"` to the factory in `llm/grok_client.py:get_llm_client()`

---

## Summarisation Trigger Logic

Compression fires when **any** of these is true:

1. Token usage exceeds **75%** of the context window
2. **Topic shift** detected (cosine similarity < 0.45 between successive messages)
3. STM buffer reaches **2× chunk size** (20 turns by default)

---

## Memory Decay

Both episodic and semantic memories implement soft decay:
- Entries with zero access count have their importance score reduced by `decay_factor` (default 0.05) each compression cycle
- Low-importance entries are retrievable but ranked lower
- Hard deletion is never automatic — a future cleanup job can prune entries below a threshold

---

## AWS Scaling Notes

For production scale-out:

| Component | Local | AWS |
|---|---|---|
| SQLite (episodic) | File | Amazon RDS (PostgreSQL) |
| FAISS (semantic) | File | Amazon OpenSearch or Pinecone |
| Session state | In-memory dict | ElastiCache (Redis) |
| API | uvicorn | ECS Fargate + ALB |
| Embeddings | Local model | SageMaker Inference |
