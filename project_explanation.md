\# Complete Beginner's Guide to the Persistent Memory Study Assistant

---

## PART 1 — PROJECT OVERVIEW

### What is the goal of this project?

Imagine you have a very smart tutor who helps you study. You ask them questions, they answer brilliantly. But here's the catch — **every time you start a new conversation, they completely forget who you are, what you've discussed before, and what your goals are.** You'd have to re-introduce yourself every single time. That would be frustrating and useless.

This project solves exactly that problem. It builds an **AI-powered study assistant** that **remembers things** — across a long conversation, across many topics, and even across multiple sessions (days or weeks apart).

The assistant uses a large language model (LLM) — specifically Grok (made by xAI) — to answer your questions. But on top of that, it adds a sophisticated memory system that the LLM alone doesn't have.

---

### What problem does it solve?

LLMs like ChatGPT or Grok have something called a **context window** — think of it like their "working memory." They can only "see" and process a limited amount of text at once (e.g., 8,000 words). Once that limit is hit, they forget the oldest parts of the conversation.

So in a long study session where you ask 50+ questions, the AI would start forgetting your early questions. It might forget that you said "I'm using PyTorch" or "I have a deadline on Friday."

**This project builds a memory management system that:**
- Automatically compresses old conversation into summaries before they fall out of the window
- Extracts important facts and stores them in a searchable database
- Remembers your personal goals and preferences permanently
- Retrieves the most relevant past context whenever you ask a new question

---

### Real-life usefulness

- **Students** using it as a study tutor — it remembers your level, goals, and past explanations
- **Developers** building long-running AI assistants or chatbots
- **Researchers** who need an AI that remembers project context over weeks
- **Anyone** who wants an AI that feels like it truly "knows" them over time

---

## PART 2 — PROJECT STRUCTURE

Here is the folder layout with a plain-English description of each file:

```
memory_agent/
│
├── config/
│   └── settings.py              ← All configuration (like a control panel)
│
├── memory/
│   ├── short_term.py            ← Recent messages (like RAM)
│   ├── episodic_memory.py       ← Summaries of old chats (like a notebook)
│   ├── semantic_memory.py       ← Important facts as vectors (like a searchable index card box)
│   └── user_profile_memory.py  ← Who you are — never forgotten (like a permanent file)
│
├── retrieval/
│   ├── embedder.py              ← Converts text to numbers for searching
│   └── vector_store.py         ← Smart search engine using those numbers
│
├── llm/
│   └── grok_client.py          ← The "phone" that calls the Grok AI API
│
├── services/
│   ├── memory_agent.py         ← The "brain" — orchestrates everything
│   ├── summarizer.py           ← Compresses old chats into short summaries
│   ├── fact_extractor.py       ← Pulls out important facts from chats
│   ├── context_assembler.py    ← Builds the final prompt for the AI
│   └── importance_scorer.py    ← Scores how important a piece of text is
│
├── api/
│   └── main.py                 ← The web server — lets you talk via HTTP
│
├── tests/
│   └── test_memory.py          ← Automated tests to verify everything works
│
├── example_usage.py            ← Demo script to try the system
├── requirements.txt            ← List of Python libraries to install
└── README.md                   ← Setup instructions
```

**The big picture analogy:**

Think of the whole system like a **very organised human student's brain**:
- `short_term.py` = what they're currently thinking about (working memory)
- `episodic_memory.py` = their notebook where they write summaries after each study session
- `semantic_memory.py` = their index card box of key facts they can quickly flip through
- `user_profile_memory.py` = their permanent identity (name, goals, preferences)
- `memory_agent.py` = their conscious mind that decides what to remember, what to compress, what to look up
- `context_assembler.py` = how they mentally prepare before answering a question
- `grok_client.py` = the phone they use to call a brilliant expert friend for answers

---

## PART 3 — FILE-BY-FILE CODE EXPLANATION

---

### FILE 1: `config/settings.py` — The Control Panel

**Purpose:** This file contains all the settings for the entire project — like a configuration dashboard. Instead of scattering numbers like "10" and "8192" all over the code, they're all defined here in one place. This makes the project easy to tune without hunting through dozens of files.

```python
@dataclass
class MemorySettings:
    short_term_max_turns: int = 10
    episodic_chunk_size: int = 6
    ...
```

**What's a `@dataclass`?**
A dataclass is a Python shortcut for creating a class that just holds data. Instead of writing `__init__` yourself, Python auto-generates it. Think of it like a form with pre-filled defaults.

**What do these settings mean?**
- `short_term_max_turns = 10` → only keep the last 10 messages in live memory
- `episodic_chunk_size = 6` → after every 6 messages, compress them into a summary
- `context_window = 8192` → the AI can only handle 8,192 tokens (words) at once
- `summarisation_threshold = 0.75` → start compressing when 75% of the window is used

```python
settings = AppSettings()
```

**Why this line at the bottom?**
This creates one single instance of the settings. Every other file in the project imports this one `settings` object. This is called the **Singleton pattern** — one object shared everywhere, like a company's single policy manual.

---

### FILE 2: `memory/short_term.py` — The Working Memory

**Purpose:** Stores the most recent conversation messages exactly as they are. When this fills up, the oldest messages are evicted (deleted) to make room. Think of it like a whiteboard — limited space, constantly updated.

```python
from collections import deque

class ShortTermMemory:
    def __init__(self, max_turns=10):
        self._buffer = deque(maxlen=10)
```

**What is a `deque`?**
A `deque` (pronounced "deck") is a special Python list that has a maximum size. When it's full and you add something new, it **automatically removes the oldest item from the front**. Like a queue at a coffee shop — when capacity is full, the oldest person has to leave when a new one joins.

```python
@dataclass
class Turn:
    role: Role          # "user" or "assistant"
    content: str        # the actual message text
    timestamp: str      # when it was said
    turn_id: int        # message number (1, 2, 3...)
    importance: float   # how important is this message?
```

**What is this?**
Each message (called a "turn") is stored as a `Turn` object containing all its info. Like a row in a table.

```python
def pop_oldest_chunk(self, chunk_size: int) -> List[Turn]:
    chunk = []
    for _ in range(min(chunk_size, len(self._buffer))):
        chunk.append(self._buffer.popleft())
    return chunk
```

**What does this do?**
This removes the oldest N messages from the front of the deque and returns them. Why? Because those old messages need to be **compressed and stored** in episodic memory before they disappear forever. Like tearing pages from the left side of your notebook to file them away.

```python
def token_estimate(self) -> int:
    chars = sum(len(t.content) for t in self._buffer)
    return chars // settings.tokens.chars_per_token
```

**What is a "token"?**
AI models don't count words — they count "tokens." A token is roughly 4 characters in English. "Hello" = 1 token. "Backpropagation" = about 4 tokens. This function estimates how many tokens the current conversation is using so we know when we're getting close to the limit.

---

### FILE 3: `memory/episodic_memory.py` — The Notebook

**Purpose:** Stores compressed summaries of older conversation chunks. When the working memory gets full, old messages are summarised (by the AI itself) and that summary is stored here permanently in a SQLite database.

**What is SQLite?**
SQLite is a file-based database — like a super-powered spreadsheet stored in a `.db` file. Python has built-in support for it. No server needed — just a file on your laptop.

```python
@dataclass
class EpisodicEntry:
    entry_id: str
    summary: str
    decisions: List[str]          # e.g. ["Use Adam optimiser"]
    technical_facts: List[str]    # e.g. ["Learning rate = 0.001"]
    constraints: List[str]        # e.g. ["Must run on CPU"]
    user_goals: List[str]         # e.g. ["Understand BPTT"]
    unresolved_questions: List[str]
    topic_tags: List[str]
    importance_score: float
```

**Why not just store the raw messages?**
Raw messages take a lot of space. A 6-message conversation might be 1,000 words. The summary might be 100 words. We compress by 10x while preserving everything important. Like how a meeting's key decisions fit in bullet points rather than the full transcript.

```python
_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS episodic_memory (
    entry_id TEXT PRIMARY KEY,
    summary  TEXT NOT NULL,
    ...
)
"""
```

**What is SQL?**
SQL is the language for talking to databases. `CREATE TABLE IF NOT EXISTS` means "create this table if it doesn't already exist." This runs once when the app starts.

```python
@contextmanager
def _conn(self):
    conn = sqlite3.connect(self._db)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
```

**What is a `@contextmanager`?**
This is a Python pattern for safely managing resources. When you use `with self._conn() as conn:`, Python guarantees the connection will be **properly closed** even if an error occurs — like a `try/finally` but cleaner. Without this, a crashed program could corrupt the database.

```python
def decay_old_entries(self, decay_factor=0.05):
    sql = """
        UPDATE episodic_memory
        SET importance_score = MAX(0.0, importance_score - ?)
        WHERE access_count = 0
    """
```

**What is "memory decay"?**
Like human memory — things you never revisit slowly fade. Entries that have never been retrieved get their importance score reduced by 0.05 each cycle. They're never deleted, but they rank lower when retrieved. This mimics how humans naturally forget things they don't use.

---

### FILE 4: `memory/semantic_memory.py` — The Searchable Index Card Box

**Purpose:** Stores important facts as mathematical vectors (lists of numbers) so they can be searched by meaning — not just by keywords. Uses FAISS, a library built by Meta, for fast similarity search.

**What is a vector?**
Imagine converting the sentence "PyTorch is a deep learning framework" into 384 numbers. Those numbers capture the **meaning** of the sentence in mathematical form. Two sentences with similar meaning will have similar numbers. This is called an "embedding."

Why is this useful? Because you can then search by meaning. If you ask "what deep learning library am I using?", the system can find "PyTorch is a deep learning framework" even though you didn't use the word "PyTorch."

```python
self._index = faiss.IndexFlatIP(self._dim)
```

**What is FAISS and IndexFlatIP?**
FAISS (Facebook AI Similarity Search) is a library for finding similar vectors quickly. `IndexFlatIP` is a specific type of index that uses the "inner product" (dot product) to measure similarity. When vectors are normalised (scaled to length 1), this equals **cosine similarity** — a standard way to measure how similar two directions are in vector space.

```python
@staticmethod
def _normalise(vec):
    norm = np.linalg.norm(vec)
    return (vec / norm) if norm > 0 else vec
```

**Why normalise?**
Like adjusting the volume before comparing two songs. Normalisation ensures all vectors have the same "scale" (length = 1), so similarity scores are fair comparisons of direction, not length.

```python
def search(self, query_embedding, top_k=5):
    scores, indices = self._index.search(q, k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        meta = self._meta[idx]
        results.append((meta, float(score)))
    return results
```

**How does search work?**
1. You provide a query as a vector (e.g., "explain attention mechanism" converted to numbers)
2. FAISS compares that vector to every stored fact vector
3. It returns the top-k most similar ones (fastest match in the entire database, even with millions of entries)

```python
def decay(self, decay_factor=0.05):
    for m in self._meta:
        if m.get("access_count", 0) == 0:
            m["importance"] = max(0.0, m["importance"] - decay_factor)
```

Same decay concept as episodic memory — facts you never retrieve slowly lose priority.

---

### FILE 5: `memory/user_profile_memory.py` — The Permanent File

**Purpose:** Stores long-term facts about you that should **never** be automatically discarded. Your goals, your projects, your preferences. These always appear in every prompt.

```python
_DEFAULT_PROFILE = {
    "user_id": "default",
    "name": "",
    "goals": [],
    "preferences": {},
    "projects": [],
    "technical_interests": [],
    "constraints": [],
    "custom_facts": {},
}
```

This is the starting template. New users start with an empty profile that gets filled as you chat.

```python
def add_goal(self, goal: str) -> None:
    if goal not in self._profile["goals"]:
        self._profile["goals"].append(goal)
        self._save()
```

**Why check `if goal not in`?**
Prevents duplicates. If the AI extracts "Learn PyTorch" twice from different conversations, it's only stored once. The `self._save()` writes the updated profile to the JSON file immediately — so it's safe even if the program crashes.

```python
def format_for_prompt(self) -> str:
    lines = []
    if goals := self._profile.get("goals", []):
        lines.append("Goals:\n" + "\n".join(f"  • {g}" for g in goals))
    ...
    return "\n".join(lines)
```

**What is `:=` (walrus operator)?**
This is Python 3.8+ syntax. `if goals := self._profile.get("goals", []):` means "get the goals list, assign it to `goals`, and check if it's not empty" — all in one line. Saves a variable assignment.

**What does `format_for_prompt` return?**
A nicely formatted text block like:
```
Goals:
  • Learn PyTorch
  • Master transformers
Technical interests: NLP, BERT
Constraints:
  • Must run on CPU
```

This text gets injected into every prompt so the AI always knows who you are.

---

### FILE 6: `retrieval/embedder.py` — The Text-to-Numbers Converter

**Purpose:** Converts text (words) into vectors (numbers) using a pre-trained model called `all-MiniLM-L6-v2`. This model was trained on millions of sentences and learned what words mean in context.

```python
_model_cache = {}

def _get_model(model_name):
    if model_name not in _model_cache:
        from sentence_transformers import SentenceTransformer
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]
```

**What is model caching?**
Loading the embedding model takes a few seconds. If you called `SentenceTransformer("all-MiniLM-L6-v2")` every time you wanted to embed a word, the program would be slow. Instead, this loads it once and saves it in the dictionary `_model_cache`. Every subsequent call instantly gets the already-loaded model — like keeping a book open on your desk instead of fetching it from the shelf every time.

```python
def encode(self, text: str) -> List[float]:
    model = self._load()
    vec = model.encode(text, normalize_embeddings=True)
    return vec.tolist()
```

`normalize_embeddings=True` means "scale the vector to length 1 automatically." This ensures the FAISS inner-product search works correctly as cosine similarity.

```python
def similarity(self, text_a: str, text_b: str) -> float:
    va = self.encode(text_a)
    vb = self.encode(text_b)
    a, b = np.array(va), np.array(vb)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
```

**What is cosine similarity?**
Imagine two arrows pointing from the origin. Cosine similarity measures the angle between them — not how long they are. If both arrows point in the same direction (angle = 0°), similarity = 1.0. If they're perpendicular (angle = 90°), similarity = 0.0. Semantically similar sentences have small angles between their vectors.

---

### FILE 7: `retrieval/vector_store.py` — The Smart Search Engine

**Purpose:** A higher-level wrapper that combines the Embedder and SemanticMemory. Adds importance-weighted reranking and topic-shift detection on top of raw FAISS search.

```python
def query(self, text: str, top_k=5) -> List[Tuple[dict, float]]:
    q_vec = self.embedder.encode(text)
    raw = self.semantic.search(q_vec, top_k=k*2)  # get 2x more candidates
    
    # Re-rank: blend cosine score with importance
    reranked = []
    for meta, cos_score in raw:
        importance = meta.get("importance", 0.5)
        combined = (0.40 * cos_score) + (0.30 * importance)
        reranked.append((meta, combined))
    
    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked[:k]
```

**Why get 2x candidates then re-rank?**
FAISS returns results purely by cosine similarity. But a slightly-less-similar fact that's been accessed many times and has high importance might be more useful. So we fetch double the needed results and re-rank using a blend of similarity + importance. This is called **hybrid retrieval** — combining multiple signals.

```python
TOPIC_SHIFT_THRESHOLD = 0.45

def detect_topic_shift(self, new_message: str) -> bool:
    new_vec = self.embedder.encode(new_message)
    similarity = cosine(self._last_topic_vec, new_vec)
    
    # Update rolling average (EMA)
    alpha = 0.4
    self._last_topic_vec = (alpha * new_vec + (1 - alpha) * old_vec)
    
    return similarity < TOPIC_SHIFT_THRESHOLD
```

**What is topic shift detection?**
The system tracks what topic you've been discussing. If your new message is very different from recent messages (similarity < 0.45), it infers you've changed topic. This triggers early summarisation — compressing the old topic before moving to the new one.

**What is EMA (Exponential Moving Average)?**
Instead of storing just the last message's vector as "the topic," it blends the new vector (40% weight) with the old running average (60% weight). This smoothly tracks the evolving topic rather than reacting to every single message.

---

### FILE 8: `llm/grok_client.py` — The Phone to the AI

**Purpose:** Handles all communication with the Grok AI API. Like a phone with redial — if the call drops, it tries again automatically.

```python
def count_tokens_simple(text: str) -> int:
    return max(1, len(text) // 4)
```

**Why divide by 4?**
In English, on average 1 token ≈ 4 characters. This is a fast approximation. Real tokenisers (like OpenAI's tiktoken) are more accurate but require an extra library. For our threshold calculations, this approximation is sufficient.

```python
class BaseLLMClient(ABC):
    @abstractmethod
    def chat(self, messages, temperature, max_tokens, system):
        ...
```

**What is ABC (Abstract Base Class)?**
An ABC is like a "contract" in code. It says: "Any class that inherits from me MUST implement the `chat` method." This ensures you can swap in a different LLM (OpenAI, Anthropic, Ollama) and the rest of the code won't break — because they all follow the same interface. Like how every phone, regardless of brand, must have a way to make a call.

```python
def _call_with_retry(self, payload):
    for attempt in range(1, self._max_retries + 1):
        try:
            response = self._client.post(url, json=payload, headers=headers)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            
            if response.status_code in (429, 500, 502, 503):
                wait = 2 ** attempt   # 2s, 4s, 8s...
                time.sleep(wait)
                continue
        except httpx.RequestError:
            time.sleep(2 ** attempt)
```

**What is exponential backoff?**
If the server is busy (HTTP 429 = "Too Many Requests"), you wait 2 seconds, then 4, then 8. Each retry waits twice as long as the previous one. This gives the server time to recover without hammering it with repeated requests.

**What is `response.json()["choices"][0]["message"]["content"]`?**
The Grok API returns a JSON response like:
```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "The attention mechanism works by..."
      }
    }
  ]
}
```
So `["choices"][0]["message"]["content"]` navigates through that nested structure to extract just the AI's reply text.

```python
def get_llm_client() -> BaseLLMClient:
    provider = settings.llm.provider.lower()
    if provider == "grok":
        return GrokClient()
    raise ValueError(f"Unknown provider: {provider}")
```

**Why a factory function?**
A factory is a function that creates and returns objects. Instead of every file doing `from llm.grok_client import GrokClient; client = GrokClient()`, they just call `get_llm_client()`. If you change the provider in settings, all files automatically get the new client. This is the **Factory pattern**.

---

### FILE 9: `services/importance_scorer.py` — The Priority Calculator

**Purpose:** Assigns a score from 0 to 1 to any piece of text based on how important it is to remember. This guides compression decisions and retrieval ranking.

```python
def score(self, text, turn_index=0, total_turns=1):
    recency  = self._recency_score(turn_index, total_turns)
    density  = self._semantic_density(text)
    keywords = self._keyword_score(text)
    
    composite = (0.30 * recency) + (0.40 * density) + (0.30 * keywords)
    return round(min(1.0, max(0.0, composite)), 4)
```

The score is a **weighted average** of three sub-scores:
- 30% from recency (how recent is this message?)
- 40% from semantic density (how information-rich is the text?)
- 30% from keywords (does it contain important words?)

```python
@staticmethod
def _recency_score(turn_index, total_turns):
    distance = total_turns - turn_index - 1
    return math.exp(-0.1 * distance)
```

**What is `math.exp`?**
This is an exponential decay function. `e^(-0.1 × distance)` gives scores like:
- 0 messages ago → score = 1.0
- 5 messages ago → score = 0.61
- 20 messages ago → score = 0.14

Recent messages are important; old ones less so. The decay is smooth, not sudden.

```python
IMPORTANT_KEYWORDS = frozenset({
    "goal", "objective", "requirement", "must", "critical",
    "deadline", "constraint", "decision", "conclusion", ...
})

def _keyword_score(text):
    words = set(text.lower().split())
    hits = words.intersection(IMPORTANT_KEYWORDS)
    return min(1.0, len(hits) / 3)
```

**Why `frozenset`?**
A `frozenset` is like a `set` but immutable (can't be changed). Checking `if word in frozenset` is extremely fast — O(1) — because Python uses a hash table internally. A `frozenset` also signals to other developers: "don't modify this, it's a constant."

---

### FILE 10: `services/summarizer.py` — The Compressor

**Purpose:** Takes a list of raw conversation turns and asks the AI to summarise them into a structured format. The AI is essentially being used to compress itself.

```python
_CHUNK_SUMMARY_PROMPT = """
You are a conversation memory manager. Below is a segment of a conversation.
Extract a structured summary in valid JSON with exactly these keys:
{
  "summary": "...",
  "decisions": ["..."],
  "technical_facts": ["..."],
  "constraints": ["..."],
  "user_goals": ["..."],
  "unresolved_questions": ["..."],
  "topic_tags": ["..."]
}
Output ONLY valid JSON — no markdown fences, no extra text.
Conversation segment:
{conversation}
"""
```

**This is "prompt engineering."**
The carefully written instructions in the prompt guide the AI to produce exactly the structured output we need. Notice:
- It says "Output ONLY valid JSON" — preventing the AI from adding preamble text
- It specifies every key — ensuring consistent output format
- "Preserve ALL technical details" — so important info isn't lost

```python
raw = self._llm.chat(
    messages=[{"role": "user", "content": prompt}],
    temperature=0.3,
)
```

**Why `temperature=0.3`?**
Temperature controls AI creativity/randomness:
- Temperature = 1.0 → creative, varied, sometimes unpredictable
- Temperature = 0.3 → focused, consistent, reliable
- Temperature = 0.0 → always the same answer

For extraction tasks (where we need accurate, structured output), low temperature is better. For creative writing, high temperature is better.

```python
def _parse_json_response(raw):
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:-1])  # strip markdown fences
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: return minimal valid dict
        return {"summary": raw[:500], "decisions": [], ...}
```

**Why handle JSON errors?**
Even with good prompting, AI models sometimes add markdown code fences (` ```json `) or make small formatting mistakes. This function cleans up common issues. The fallback ensures the program never crashes — even if the AI misbehaves, the summary still gets stored (with just the raw text).

---

### FILE 11: `services/fact_extractor.py` — The Knowledge Miner

**Purpose:** Asks the AI to pull out discrete, reusable facts from a conversation. These facts are stored in the FAISS vector database for later retrieval.

```python
_FACT_EXTRACTION_PROMPT = """
Given the conversation below, extract a list of stable, reusable facts.
Return ONLY a valid JSON array. Each element:
{
  "text": "<the fact in one clear sentence>",
  "category": "<technical | goal | constraint | preference | definition>"
}
Rules:
- Extract only facts useful in a future conversation.
- Do NOT extract filler or transient state.
- Maximum 10 facts per call.
"""
```

**What kinds of facts get extracted?**
From "I'm building a text summarisation model for scientific papers using PyTorch, and it must run on CPU in under 2 seconds," the extractor might produce:
```json
[
  {"text": "User is building a text summarisation model for scientific papers", "category": "goal"},
  {"text": "User is using PyTorch as the deep learning framework", "category": "technical"},
  {"text": "Model must run on CPU only", "category": "constraint"},
  {"text": "Inference time constraint: under 2 seconds per document", "category": "constraint"}
]
```

These are stored as vectors so that if you later ask "what framework am I using?", the system can find the relevant fact.

---

### FILE 12: `services/context_assembler.py` — The Prompt Builder

**Purpose:** This is like a chef who assembles all the ingredients (four memory types) into a single, well-structured meal (prompt) for the AI. This is the most important file for prompt quality.

```python
_SYSTEM_HEADER = """You are an expert AI study assistant with persistent memory.
You have access to the user's profile, past conversation summaries, semantic 
knowledge, and the current conversation.
Use all this context to provide accurate, personalised, and coherent answers.
Never pretend you don't remember something that appears in the context below."""
```

**What is a "system prompt"?**
In LLM APIs, messages have roles: "system", "user", "assistant". The system message is instructions that shape the AI's behaviour for the entire conversation — like a job description given before an interview. It always comes first.

```python
def assemble(self, query, recent_turns):
    profile_text    = self._profile.format_for_prompt()
    summaries_text  = self._fetch_relevant_summaries(query)
    facts_text      = self._fetch_relevant_facts(query)
    system          = self._build_system(profile_text, summaries_text, facts_text)
    messages        = [{"role": t.role, "content": t.content} for t in recent_turns]
    messages        = self._fit_messages_to_budget(messages, budget)
    return system, messages
```

**The full assembled prompt looks like:**

```
[SYSTEM]
You are an expert AI study assistant...

── USER PROFILE ──────────────────────────────────────────
Goals:
  • Master transformer architectures
Technical interests: NLP, PyTorch

── RELEVANT CONVERSATION SUMMARIES ──────────────────────
[Summary 1] We discussed attention mechanisms in detail.
Decisions: Use scaled dot-product attention
Facts: d_k is the key dimension; scaling by 1/sqrt(d_k) prevents gradient issues

── RELEVANT KNOWLEDGE (from memory) ─────────────────────
• [technical] PyTorch is the user's framework  (relevance: 0.87)
• [constraint] Model must run on CPU only      (relevance: 0.81)

── RECENT CONVERSATION ───────────────────────────────────
(The last 8 turns are provided below as structured messages.)

[MESSAGES]
user: What is backpropagation through time?
assistant: BPTT is...
user: Why does it have vanishing gradients?
```

This rich context is what allows the AI to give personalised, context-aware answers even deep into a long conversation.

```python
@staticmethod
def _fit_messages_to_budget(messages, token_budget):
    while len(messages) > 2:
        total = sum(count_tokens_simple(m["content"]) for m in messages)
        if total <= token_budget:
            break
        messages = messages[1:]   # drop oldest
    return messages
```

**What is "budget trimming"?**
Even after compression, the recent messages might still be too long. This loop drops the oldest message one at a time until everything fits within the token budget. It always keeps the last 2 messages (the most recent exchange) no matter what.

---

### FILE 13: `services/memory_agent.py` — The Brain / Orchestrator

**Purpose:** This is the central "brain" that connects all other components. Every user message goes through this file's `chat()` method, which runs the full 7-step pipeline.

```python
class MemoryAgent:
    def __init__(self, session_id="default"):
        self.stm          = ShortTermMemory()
        self.episodic     = EpisodicMemory()
        self.semantic     = SemanticMemory()
        self.vector_store = VectorStore(...)
        self.user_profile = UserProfileMemory()
        self._llm         = get_llm_client()
        self._summarizer  = Summarizer(llm=self._llm)
        self._fact_extractor = FactExtractor(...)
        self._assembler   = ContextAssembler(...)
```

The constructor creates one instance of every component. They're all wired together here.

```python
def chat(self, user_message: str) -> AgentResponse:
    # Step 1: Store message
    self.stm.add("user", user_message)
    
    # Step 2: Estimate tokens
    current_tokens = self._estimate_total_tokens(user_message)
    threshold = int(8192 * 0.75)  # 6144 tokens
    
    # Step 3: Compress if needed
    should_compress = (
        current_tokens > threshold      # too many tokens
        or topic_shift                  # topic changed
        or len(self.stm) >= chunk_size * 2  # buffer too full
    )
    if should_compress:
        self._compress_oldest_chunk()
    
    # Step 4+5: Assemble context
    system_prompt, messages = self._assembler.assemble(
        query=user_message,
        recent_turns=self.stm.get_all(),
    )
    
    # Step 6: Call LLM
    reply = self._llm.chat(messages=messages, system=system_prompt)
    
    # Step 7: Store reply
    self.stm.add("assistant", reply)
    self._maybe_update_user_profile(user_message, reply)
    
    return AgentResponse(reply=reply, ...)
```

This `chat()` method is the heartbeat of the entire system. Every single user message flows through these 7 steps.

```python
def _compress_oldest_chunk(self):
    chunk = self.stm.pop_oldest_chunk(chunk_size)
    
    # Summarise into episodic memory
    entry = self._summarizer.summarise_chunk(chunk, session_id=self.session_id)
    self.episodic.store(entry)
    
    # Extract facts into semantic memory
    facts = self._fact_extractor.extract_from_turns(chunk)
    self.vector_store.add_facts_batch(facts)
    
    # Apply decay
    self.vector_store.decay()
    self.episodic.decay_old_entries()
```

**Compression is a 3-step process:**
1. Pop old turns from STM
2. LLM summarises → stored in SQLite
3. LLM extracts facts → stored in FAISS
Then decay is applied to old, unused memories.

---

### FILE 14: `api/main.py` — The Web Server

**Purpose:** Wraps the MemoryAgent in a web API using FastAPI, so you can send messages via HTTP (from a browser, mobile app, or curl).

```python
_agents: dict[str, MemoryAgent] = {}

def _get_agent(session_id: str) -> MemoryAgent:
    if session_id not in _agents:
        _agents[session_id] = MemoryAgent(session_id=session_id)
    return _agents[session_id]
```

**Session management:** Different users (or the same user on different devices) can have different session IDs. Each session gets its own MemoryAgent with isolated memory. Like different customer accounts at a bank.

```python
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4096)
    session_id: str = Field(default="default")

class ChatResponse(BaseModel):
    response: str
    turn_count: int
    token_estimate: int
    compression_triggered: bool
    latency_ms: float
```

**What is Pydantic `BaseModel`?**
FastAPI uses Pydantic to automatically validate incoming and outgoing data. `min_length=1` means the API will reject empty messages with an error before your code even runs. `max_length=4096` prevents very long inputs. This is input validation — a critical security and reliability feature in web APIs.

```python
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    agent = _get_agent(request.session_id)
    result = agent.chat(request.message)
    return ChatResponse(response=result.reply, ...)
```

**What does `async def` mean?**
`async` means this function is asynchronous — it can handle multiple requests simultaneously without blocking. While waiting for the Grok API to respond (which might take 1-2 seconds), the server can handle other incoming requests. Without `async`, the server would be frozen during each API call.

---

### FILE 15: `tests/test_memory.py` — The Quality Assurance

**Purpose:** Automated tests that verify every component works correctly. Run them any time you make changes to catch bugs immediately.

```python
class TestShortTermMemory(unittest.TestCase):
    def setUp(self):
        self.stm = ShortTermMemory(max_turns=5)
    
    def test_fifo_eviction(self):
        for i in range(7):
            self.stm.add("user", f"msg {i}")
        self.assertEqual(len(self.stm), 5)
        texts = [t.content for t in self.stm.get_all()]
        self.assertNotIn("msg 0", texts)
        self.assertIn("msg 6", texts)
```

**How to read this test:**
1. Create a STM with max 5 turns
2. Add 7 messages
3. Assert only 5 remain (FIFO eviction worked)
4. Assert the oldest 2 messages are gone
5. Assert the newest message is still there

Each test is a small, focused check. When all 35 pass, you have high confidence the system works as designed.

**Why `setUp`?**
`setUp` runs before every single test method, creating a fresh instance. This prevents tests from affecting each other — each test starts from a clean slate.

---

## PART 4 — PROJECT FLOW (End-to-End)

Let's trace exactly what happens when you type one message: **"Explain the attention mechanism in transformers."**

---

### Step-by-step walkthrough

```
YOU TYPE: "Explain the attention mechanism in transformers."
```

**Step 1 → api/main.py**
FastAPI receives your HTTP POST request. It validates the JSON body. It calls `_get_agent("your-session")` — if this session already exists, it reuses it; if new, it creates a fresh MemoryAgent. Then it calls `agent.chat("Explain the attention mechanism...")`.

**Step 2 → services/memory_agent.py (chat method)**
The orchestrator takes over. It immediately calls `self.stm.add("user", "Explain the attention mechanism...")`.

**Step 3 → memory/short_term.py**
The turn is appended to the deque as a `Turn` object with role="user", content="Explain...", timestamp, and turn_id.

**Step 4 → services/memory_agent.py (token check)**
Estimates current token count. Let's say we're at Turn 15 and the buffer is filling up. The estimate comes back at 5,200 tokens. The threshold is 6,144 (75% of 8,192).
- 5,200 < 6,144 → no compression needed yet

Also checks `detect_topic_shift()` in `vector_store.py` — embedder converts the new message to a vector, compares with the running topic vector. Similarity is 0.72 — no topic shift. Buffer has 10 turns — not yet at 12 (2× chunk size).

**No compression triggered.** Proceed to assembly.

*(But let's say at Turn 20 the buffer fills up — here's what happens in compression:)*

**Compression step → services/memory_agent.py (_compress_oldest_chunk)**
```
STM currently has: [turns 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
pop_oldest_chunk(6) removes: [turns 11, 12, 13, 14, 15, 16]
STM now has: [turns 17, 18, 19, 20]
```
The 6 popped turns go to:

→ **services/summarizer.py**
Formats those 6 turns as a conversation text, builds the summarisation prompt, calls `grok_client.py` with temperature=0.3. Grok returns JSON:
```json
{
  "summary": "Discussed transformer attention mechanism, specifically scaled dot-product and multi-head attention.",
  "decisions": ["Use multi-head attention with 8 heads"],
  "technical_facts": ["Scaling factor is 1/sqrt(d_k)", "Attention = softmax(QKᵀ/√dₖ)V"],
  ...
}
```
This gets stored in `episodic_memory.db` (SQLite).

→ **services/fact_extractor.py**
Separately, calls Grok again with the fact extraction prompt. Returns:
```json
[
  {"text": "Transformers use scaled dot-product attention", "category": "technical"},
  {"text": "Multi-head attention uses 8 parallel attention heads", "category": "technical"}
]
```
Each fact gets embedded by `retrieval/embedder.py` (converted to 384 numbers), then stored in the FAISS index via `memory/semantic_memory.py`.

**Back to main flow — Step 5 → services/context_assembler.py**
Now we build the prompt for your current question. The assembler:

1. Calls `user_profile.format_for_prompt()` → gets your goals, projects, interests as text
2. Calls `episodic.get_recent_summaries()` → retrieves the most recent summaries from SQLite
3. Calls `vector_store.query("attention mechanism transformer")` → embeds your query, searches FAISS, gets the top 5 relevant facts
4. Gets recent turns from STM → the last few messages

Combines all of these into the structured system prompt + messages list.

**Step 6 → llm/grok_client.py**
Sends the full assembled prompt to the Grok API via HTTP POST. Waits for response. If it fails (HTTP 429), waits and retries with exponential backoff. Eventually gets back the answer text.

**Step 7 → services/memory_agent.py (store reply)**
The AI's reply is added to STM: `self.stm.add("assistant", "The attention mechanism works by...")`.

Every 10 turns, it also calls Grok to extract user profile updates from the recent exchange and merges them into `user_profile_memory.py`.

**Step 8 → api/main.py**
The `AgentResponse` is converted to a `ChatResponse` Pydantic model and returned as JSON to you:
```json
{
  "response": "The attention mechanism works by...",
  "turn_count": 20,
  "token_estimate": 4200,
  "compression_triggered": true,
  "latency_ms": 1580
}
```

**Total time: ~1-2 seconds** (mostly waiting for the Grok API).

---

## PART 5 — KEY CONCEPTS EXPLAINED

---

### 1. Large Language Models (LLMs)
An LLM is a neural network trained on billions of text documents. It learned patterns of language so well that it can generate coherent, knowledgeable text on almost any topic. Think of it as a person who has read the entire internet and can answer questions based on everything they've absorbed.

**Limitation:** LLMs don't have memory between API calls. Each call is stateless — like calling a brilliant consultant who has amnesia after every call.

---

### 2. Context Window
Every LLM can only process a limited amount of text at once. This is the "context window." For Grok, it's about 8,192 tokens (~6,000 words). Everything outside the window is simply invisible to the model.

**Analogy:** Imagine reading a book, but you can only see 10 pages at a time. The further you read, the less you can see of what you read earlier.

---

### 3. Embeddings / Vectors
Converting text to numbers that represent its meaning. The model `all-MiniLM-L6-v2` converts "I love Python programming" to 384 numbers. Similar sentences have numerically similar vectors.

**Analogy:** Like GPS coordinates for meaning. Paris and Lyon are geographically close. "Cat" and "kitten" are semantically close — their vector coordinates are nearby in 384-dimensional space.

---

### 4. Vector Similarity Search (FAISS)
Given a query vector, FAISS finds the most similar stored vectors in milliseconds — even across millions of entries. It's like a GPS that instantly finds the nearest 5 locations to your current position in a database of 1 million places.

---

### 5. SQLite
A file-based relational database. All data is stored in a single `.db` file. No server to set up. Python has built-in support (`import sqlite3`). For this project's scale (thousands of summaries), SQLite is perfect. For millions of users, you'd switch to PostgreSQL.

---

### 6. FastAPI
A modern Python web framework that turns Python functions into HTTP endpoints. You write:
```python
@app.post("/chat")
async def chat(request: ChatRequest):
    ...
```
And FastAPI automatically handles HTTP parsing, JSON serialisation, validation, error responses, and even generates interactive documentation at `/docs`.

---

### 7. Prompt Engineering
The art of writing instructions to an LLM to get specific, reliable outputs. In this project:
- The summarisation prompt instructs the AI to output structured JSON with specific keys
- The fact extraction prompt instructs it to categorise facts
- The system prompt tells it to always use the provided context
The quality of these prompts directly determines the quality of the system.

---

### 8. Cosine Similarity
A mathematical measure of how similar two vectors are. Value ranges from -1 (opposite) to 1 (identical). For text embeddings, typically 0 to 1. Used to:
- Find relevant facts in FAISS
- Detect topic shifts (similarity drops below 0.45)
- Re-rank retrieval results

---

### 9. Memory Decay
Inspired by human memory — information we don't use fades. In this system, facts and summaries with zero access count have their importance score reduced each cycle. They're never deleted (you might need them), but they rank lower. Facts you frequently retrieve maintain high importance.

---

### 10. The Four Memory Types (and their human analogies)

| System Memory | Human Brain Equivalent | Technical Implementation |
|---|---|---|
| Short-Term (STM) | What you're currently thinking about | Python deque |
| Episodic | Diary entries — what happened in past sessions | SQLite summaries |
| Semantic | General knowledge — facts you "just know" | FAISS vectors |
| Profile | Your identity — who you are | JSON file |

---

## PART 6 — HOW IT ALL CONNECTS

Here's the complete picture of how all pieces work together:

```
               YOU (Human)
                   │
                   │ HTTP POST /chat
                   ▼
          ┌─────────────────┐
          │  FastAPI Server  │  ← api/main.py
          │  (api/main.py)   │
          └────────┬────────┘
                   │ calls agent.chat(message)
                   ▼
          ┌─────────────────────────────────────────────────────┐
          │              MemoryAgent                            │
          │            (services/memory_agent.py)               │
          │                                                     │
          │  1. ADD to ShortTermMemory (deque)                  │
          │  2. CHECK token count                               │
          │  3. IF threshold exceeded:                          │
          │     ├── POP chunk from STM                         │
          │     ├── SUMMARIZE chunk → EpisodicMemory (SQLite)  │
          │     └── EXTRACT facts → SemanticMemory (FAISS)     │
          │  4. RETRIEVE context:                              │
          │     ├── Recent turns from STM                      │
          │     ├── Top summaries from EpisodicMemory          │
          │     ├── Top facts from VectorStore/FAISS           │
          │     └── User profile from UserProfileMemory        │
          │  5. ASSEMBLE prompt (ContextAssembler)             │
          │  6. CALL Grok API (GrokClient)                     │
          │  7. STORE reply in STM                             │
          └─────────────────────────────────────────────────────┘
                   │
                   │ JSON response
                   ▼
               YOU (Human)
```

**The project achieves its goal by solving two problems simultaneously:**

**Problem A — Memory loss:** As conversations grow, old messages fall out of the context window. The system solves this by compressing old messages into summaries and extracting facts — both of which are much smaller than the original messages.

**Problem B — Relevant retrieval:** With summaries and facts stored, the system needs to know which ones are relevant to the current question. It solves this with vector similarity search — finding the most semantically relevant stored knowledge for each new query.

**The result:** An AI that can have a coherent, personalised, knowledgeable 50+ turn conversation while never exceeding the model's context window — because it intelligently manages what to keep, what to compress, and what to retrieve.

---

*This project is a miniature version of how production AI assistants like memory-enabled ChatGPT work internally. The same principles — short-term working memory, compression, vector retrieval, and persistent profiles — are used in real AI products at scale.*
