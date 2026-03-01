

# Project 1 — What Even Is a Token?

## 📌 Problem Statement

Building an AI application  to estimate API costs before launching.



# 🎯 Project Goal

Build a tokenizer visualizer that:

* Takes any raw text (paragraph, tweet, code, emoji, Hindi, etc.)
* Breaks it into tokens
* Displays color-coded tokens
* Shows token count
* Shows character-to-token ratio
* Estimates cost

---

# 🛠 Tech Stack

* **Python**
* **Tiktoken** (OpenAI tokenizer)
* **Streamlit** (UI framework)

---

# 🧠 What is a Token?

Before understanding the project, you must understand this:

### A token is NOT always a word.

Examples:

| Text       | Tokens                         |
| ---------- | ------------------------------ |
| `hello`    | 1 token                        |
| `unhappy`  | `un` + `happy` (2 tokens)      |
| `ChatGPT`  | may split into multiple tokens |
| `🤖🔥`     | often multiple tokens          |
| Hindi text | often more tokens than English |

LLMs process **token IDs**, not text.

---

#  "unhappy" Become 2 Tokens?

Tokenizers use **subword tokenization** (BPE — Byte Pair Encoding).

Instead of storing entire words:

* It stores common chunks.

Example:

* `happy` is common
* `un` is common
* `unhappy` may not be common enough

So it splits:

```
un + happy
```

This helps:

* Handle new words
* Keep vocabulary manageable
* Reduce model size

---

# 🌍 What Happens With Non-English Text?

Languages like:

* Hindi
* Chinese
* Arabic

Often produce **more tokens per sentence**.


Because:

* The tokenizer vocabulary is heavily optimized for English
* Subword pieces may not match cleanly
* Words get split into smaller fragments

More tokens = more cost.

---

# 💰 Why Does This Matter for Billing?

OpenAI charges **per token**.

Not per word.
Not per character.

Example pricing (hypothetical):

```
$0.002 per 1,000 tokens
```

If your prompt uses:

* 500 tokens → cheap
* 5,000 tokens → 10x more expensive
* 50,000 tokens → very expensive

Now imagine:

* 10,000 users
* Each sending 2,000 tokens per request

If you don’t understand tokenization,
you will underestimate your costs.

---

# 🚀 How to Run the Project

## 1️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate it:

Windows:

```bash
venv\Scripts\activate
```

Mac/Linux:

```bash
source venv/bin/activate
```

---

## 2️⃣ Install Dependencies

```bash
pip install streamlit tiktoken
```

---

## 3️⃣ Run the App

```bash
streamlit run app.py
```

Your browser will open automatically.

---

# 🖥 How the App Works (Step-by-Step Flow)

1. User selects a model
2. User enters text
3. The app loads the tokenizer for that model
4. Text is converted into token IDs
5. Token count is calculated
6. Character-to-token ratio is calculated
7. Cost estimate is calculated
8. Each token is displayed with a random color
9. Detailed token table is shown

---

# 📊 What You Should Experiment With

Try these inputs:

### 1️⃣ Simple English

```
Hello world
```

### 2️⃣ Complex Word

```
unhappy
```

### 3️⃣ Code Snippet

```python
def train_model(x, y):
    return x + y
```

Notice:

* Spaces are tokens
* Symbols are tokens
* Variable names split

### 4️⃣ Emojis

```
I love AI 🤖🔥
```

### 5️⃣ Hindi

```
मुझे मशीन लर्निंग पसंद है
```

Compare token counts.

---

# 🧩 What This Project Teaches You

## 1️⃣ LLMs process numbers, not text

Text → Token IDs → Embeddings → Math

## 2️⃣ Subword tokenization is a tradeoff

Not full words
Not characters
Best balance

## 3️⃣ Context window is measured in tokens

If model supports 128k tokens
That’s not 128k words.

## 4️⃣ Token count affects:

* Cost
* Latency
* Memory usage
* RAG chunking
* Prompt design


