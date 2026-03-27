# 🧪 Prompt Testing Lab

> Systematically compare **Zero-Shot**, **Few-Shot**, **Chain-of-Thought**, and **Role Prompting** strategies against the Groq API — side by side, with analytics.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Module Reference](#module-reference)
- [Prompt Strategies](#prompt-strategies)
- [Analytics Explained](#analytics-explained)
- [Configuration](#configuration)
- [Extending the Lab](#extending-the-lab)
- [Troubleshooting](#troubleshooting)

---

## Overview

Prompt Testing Lab is a **Streamlit** application that lets you:

1. Describe any NLP task (e.g. *"classify customer sentiment"*)
2. Provide 1–5 test inputs (one per line)
3. Run all four prompting strategies against Claude in a single click
4. Compare responses, consistency scores, response lengths, and robustness flags in a polished dashboard

All API calls go through `api_client.py`. All analytics live in `analytics.py`. The UI is split into four tab modules under `ui/`. Nothing is hard-coded — API key is entered at runtime via the sidebar.

---

## Features

| Feature | Description |
|---|---|
| **4-Strategy Comparison** | Zero-Shot, Few-Shot (auto-examples), CoT, Role Prompting |
| **Auto Few-Shot Examples** | Claude generates 3 task-relevant examples automatically |
| **Results Table** | Truncated responses in a clean dataframe with consistency scores |
| **Consistency Score** | Pairwise Jaccard similarity across all 4 responses per input |
| **Response Length Chart** | Altair bar chart — avg word count per strategy |
| **Outlier / Robustness Flag** | Highlights strategies that diverge significantly |
| **Best Strategy Ranking** | Composite score: detail × (1 − outlier rate) |
| **Prompt Inspector** | Exact prompts sent — learn by reading the real payloads |
| **Bad Prompt Demo** | Shows what a vague prompt produces and *why* it fails |
| **Dark Lab UI** | Syne + JetBrains Mono, #0d0f14 background, green accents |

---

## Project Structure

```
prompt_testing_lab/
│
├── app.py                  # Entry point — page config, tabs, wiring
├── config.py               # Constants: model, strategy list, colours, thresholds
├── prompts.py              # Pure prompt-builder functions (no API, no Streamlit)
├── api_client.py           # All Anthropic API interactions
├── analytics.py            # Scoring, outlier detection, ranking (pure Python)
│
├── ui/
│   ├── __init__.py
│   ├── styles.py           # Global CSS injection
│   ├── sidebar.py          # Sidebar component → returns api_key
│   ├── tab_run.py          # "Run Lab" tab
│   ├── tab_results.py      # "Results" tab
│   ├── tab_analytics.py    # "Analytics" tab
│   └── tab_inspector.py    # "Prompt Inspector" tab
│
├── requirements.txt
└── README.md
```

### Dependency map

```
app.py
  ├── ui/styles.py
  ├── ui/sidebar.py           → config
  ├── ui/tab_run.py           → config, api_client, prompts
  ├── ui/tab_results.py       → config, analytics
  ├── ui/tab_analytics.py     → config, analytics
  └── ui/tab_inspector.py     → config, prompts

api_client.py                 → config, prompts, analytics
analytics.py                  → config
prompts.py                    (no internal deps)
config.py                     (no internal deps)
```

---

## Quick Start

### 1. Clone / download

```bash
git clone https://github.com/your-org/prompt-testing-lab.git
cd prompt-testing-lab
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

### 5. Use the lab

1. Paste your **groq API key** into the sidebar (`sk-ant-...`)
2. Open the **Run Lab** tab
3. Enter a task description, e.g.:
   > classify customer support messages as: billing, shipping, returns, or general
4. Enter 3–5 test inputs, one per line:
   ```
   My package never arrived and it's been 2 weeks.
   I was charged twice for the same order.
   How do I return a damaged item?
   ```
5. Click **⚡ Run Comparison**
6. Explore **Results**, **Analytics**, and **Prompt Inspector** tabs

---

## Module Reference

### `config.py`

| Symbol | Type | Description |
|---|---|---|
| `MODEL` | `str` | Claude model string used for all calls |
| `MAX_TOKENS` | `int` | Max tokens per completion |
| `STRATEGIES` | `list[str]` | Ordered list of strategy names |
| `STRATEGY_BADGE` | `dict` | CSS badge class per strategy |
| `STRATEGY_COLORS` | `dict` | Hex colour per strategy (charts) |
| `OUTLIER_THRESHOLD` | `float` | Jaccard threshold for flagging outliers |
| `CELL_TRUNCATE` | `int` | Character limit for table cell display |

---

### `prompts.py`

| Function | Signature | Description |
|---|---|---|
| `build_zero_shot` | `(task, inp) → str` | Bare task + input prompt |
| `build_few_shot` | `(task, inp, examples) → str` | 3-example scaffold |
| `build_chain_of_thought` | `(task, inp) → str` | Step-by-step reasoning prompt |
| `build_role_prompting` | `(task, inp) → str` | Expert persona + task |
| `build_bad_prompt` | `(task, inp) → str` | Deliberately vague prompt |
| `build_prompt` | `(strategy, task, inp, examples?) → str` | Dispatcher |

All functions are **pure** — no side effects, no API calls. Safe to unit-test.

---

### `api_client.py`

| Function | Description |
|---|---|
| `make_client(api_key)` | Returns an `anthropic.Anthropic` client |
| `call_claude(client, prompt, system?)` | Single completion → `str` |
| `generate_few_shot_examples(client, task)` | Returns 3 `{input, output}` dicts |
| `run_comparison(api_key, task, inputs, progress_callback?)` | Full 4×N run → `results` dict |
| `run_bad_prompt(api_key, task, inputs)` | Bad prompt on all inputs → `list[str]` |

The `results` dict schema:

```python
{
    "task":              str,
    "inputs":            list[str],
    "responses":         dict[strategy, list[str]],   # [per_input]
    "prompts":           dict[strategy, list[str]],   # [per_input]
    "few_shot_examples": list[{"input": str, "output": str}],
    "scores":            list[float],    # consistency per row
    "lengths":           dict[strategy, float],       # avg word count
    "outliers":          list[str | None],            # per row
}
```

---

### `analytics.py`

| Function | Description |
|---|---|
| `tokenize(text)` | Word-level token set |
| `avg_word_length(responses)` | Average word count over a list |
| `jaccard(a, b)` | Jaccard similarity between two strings |
| `pairwise_overlap(responses)` | Mean Jaccard over all pairs |
| `find_outlier(responses, threshold?)` | Least-consistent strategy name or `None` |
| `rank_strategies(results)` | Returns `(best, worst)` strategy names |
| `score_to_color(score)` | `float → hex` traffic-light colour |

---

### `ui/` modules

| Module | Exported function | Description |
|---|---|---|
| `styles.py` | `inject_styles()` | Injects global CSS |
| `sidebar.py` | `render_sidebar() → str` | Renders sidebar, returns API key |
| `tab_run.py` | `render_tab_run(api_key)` | Run Lab tab |
| `tab_results.py` | `render_tab_results()` | Results tab |
| `tab_analytics.py` | `render_tab_analytics()` | Analytics tab |
| `tab_inspector.py` | `render_tab_inspector()` | Prompt Inspector tab |

All UI modules read from `st.session_state["results"]` (set by `tab_run`).

---

## Prompt Strategies

### Zero-Shot
```
Task: {task}
Input: {input}
Answer:
```
Best for: well-defined tasks where Claude's training covers the domain.

### Few-Shot
```
Task: {task}

Examples:
  Input: {ex1_in} → Output: {ex1_out}
  Input: {ex2_in} → Output: {ex2_out}
  Input: {ex3_in} → Output: {ex3_out}

Now answer:
Input: {input}
Output:
```
The 3 examples are **auto-generated** by a separate Claude call before the main run.
Best for: classification, formatting, style-matching tasks.

### Chain-of-Thought
```
Task: {task}
Input: {input}

Let's think step by step before giving a final answer.

Step-by-step reasoning:
Final Answer:
```
Best for: multi-step reasoning, ambiguous inputs, explanation tasks.

### Role Prompting
```
You are an expert specializing in {task}. You have 20 years of experience
and always give precise, accurate answers.

Task: {task}
Input: {input}
Expert Answer:
```
Best for: tasks that benefit from authoritative, domain-specific framing.

### Bad Prompt (Demo)
```
do the thing with {input}
```
Illustrates why vague prompts produce unreliable, generic responses.

---

## Analytics Explained

### Consistency Score
Calculated as the **mean pairwise Jaccard similarity** across the 4 strategy responses for each input row:

```
Jaccard(A, B) = |words(A) ∩ words(B)| / |words(A) ∪ words(B)|

consistency = mean(Jaccard(A,B), Jaccard(A,C), Jaccard(A,D),
                   Jaccard(B,C), Jaccard(B,D), Jaccard(C,D))
```

- **≥ 60%** → green (strategies agree)
- **35–60%** → amber (moderate variation)
- **< 35%** → red (strategies diverge significantly)

### Outlier / Robustness Flag
For each row, the strategy whose response overlaps least with the others is computed. If its mean Jaccard similarity falls below `OUTLIER_THRESHOLD` (default **0.30**), it is flagged as an outlier.

### Strategy Ranking
```
score(strategy) = normalised_avg_length × (1 − outlier_rate)
```
- `normalised_avg_length` = avg word count / max avg word count across strategies
- `outlier_rate` = fraction of rows where this strategy was flagged

Best = highest composite score. Worst = lowest.

---

## Configuration

All tuneable values live in `config.py`. No need to touch other files:

```python
MODEL           = "llama-3.3-70b-versatile"   # swap to any Claude model
MAX_TOKENS      = 512                           # increase for longer outputs
OUTLIER_THRESHOLD = 0.30                        # lower = fewer flags
CELL_TRUNCATE   = 120                           # chars shown in results table
```

---

## Extending the Lab

### Add a new strategy

1. Write a builder in `prompts.py`:
   ```python
   def build_my_strategy(task: str, inp: str) -> str:
       return f"[my custom prompt] {task} ... {inp}"
   ```

2. Add a case to the `build_prompt` dispatcher in `prompts.py`.

3. Add the strategy name to `STRATEGIES` in `config.py`.

4. Add a badge class and colour to `STRATEGY_BADGE` / `STRATEGY_COLORS`.

5. Add the CSS for the new badge to `ui/styles.py`.

That's it — all tab rendering loops over `STRATEGIES` automatically.

### Swap the model

Change `MODEL` in `config.py`. The string must be a valid Anthropic model identifier.

### Persist results across sessions

Replace `st.session_state` storage in `ui/tab_run.py` with a local SQLite write (or any store), and load from it in `ui/tab_results.py`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `AuthenticationError` | Check your API key in the sidebar — must start with `sk-ant-` |
| `RateLimitError` | Reduce the number of test inputs, or add a longer sleep in `api_client.py` |
| JSON parse error in few-shot generation | The fallback dummy examples are used automatically; results still work |
| Blank results after switching tabs | Results live in `st.session_state` — they persist as long as the browser tab is open |
| Module not found | Ensure you're running `streamlit run app.py` from inside the `prompt_testing_lab/` directory |

---

## License

MIT — use freely, attribution appreciated.
