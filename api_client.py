"""
api_client.py  (Groq version)
"""

import re
import json
import time
from groq import Groq

from config import MODEL, MAX_TOKENS, STRATEGIES
from prompts import build_prompt, build_bad_prompt


def make_client(api_key: str) -> Groq:
    return Groq(api_key=api_key)


def call_claude(client: Groq, prompt: str, system: str = "") -> str:
    """Same function name kept so nothing else needs to change."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=messages,
    )
    return response.choices[0].message.content.strip()


def generate_few_shot_examples(client: Groq, task: str) -> list[dict]:
    meta_prompt = (
        f'You are a prompt engineer. Generate exactly 3 short, realistic '
        f'input/output examples for this task: "{task}".\n\n'
        f'Return ONLY a JSON array (no markdown, no explanation):\n'
        f'[\n'
        f'  {{"input": "...", "output": "..."}},\n'
        f'  {{"input": "...", "output": "..."}},\n'
        f'  {{"input": "...", "output": "..."}}\n'
        f']'
    )
    raw = call_claude(client, meta_prompt)
    raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
    try:
        return json.loads(raw)[:3]
    except Exception:
        return [
            {"input": "Sample input A", "output": "Sample output A"},
            {"input": "Sample input B", "output": "Sample output B"},
            {"input": "Sample input C", "output": "Sample output C"},
        ]


def run_comparison(api_key, task, test_inputs, progress_callback=None):
    from analytics import pairwise_overlap, find_outlier, avg_word_length

    client = make_client(api_key)
    results = {
        "task": task, "inputs": test_inputs,
        "responses": {s: [] for s in STRATEGIES},
        "prompts":   {s: [] for s in STRATEGIES},
        "few_shot_examples": [], "scores": [],
        "lengths": {}, "outliers": [],
    }

    results["few_shot_examples"] = generate_few_shot_examples(client, task)
    total = len(STRATEGIES) * len(test_inputs)
    step  = 0

    for inp in test_inputs:
        row_responses = {}
        for strategy in STRATEGIES:
            prompt = build_prompt(strategy, task, inp, results["few_shot_examples"])
            results["prompts"][strategy].append(prompt)
            try:
                resp = call_claude(client, prompt)
            except Exception as exc:
                resp = f"⚠️ Error: {exc}"
            results["responses"][strategy].append(resp)
            row_responses[strategy] = resp
            step += 1
            if progress_callback:
                progress_callback(step, total, f"Running {strategy}…")
            time.sleep(0.05)

        results["scores"].append(pairwise_overlap(list(row_responses.values())))
        results["outliers"].append(find_outlier(row_responses))

    for strategy in STRATEGIES:
        results["lengths"][strategy] = avg_word_length(results["responses"][strategy])

    return results


def run_bad_prompt(api_key, task, test_inputs):
    client = make_client(api_key)
    responses = []
    for inp in test_inputs:
        try:
            responses.append(call_claude(client, build_bad_prompt(task, inp)))
        except Exception as exc:
            responses.append(f"⚠️ Error: {exc}")
    return responses