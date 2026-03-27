"""
analytics.py
────────────
Pure analytical functions: no Streamlit, no API calls.
All functions take plain Python data structures and return plain results,
making them independently testable.
"""

import re
from collections import Counter
from config import STRATEGIES


# ── Text Utilities ────────────────────────────────────────────────────────────

def tokenize(text: str) -> set[str]:
    """Case-fold and extract word tokens for overlap comparison."""
    return set(re.findall(r'\b\w+\b', text.lower()))


def avg_word_length(responses: list[str]) -> float:
    """Return the average word count across a list of response strings."""
    if not responses:
        return 0.0
    return sum(len(r.split()) for r in responses) / len(responses)


# ── Similarity / Consistency ──────────────────────────────────────────────────

def jaccard(a: str, b: str) -> float:
    """Jaccard similarity between two text strings (word-level)."""
    ta, tb = tokenize(a), tokenize(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def pairwise_overlap(responses: list[str]) -> float:
    """
    Average Jaccard similarity across all unique pairs of responses.
    Returns a score in [0, 1]:
      0 = completely different words
      1 = identical content
    """
    if len(responses) < 2:
        return 1.0
    scores: list[float] = []
    for i in range(len(responses)):
        for j in range(i + 1, len(responses)):
            scores.append(jaccard(responses[i], responses[j]))
    return sum(scores) / len(scores)


# ── Outlier Detection ─────────────────────────────────────────────────────────

def find_outlier(responses: dict[str, str], threshold: float | None = None) -> str | None:
    """
    Identify the strategy whose response overlaps least with the others.
    Returns that strategy's name if its mean overlap is below `threshold`,
    otherwise returns None (no significant outlier).

    Args:
        responses:  {strategy_name: response_text}
        threshold:  Jaccard threshold below which a strategy is flagged.
                    Defaults to config.OUTLIER_THRESHOLD.
    """
    from config import OUTLIER_THRESHOLD  # avoid top-level circular imports
    if threshold is None:
        threshold = OUTLIER_THRESHOLD

    strats = list(responses.keys())
    if len(strats) < 2:
        return None

    mean_overlaps: dict[str, float] = {}
    for s in strats:
        others = [responses[o] for o in strats if o != s]
        scores = [
            jaccard(responses[s], other)
            for other in others
        ]
        mean_overlaps[s] = sum(scores) / len(scores)

    min_strat = min(mean_overlaps, key=mean_overlaps.get)
    if mean_overlaps[min_strat] < threshold:
        return min_strat
    return None


# ── Strategy Ranking ──────────────────────────────────────────────────────────

def rank_strategies(results: dict) -> tuple[str, str]:
    """
    Compute a composite score for each strategy and return (best, worst).

    Scoring formula:
        score = normalised_avg_length × (1 − outlier_rate)

    - More detailed responses → higher length score
    - Appearing as outlier → penalty proportional to how often
    """
    outlier_counts = Counter(o for o in results["outliers"] if o is not None)
    n_inputs = max(len(results["inputs"]), 1)
    max_len = max(results["lengths"].values(), default=1)

    strategy_scores: dict[str, float] = {}
    for strat in STRATEGIES:
        length_score     = results["lengths"].get(strat, 0) / max_len
        outlier_penalty  = outlier_counts.get(strat, 0) / n_inputs
        strategy_scores[strat] = length_score * (1 - outlier_penalty)

    best  = max(strategy_scores, key=strategy_scores.get)
    worst = min(strategy_scores, key=strategy_scores.get)
    return best, worst


# ── Colour Helpers ────────────────────────────────────────────────────────────

def score_to_color(score: float) -> str:
    """
    Map a 0-1 consistency score to a traffic-light hex colour.
    ≥ 0.6 → green, ≥ 0.35 → amber, else → red.
    """
    if score >= 0.60:
        return "#1e8f5e"
    elif score >= 0.35:
        return "#f59e0b"
    else:
        return "#ef4444"
