"""
config.py
─────────
Central configuration: model name, strategy list, colours.
Import from this file everywhere — never scatter magic strings.
"""

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL: str = "llama-3.3-70b-versatile"
MAX_TOKENS: int = 512

# ── Strategies ────────────────────────────────────────────────────────────────
STRATEGIES: list[str] = [
    "Zero-Shot",
    "Few-Shot",
    "Chain-of-Thought",
    "Role Prompting",
]

# Badge CSS class per strategy (maps to classes defined in ui/styles.py)
STRATEGY_BADGE: dict[str, str] = {
    "Zero-Shot":        "badge-zs",
    "Few-Shot":         "badge-fs",
    "Chain-of-Thought": "badge-cot",
    "Role Prompting":   "badge-rp",
}

# Altair / chart colours per strategy
STRATEGY_COLORS: dict[str, str] = {
    "Zero-Shot":        "#60a5fa",
    "Few-Shot":         "#4ade80",
    "Chain-of-Thought": "#fb923c",
    "Role Prompting":   "#c084fc",
}

# ── Thresholds ────────────────────────────────────────────────────────────────
# Jaccard similarity below this → strategy flagged as outlier
OUTLIER_THRESHOLD: float = 0.30

# Truncation length for cells in the results table
CELL_TRUNCATE: int = 120
