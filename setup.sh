#!/usr/bin/env bash
# ============================================================
# setup.sh — One-shot environment setup for Multilingual RAG
# Usage: bash setup.sh
# ============================================================

set -euo pipefail

PYTHON=${PYTHON:-python3}
VENV_DIR="venv"

echo "════════════════════════════════════════════════════════"
echo "  Multilingual PDF RAG — Environment Setup"
echo "════════════════════════════════════════════════════════"

# ── 1. Python version check ──────────────────────────────────
PY_VERSION=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
REQUIRED="3.10"
if [[ "$(echo -e "$PY_VERSION\n$REQUIRED" | sort -V | head -1)" != "$REQUIRED" ]]; then
    echo "❌ Python $REQUIRED+ required (found $PY_VERSION). Please install Python $REQUIRED or higher."
    exit 1
fi
echo "✅ Python $PY_VERSION detected."

# ── 2. Create virtual environment ────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "→ Creating virtual environment in ./$VENV_DIR …"
    $PYTHON -m venv "$VENV_DIR"
fi

# Activate
source "$VENV_DIR/bin/activate" 2>/dev/null || source "$VENV_DIR/Scripts/activate" 2>/dev/null
echo "✅ Virtual environment activated."

# ── 3. Upgrade pip ───────────────────────────────────────────
pip install --upgrade pip --quiet

# ── 4. Install dependencies ──────────────────────────────────
echo "→ Installing dependencies (this may take a few minutes)…"
pip install -r requirements.txt

# ── 5. Download NLTK data ────────────────────────────────────
echo "→ Downloading NLTK tokenizer data…"
python -c "
import nltk
for pkg in ['punkt', 'punkt_tab']:
    try:
        nltk.download(pkg, quiet=True)
        print(f'  ✅ NLTK {pkg} ready.')
    except Exception as e:
        print(f'  ⚠️  Could not download {pkg}: {e}')
"

# ── 6. Create directories ────────────────────────────────────
mkdir -p pdfs chroma_db
echo "✅ Created ./pdfs and ./chroma_db directories."

# ── 7. Create .env from example ──────────────────────────────
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "✅ Created .env from .env.example"
    echo ""
    echo "⚠️  ACTION REQUIRED: Open .env and set your GROQ_API_KEY."
else
    echo "ℹ️  .env already exists — skipping."
fi

# ── 8. Pre-download the embedding model ──────────────────────
echo "→ Pre-downloading multilingual embedding model…"
python -c "
from sentence_transformers import SentenceTransformer
print('  Downloading paraphrase-multilingual-mpnet-base-v2 …')
m = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
v = m.encode(['hello'], show_progress_bar=False)
print(f'  ✅ Model ready — embedding dimension: {v.shape[1]}')
"

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Setup complete! Next steps:"
echo ""
echo "  1. Add your PDFs to ./pdfs/"
echo "  2. Set GROQ_API_KEY in .env"
echo "  3. Run: source venv/bin/activate"
echo "  4. Run: python ingest.py"
echo "  5. Run: streamlit run app.py"
echo "════════════════════════════════════════════════════════"
