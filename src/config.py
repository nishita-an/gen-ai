DATA_PATH = "./data"

CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 100

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Increased from 5 → 10 so multi-table questions get broader coverage
TOP_K    = 10
BM25_K   = 10

HYBRID_ALPHA = 0.6