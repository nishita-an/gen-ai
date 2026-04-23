from src.data_loader import load_csv, load_pdf, load_txt
from src.chunking import chunk_docs
from src.embeddings import load_embeddings
from src.vector_store import create_vector_store
from src.bm25_retriever import BM25Retriever
from src.hybrid_retriever import HybridRetriever
from src.rag_pipeline import generate_answer
from src.config import DATA_PATH


# ── Load all document types ──────────────────────────────────────────
docs = []

csv_docs = load_csv(DATA_PATH)
print(f"  CSV documents loaded  : {len(csv_docs)}")
docs.extend(csv_docs)

txt_docs = load_txt(DATA_PATH)
print(f"  TXT documents loaded  : {len(txt_docs)}")
docs.extend(txt_docs)

pdf_docs = load_pdf(DATA_PATH)
print(f"  PDF documents loaded  : {len(pdf_docs)}")
docs.extend(pdf_docs)

print(f"\nTotal documents loaded  : {len(docs)}")

if len(docs) == 0:
    print("\n[ERROR] No documents found. Check DATA_PATH in config.py")
    exit(1)

# ── Chunk → Embed → Index ────────────────────────────────────────────
chunks = chunk_docs(docs)
print(f"Total chunks created    : {len(chunks)}")

embeddings = load_embeddings()

vector_db = create_vector_store(chunks, embeddings)

bm25 = BM25Retriever(chunks)

retriever = HybridRetriever(vector_db, bm25)

print("\nRAG pipeline ready. Type 'exit' to quit.\n")

# ── Query loop ───────────────────────────────────────────────────────
while True:

    query = input("Question: ").strip()

    if query.lower() in ("exit", "quit"):
        print("Goodbye!")
        break

    if not query:
        continue

    answer = generate_answer(query, retriever)

    print("\nAnswer:")
    print(answer)
    print()