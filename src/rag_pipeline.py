import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def generate_answer(query, retriever):

    docs = retriever.retrieve(query)

    # Label each chunk with its source file so the LLM knows which table it's from
    context_parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        dtype  = doc.metadata.get("type", "")
        label  = f"[{i}] Source: {source}" + (f" ({dtype})" if dtype else "")
        context_parts.append(f"{label}\n{doc.page_content}")

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are a finance data assistant. You have access to structured data from multiple tables and policy documents.

IMPORTANT RULES:
- Answer using ONLY the context provided below.
- The context contains data from multiple CSV tables (invoices, payments, vendors, expenses) and policy documents.
- To answer multi-step questions, JOIN data across tables using matching IDs (e.g. vendor_id links invoices → vendors; invoice_id links invoices → payments).
- For aggregation questions (totals, counts), compute the answer from the data in context.
- Show your reasoning step by step before giving the final answer.
- If the required data is genuinely not present, say: "Not found in knowledge base."

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return completion.choices[0].message.content