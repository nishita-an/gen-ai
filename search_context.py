from tavily import TavilyClient
from config import TAVILY_API_KEY
from grok_llm import call_groq

# -------------------------------
# Groq: Job Description → Search Query
# -------------------------------

def jd_to_search_query(job_description: str) -> str:
    prompt = f"""
You are an expert technical recruiter.

Convert the following job description into a concise
Google-style search query for finding technical interview questions.

Rules:
- Max 12 words
- Use only keywords
- No punctuation
- No explanations
- Output ONLY the query text

Job Description:
{job_description}
"""

    query = call_groq(prompt).strip()
    return query


# -------------------------------
# Tavily: Fetch Interview Context
# -------------------------------

def fetch_interview_context(job_description: str) -> str:
    client = TavilyClient(api_key=TAVILY_API_KEY)

    # Step 1: JD → search query
    query = jd_to_search_query(job_description)
    print("\n[DEBUG] Tavily search query:", query)

    # Step 2: Tavily search
    response = client.search(
        query=query,
        max_results=5,
        search_depth="basic"
    )

    context = ""
    for item in response["results"]:
        context += item["content"] + "\n"

    return context.strip()



