from grok_llm import call_groq

def generate_questions(resume: str, jd: str, tavily_context: str) -> list:
    prompt = f"""
You are a senior technical interviewer.

Resume:
{resume[:1500]}

Job Description:
{jd}

Interview Insights:
{tavily_context[:1500]}

Generate:
- 5 technical questions
- 2 system design questions
- 1 behavioral question

Return ONLY a numbered list.
"""

    response = call_groq(prompt)

    questions = [
        q.strip() for q in response.split("\n") if q.strip()
    ]

    return questions
