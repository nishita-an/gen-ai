from pdf_reader import read_resume_pdf
from search_context import fetch_interview_context 
from qgen import generate_questions
from voice_input import get_voice_answer
from evaluator import evaluate_answer


def main():
    resume_text = read_resume_pdf("NishitaNaragund.pdf")

    job_description = input("Paste job description:\n")

    tavily_context = fetch_interview_context(job_description)

    questions = generate_questions(
        resume_text,
        job_description,
        tavily_context
    )

    print("\n--- INTERVIEW START ---\n")

    for q in questions:
        print(f"\nQUESTION: {q}")
        answer = get_voice_answer()
        print("Your Answer:", answer)

        feedback = evaluate_answer(q, answer)
        print("\nFEEDBACK:\n", feedback)


if __name__ == "__main__":
    main()
