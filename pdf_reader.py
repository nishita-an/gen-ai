from pypdf import PdfReader # type: ignore

def read_resume_pdf(path: str) -> str:
    reader = PdfReader(path)
    text = ""

    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"

    return text.strip()

pdf_path = r"D:\Project\Interview_agent\NishitaNaragund.pdf"  # make sure this path is correct

text = read_resume_pdf(pdf_path)

print("===== EXTRACTED TEXT START =====")
print(text)
print("===== EXTRACTED TEXT END =====")
print("\nTotal characters extracted:", len(text))
