"""Input module: reads text out of uploaded .txt / .docx / .pdf files."""
import docx2txt
from PyPDF2 import PdfReader


def read_text_file(uploaded_file) -> str:
    return uploaded_file.read().decode("utf-8", errors="ignore")


def read_docx_file(uploaded_file) -> str:
    return docx2txt.process(uploaded_file) or ""


def read_pdf_file(uploaded_file) -> str:
    reader = PdfReader(uploaded_file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def get_text_from_file(uploaded_file) -> str:
    """Dispatch on Streamlit's UploadedFile.type."""
    if uploaded_file is None:
        return ""
    mime = uploaded_file.type
    if mime == "text/plain":
        return read_text_file(uploaded_file)
    if mime == "application/pdf":
        return read_pdf_file(uploaded_file)
    if mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return read_docx_file(uploaded_file)
    # Fallback: guess from the extension.
    name = uploaded_file.name.lower()
    if name.endswith(".docx"):
        return read_docx_file(uploaded_file)
    if name.endswith(".pdf"):
        return read_pdf_file(uploaded_file)
    return read_text_file(uploaded_file)
