from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

def load_pdf(file_path: str):
    return PyPDFLoader(file_path).load()

def load_docx(file_path: str):
    return Docx2txtLoader(file_path).load()

def load_txt(file_path: str):
    return TextLoader(file_path).load()

LOADER_MAP = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".txt": load_txt
}