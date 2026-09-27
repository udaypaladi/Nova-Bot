from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from parsing import parse_folder

SOURCE_FOLDER = "documents"
VECTOR_DB_PATH = "vector_store"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def chunk_documents(documents, embeddings):
    splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
    )
    chunks = splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")
    return chunks

def build_vector_store(chunks, embeddings):
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_PATH,
    )
    print(f"Vector store saved to: {VECTOR_DB_PATH}")
    return vector_store

if __name__ == "__main__":
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    documents= parse_folder(SOURCE_FOLDER)
    chunks = chunk_documents(documents, embeddings)
    build_vector_store(chunks, embeddings)
