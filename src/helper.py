from typing import List
from langchain_core.documents import Document
#from langchain_huggingface import 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters  import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

# Load the pdf documents
def load_pdf_files(directory):
    loader = DirectoryLoader(directory, glob = "*.pdf", loader_cls = PyPDFLoader)
    documents = loader.load()
    return documents

# Filter the documents to keep only essential metadata and content
def filter_documents(documents: List[Document]) -> List[Document]:
    """Filter documents to retain only essential metadata.

    Args:
        documents (List[Document]): List of Document objects to be filtered.

    Returns:
        List[Document]: List of filtered Document objects with minimal metadata.
    """
    minimal_docs: List[Document] = []
    for doc in documents:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs

# Split the documents into smaller chunks
def text_splitter(minimal_docs):
    """Split documents into smaller text chunks.

    Args:
        minimal_docs (List[Document]): List of Document objects with minimal metadata.

    Returns:
        List[Document]: List of smaller text chunks as Document objects.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )
    text_chunks = text_splitter.split_documents(minimal_docs)
    return text_chunks

# Create the embedding model from HuggingFace
def create_embeddings():
    """Create embeddings for a list of text chunks.

    Args:
        text_chunks (List[Document]): List of Document objects representing text chunks.

    Returns:
        Embedding object.
    """
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    return embeddings

embeddings = create_embeddings()