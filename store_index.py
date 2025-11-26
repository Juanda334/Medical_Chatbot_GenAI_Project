import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from src.helper import load_pdf_files, filter_documents, text_splitter, create_embeddings

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

extracted_docs = load_pdf_files("data/")
filter_docs = filter_documents(documents = extracted_docs)
docs_chucks = text_splitter(minimal_docs = filter_docs)
embeddings = create_embeddings()

pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key = pinecone_api_key)

index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name = index_name,
        dimension = 384,
        metric = "cosine",
        spec = ServerlessSpec(cloud = "aws", region = "us-east-1")
    )
    
index = pc.Index(index_name)

docsearch = PineconeVectorStore.from_documents(
    documents = docs_chucks,
    embedding = embeddings,
    index_name = index_name
)