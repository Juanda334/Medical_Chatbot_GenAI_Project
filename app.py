import os
import boto3
from src.prompt import *
from dotenv import load_dotenv
from langchain_aws import ChatBedrock
from src.helper import create_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

embeddings = create_embeddings()

index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name = index_name,
    embedding = embeddings
)

retriever = docsearch.as_retriever(search_type = "similarity", search_kwargs = {"k":3})

# Bedrock Clients
bedrock = boto3.client(
    service_name = 'bedrock-runtime',
    region_name = 'us-east-1'
)

# Defining the LLM model
def get_llama3_llm():
    llama3_llm = ChatBedrock(
        client = bedrock,
        model_id = 'us.meta.llama3-2-1b-instruct-v1:0',
        model_kwargs = {
            'max_gen_len': 2048,
            'temperature': 0.0,
            'top_p': 0.5
        }
    )
    return llama3_llm

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", prompt_template),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(get_llama3_llm(), prompt = PROMPT)
rag_chain = create_retrieval_chain(retriever = retriever, combine_docs_chain = question_answer_chain)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    result = rag_chain.invoke({"input": input})
    print("Response:", result["answer"])
    return str(result["answer"])

if __name__ == "__main__":
    app.run(host = "0.0.0.0", port = 8080, debug=True)