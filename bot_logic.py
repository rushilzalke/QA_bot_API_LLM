import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# NEW IMPORTS FOR 2026 STANDARDS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import google.generativeai as genai
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# List available embedding models
for m in genai.list_models():
    if 'embedContent' in m.supported_generation_methods:
        print(f"Supported Embedding Model: {m.name}")

def process_pdf(pdf_path):
    # Load the PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    # Create Embeddings & Store in ChromaDB
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return vectorstore

def get_answer(vectorstore, question):
    llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.3)
    # 1. Define the System Prompt
    system_prompt = (
        "You are a helpful assistant. Use the following pieces of "
        "retrieved context to answer the user's question. "
        "If you don't know the answer, just say you don't know."
        "\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    # 2. Create the "Combine Documents" chain (How to answer)
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    # 3. Create the "Retrieval" chain (How to find + How to answer)
    rag_chain = create_retrieval_chain(vectorstore.as_retriever(), combine_docs_chain)
    # 4. Invoke the chain
    response = rag_chain.invoke({"input": question})
    # In the new chain, the answer is inside the "answer" key
    return response["answer"]
