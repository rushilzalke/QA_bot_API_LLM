import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

try:
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ImportError:
    from langchain_classic.chains import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

CHROMA_DIR = "./chroma_db"


def _safe_chroma_create(splits, embeddings, persist_directory):
    """
    Create ChromaDB vectorstore. If the existing DB folder is corrupted
    (old format / missing tenant), wipe it and start fresh automatically.

    Root cause: ChromaDB v0.4+ changed its internal storage format.
    Old `chroma_db` folders created with earlier versions don't have the
    `default_tenant` structure the new client expects → ValueError on open.
    Solution: delete the stale folder and recreate it cleanly.
    """
    try:
        return Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=persist_directory,
        )
    except ValueError as e:
        if "tenant" in str(e).lower():
            # Stale / incompatible DB — wipe and retry once
            if os.path.exists(persist_directory):
                shutil.rmtree(persist_directory)
            return Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=persist_directory,
            )
        raise  # re-raise if it's a different ValueError


def process_pdf(pdf_path: str):
    """Load PDF → split → embed with Google embeddings → store in ChromaDB."""
    loader = PyPDFLoader(pdf_path)
    docs   = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits        = text_splitter.split_documents(docs)

    embeddings  = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = _safe_chroma_create(splits, embeddings, CHROMA_DIR)
    return vectorstore


def get_answer(vectorstore, question: str) -> str:
    """Retrieve relevant chunks and answer using Google Gemini."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0.3)

    system_prompt = (
        "You are a helpful assistant. Use the following pieces of "
        "retrieved context to answer the user's question. "
        "If you don't know the answer, just say you don't know."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain          = create_retrieval_chain(vectorstore.as_retriever(), combine_docs_chain)
    response           = rag_chain.invoke({"input": question})

    return response["answer"]