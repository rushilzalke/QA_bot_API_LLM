import os
import shutil
import requests
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# ── Models ────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # local, free, no key needed

# WHY THIS MODEL + PROVIDER:
# HF's "hf-inference" free provider has a very small, constantly-changing
# whitelist — flan-t5 and Phi-3 were both dropped, causing 400 errors.
# "novita" is a stable HF router provider that reliably supports
# Meta-Llama-3.1-8B-Instruct on the free tier via OpenAI-compatible API.
LLM_MODEL  = "meta-llama/Meta-Llama-3.1-8B-Instruct"
PROVIDER   = "novita"
CHROMA_DIR = "./chroma_db_hf"

HF_API_URL = f"https://router.huggingface.co/{PROVIDER}/v1/chat/completions"
HF_HEADERS = {
    "Authorization": f"Bearer {os.getenv('HUGGINGFACEHUB_API_TOKEN')}",
    "Content-Type":  "application/json",
}


# ── ChromaDB safe create ──────────────────────────────────────────────────────
def _safe_chroma_create(splits, embeddings, persist_directory):
    """Auto-wipe corrupted ChromaDB folder and recreate if tenant error occurs."""
    try:
        return Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=persist_directory,
        )
    except ValueError as e:
        if "tenant" in str(e).lower():
            if os.path.exists(persist_directory):
                shutil.rmtree(persist_directory)
            return Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=persist_directory,
            )
        raise


# ── PDF Processing ────────────────────────────────────────────────────────────
def process_pdf(pdf_path: str):
    """Load PDF → split → embed → store in ChromaDB. Returns vectorstore."""
    loader = PyPDFLoader(pdf_path)
    docs   = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits   = splitter.split_documents(docs)

    embeddings  = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = _safe_chroma_create(splits, embeddings, CHROMA_DIR)
    return vectorstore


# ── Q&A ───────────────────────────────────────────────────────────────────────
def get_answer(vectorstore, question: str) -> str:
    """Retrieve relevant chunks and answer using Llama-3.1 via HF novita provider."""

    # 1. Retrieve top-4 relevant chunks
    retriever     = vectorstore.as_retriever(search_kwargs={"k": 4})
    relevant_docs = retriever.invoke(question)
    context       = "\n\n".join(doc.page_content for doc in relevant_docs)

    # 2. Build OpenAI-style chat payload
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. "
                    "Answer the question using ONLY the context provided below. "
                    "If the answer is not in the context, say 'I don't know'.\n\n"
                    f"Context:\n{context}"
                ),
            },
            {
                "role": "user",
                "content": question,
            },
        ],
        "max_tokens":  512,
        "temperature": 0.3,
    }

    # 3. POST to HF router via novita provider
    response = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload, timeout=60)

    if response.status_code != 200:
        return f"HuggingFace API error {response.status_code}: {response.text}"

    result = response.json()

    # 4. Parse OpenAI-style response
    try:
        answer = result["choices"][0]["message"]["content"].strip()
        return answer if answer else "I don't know based on the provided context."
    except (KeyError, IndexError):
        return f"Unexpected response format: {result}"