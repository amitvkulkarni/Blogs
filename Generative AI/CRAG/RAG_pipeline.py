"""
rag_pipeline.py
---------------
Standard Retrieval-Augmented Generation (RAG) using Azure OpenAI + LangChain.

Pipeline:
  1. Load documents from ./documents/
  2. Split into chunks and embed with Azure OpenAI Embeddings
  3. Store in a FAISS vector index
  4. For a query: retrieve top-K chunks -> pass to LLM -> return answer
"""

import os
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ── Load environment variables ────────────────────────────────────────────────
load_dotenv()

# ── Azure OpenAI Configuration ────────────────────────────────────────────────
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")
EMBEDDING_DEPLOYMENT = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002"
)


# ── RAG Components ─────────────────────────────────────────────────────────────


def load_documents(docs_dir: str = "./documents") -> list:
    """Load all .txt files from the documents directory."""
    loader = DirectoryLoader(docs_dir, glob="**/*.txt", loader_cls=TextLoader)
    docs = loader.load()
    print(f"[RAG] Loaded {len(docs)} document(s) from '{docs_dir}'")
    return docs


def build_vector_store(docs: list) -> FAISS:
    """Split documents into chunks and build a FAISS vector store."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"[RAG] Split into {len(chunks)} chunks")

    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=EMBEDDING_DEPLOYMENT,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    print("[RAG] Vector store built (FAISS)")
    return vector_store


def build_rag_chain(vector_store: FAISS) -> RetrievalQA:
    """Create a simple RetrievalQA chain (standard RAG)."""
    llm = AzureChatOpenAI(
        azure_deployment=CHAT_DEPLOYMENT,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
    )

    prompt_template = """You are a helpful engineering and compliance assistant.
Use the following retrieved context to answer the question.
If the context does not contain enough information, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:"""

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return chain


def run_rag(query: str, chain: RetrievalQA) -> dict:
    """Run a query through the standard RAG pipeline and return result."""
    print(f"\n[RAG] Query: {query}")
    result = chain.invoke({"query": query})

    sources = list(
        {
            os.path.basename(doc.metadata.get("source", "unknown"))
            for doc in result.get("source_documents", [])
        }
    )

    return {
        "query": query,
        "answer": result["result"],
        "sources": sources,
        "pipeline": "Standard RAG",
    }


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    docs = load_documents()
    vector_store = build_vector_store(docs)
    rag_chain = build_rag_chain(vector_store)

    # Sample queries aligned with the blog content
    queries = [
        "Does the vendor fire suppression system cover all three cities including Seattle?"
        # "What seismic bracing requirements apply to structural beams installed in Seattle?",
        # "What HVAC efficiency minimum is required for chillers in Chicago?",
    ]

    for q in queries:
        output = run_rag(q, rag_chain)
        print("\n" + "=" * 70)
        print(f"PIPELINE : {output['pipeline']}")
        print(f"QUERY    : {output['query']}")
        print(f"ANSWER   : {output['answer']}")
        print(f"SOURCES  : {', '.join(output['sources'])}")
        print("=" * 70)
