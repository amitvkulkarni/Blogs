"""
crag_pipeline.py
----------------
Corrective Retrieval-Augmented Generation (CRAG) using Azure OpenAI + LangChain.

CRAG adds three quality-control layers on top of standard RAG:
  1. Document Grading  — LLM scores each retrieved doc as relevant/irrelevant/ambiguous
  2. Draft + Self-Reflection — LLM critiques its own initial answer
  3. Confidence Scoring — final answer includes a confidence level and citations

Pipeline:
  Retrieve → Grade docs → Filter → Draft answer → Self-critique → Final answer
"""

import os
import json
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

# ── Load environment variables ────────────────────────────────────────────────
load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")
EMBEDDING_DEPLOYMENT = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002"
)


# ── LLM singleton ─────────────────────────────────────────────────────────────


def get_llm() -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_deployment=CHAT_DEPLOYMENT,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
    )


# ── Step 1: Document loading & vector store (same as RAG) ─────────────────────


def load_documents(docs_dir: str = "./documents") -> list:
    loader = DirectoryLoader(docs_dir, glob="**/*.txt", loader_cls=TextLoader)
    docs = loader.load()
    print(f"[CRAG] Loaded {len(docs)} document(s)")
    return docs


def build_vector_store(docs: list) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"[CRAG] Split into {len(chunks)} chunks")

    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=EMBEDDING_DEPLOYMENT,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    print("[CRAG] Vector store built (FAISS)")
    return vector_store


# ── Step 2: Document Grading ──────────────────────────────────────────────────

GRADER_TEMPLATE = """You are a document relevance grader for an engineering compliance assistant.
Given the user question and a document chunk, decide if the document is relevant.

Respond only with valid JSON in exactly this format:
{{
  "grade": "relevant" | "irrelevant" | "ambiguous",
  "reason": "one sentence explanation"
}}

User Question: {question}

Document Chunk:
{document}

JSON Response:"""

GRADER_PROMPT = PromptTemplate(
    input_variables=["question", "document"],
    template=GRADER_TEMPLATE,
)


def grade_documents(
    query: str, docs: list[Document], llm: AzureChatOpenAI
) -> list[dict]:
    """
    Grade each retrieved document for relevance to the query.
    Returns a list of dicts: {doc, grade, reason}
    """
    graded = []
    for doc in docs:
        prompt = GRADER_PROMPT.format(question=query, document=doc.page_content[:800])
        response = llm.invoke(prompt)
        raw = response.content.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            result = {
                "grade": "ambiguous",
                "reason": "Could not parse grader response.",
            }

        graded.append(
            {
                "doc": doc,
                "grade": result.get("grade", "ambiguous"),
                "reason": result.get("reason", ""),
            }
        )

    relevant_count = sum(1 for g in graded if g["grade"] == "relevant")
    print(
        f"[CRAG] Graded {len(graded)} chunks → "
        f"{relevant_count} relevant, "
        f"{sum(1 for g in graded if g['grade'] == 'irrelevant')} irrelevant, "
        f"{sum(1 for g in graded if g['grade'] == 'ambiguous')} ambiguous"
    )
    return graded


# ── Step 3: Draft Generation ──────────────────────────────────────────────────

DRAFT_TEMPLATE = """You are a precise engineering and compliance assistant.
Use only the verified, relevant context below to answer the question.
Cite the source document name in your answer wherever possible.
If the context does not contain enough information, state that clearly.

Relevant Context:
{context}

Question: {question}

Draft Answer:"""

DRAFT_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=DRAFT_TEMPLATE,
)


def generate_draft(
    query: str, relevant_docs: list[Document], llm: AzureChatOpenAI
) -> str:
    """Generate an initial draft answer from relevant documents."""
    if not relevant_docs:
        return "No relevant documents were found to answer this question."

    context = "\n\n---\n\n".join(
        f"[Source: {os.path.basename(d.metadata.get('source', 'unknown'))}]\n{d.page_content}"
        for d in relevant_docs
    )
    prompt = DRAFT_PROMPT.format(context=context, question=query)
    response = llm.invoke(prompt)
    return response.content.strip()


# ── Step 4: Self-Reflection / Critique ────────────────────────────────────────

CRITIQUE_TEMPLATE = """You are a critical reviewer for an engineering compliance assistant.
Review the draft answer below for accuracy and groundedness.

Check for:
1. Claims not explicitly supported by the provided context
2. Missing important caveats or conditions from the context
3. Overconfident statements that should be hedged

Respond only with valid JSON in exactly this format:
{{
  "issues_found": true | false,
  "critique": "summary of issues found, or 'No issues found' if clean",
  "confidence": "high" | "medium" | "low",
  "revised_answer": "improved version of the answer (or original if no changes needed)"
}}

Original Question: {question}

Context Used:
{context}

Draft Answer:
{draft}

JSON Response:"""

CRITIQUE_PROMPT = PromptTemplate(
    input_variables=["question", "context", "draft"],
    template=CRITIQUE_TEMPLATE,
)


def self_reflect(
    query: str,
    draft: str,
    relevant_docs: list[Document],
    llm: AzureChatOpenAI,
) -> dict:
    """
    LLM critiques its own draft answer.
    Returns dict with: issues_found, critique, confidence, revised_answer
    """
    if not relevant_docs:
        return {
            "issues_found": True,
            "critique": "No relevant documents available to verify claims.",
            "confidence": "low",
            "revised_answer": "Insufficient information to provide a verified answer.",
        }

    context = "\n\n---\n\n".join(
        f"[Source: {os.path.basename(d.metadata.get('source', 'unknown'))}]\n{d.page_content}"
        for d in relevant_docs
    )
    prompt = CRITIQUE_PROMPT.format(question=query, context=context, draft=draft)
    response = llm.invoke(prompt)
    raw = response.content.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {
            "issues_found": False,
            "critique": "Could not parse critique response.",
            "confidence": "medium",
            "revised_answer": draft,
        }
    return result


# ── Full CRAG Pipeline ─────────────────────────────────────────────────────────


def run_crag(query: str, vector_store: FAISS, llm: AzureChatOpenAI) -> dict:
    """
    Full CRAG pipeline:
      Retrieve -> Grade -> Filter -> Draft -> Self-Reflect -> Final Answer
    """
    print(f"\n[CRAG] Query: {query}")

    # ── 1. Retrieve (same as RAG) ──────────────────────────────────────────────
    retriever = vector_store.as_retriever(search_kwargs={"k": 6})
    retrieved_docs = retriever.invoke(query)
    print(f"[CRAG] Retrieved {len(retrieved_docs)} candidate chunks")

    # ── 2. Grade documents ────────────────────────────────────────────────────
    graded = grade_documents(query, retrieved_docs, llm)

    # ── 3. Filter: keep relevant (and ambiguous as fallback) ──────────────────
    relevant_docs = [g["doc"] for g in graded if g["grade"] == "relevant"]
    ambiguous_docs = [g["doc"] for g in graded if g["grade"] == "ambiguous"]

    # If no relevant docs found, fall back to ambiguous ones
    if not relevant_docs:
        print("[CRAG] No 'relevant' docs found — using 'ambiguous' docs as fallback")
        relevant_docs = ambiguous_docs

    grading_summary = [
        {
            "source": os.path.basename(g["doc"].metadata.get("source", "unknown")),
            "grade": g["grade"],
            "reason": g["reason"],
        }
        for g in graded
    ]

    # ── 4. Generate draft answer ─────────────────────────────────────────────
    draft = generate_draft(query, relevant_docs, llm)
    print(f"[CRAG] Draft generated ({len(draft)} chars)")

    # ── 5. Self-reflection / critique ────────────────────────────────────────
    critique_result = self_reflect(query, draft, relevant_docs, llm)
    print(
        f"[CRAG] Self-reflection → confidence: {critique_result.get('confidence', 'unknown')}, "
        f"issues found: {critique_result.get('issues_found', False)}"
    )

    # ── 6. Compile final output ───────────────────────────────────────────────
    sources = list(
        {os.path.basename(d.metadata.get("source", "unknown")) for d in relevant_docs}
    )

    return {
        "query": query,
        "answer": critique_result.get("revised_answer", draft),
        "draft": draft,
        "critique": critique_result.get("critique", ""),
        "issues_found": critique_result.get("issues_found", False),
        "confidence": critique_result.get("confidence", "medium"),
        "sources": sources,
        "grading_summary": grading_summary,
        "pipeline": "CRAG (Corrective RAG)",
    }


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    docs = load_documents()
    vector_store = build_vector_store(docs)
    llm = get_llm()

    queries = [
        "Does the vendor fire suppression system cover all three cities including Seattle?"
        # "What seismic bracing requirements apply to structural beams installed in Seattle?",
        # "What HVAC efficiency minimum is required for chillers in Chicago?",
    ]

    for q in queries:
        output = run_crag(q, vector_store, llm)
        print("\n" + "=" * 70)
        print(f"PIPELINE   : {output['pipeline']}")
        print(f"QUERY      : {output['query']}")
        print(f"CONFIDENCE : {output['confidence'].upper()}")
        print(
            f"ISSUES     : {'Yes — see critique' if output['issues_found'] else 'None detected'}"
        )
        print(f"CRITIQUE   : {output['critique']}")
        print(f"ANSWER     : {output['answer']}")
        print(f"SOURCES    : {', '.join(output['sources'])}")
        print("\nDOCUMENT GRADING:")
        for g in output["grading_summary"]:
            print(f"  [{g['grade'].upper():11}] {g['source']} — {g['reason']}")
        print("=" * 70)
