"""RAG_HF_MISTRAL_FAISS_INSTRUCTOR_EMBEDDING_TESTING
https://amitvkulkarni.medium.com/how-to-improve-the-performance-of-the-rag-with-multiqueryretriever-24b1e36e4d9d
"""

# pip install langchain-huggingface
# pip install faiss-cpu
# pip install langchain
# pip install -q langchain_community
# pip install -q InstructorEmbedding
# pip install sentence-transformers==2.2.2

# pip -q install tiktoken pypdf sentence-transformers==2.2.2 InstructorEmbedding langchain_community huggingface_hub langchain-huggingface

import pickle
import faiss
import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.vectorstores import FAISS
from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from google.colab import userdata

userdata.get("HF_API_KEY")
userdata.get("HF_TOKEN")

loader = DirectoryLoader(f"/content/data", glob="*.txt", loader_cls=TextLoader)
documents = loader.load()

documents

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)

texts = text_splitter.split_documents(documents)

texts

len(texts)

instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")

db_instructEmbedd = FAISS.from_documents(texts, instructor_embeddings)

retriever = db_instructEmbedd.as_retriever(search_kwargs={"k": 5})
retriever

# con = [info.page_content for info in retriever.get_relevant_documents("What is the player talking about?")]

doc1 = retriever.get_relevant_documents("How many runs did the captain score?")
# [info.page_content for info in doc1]

doc1

question = "How many runs did the captain score?"
docs = retriever.invoke(question)
# print(docs[0].page_content)
docs

question = "What is G7 and how is it different from G20?"
docs = retriever.invoke(question)
docs

#################### Improvement 1 ######################

from langchain_core.runnables import RunnablePassthrough

repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
llm = HuggingFaceEndpoint(
    repo_id=repo_id, max_length=128, temperature=0.7, token="HF_TOKEN"
)

# Define a prompt for the RAG (Retrieve and Generate) model
template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

# Create a prompt from the template
prompt = ChatPromptTemplate.from_template(template)

# Create a processing chain with the retriever, the RAG prompt, the language model, and a string output parser
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Invoke the chain with a user question
question = "How many runs did the captain score?"
response = chain.invoke(question)
print(response)

question = "What is captain talking about?"
response = chain.invoke(question)
print(response)

question = "What is G7 and how is it different from G20?"
response = chain.invoke(question)
print(response)

#################### Improvement 2 ######################

# Set logging for the queries
import logging

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

question = "What is captain talking about?"

unique_docs = retriever.invoke(question)
len(unique_docs)

unique_docs


question = "What is G7 and how is it different from G20?"

unique_docs = retriever.invoke(question)
len(unique_docs)

unique_docs


question = "How many runs did Rohit Sharma score?"

unique_docs = retriever.invoke(question)
len(unique_docs)

unique_docs


from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain import PromptTemplate, LLMChain

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
)

repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
llm = HuggingFaceEndpoint(
    repo_id=repo_id, max_length=128, temperature=0.7, token="HF_TOKEN"
)

# Initialize a retriever with the vector database and the language model
retriever_3 = MultiQueryRetriever.from_llm(
    db_instructEmbedd.as_retriever(), llm, prompt=QUERY_PROMPT
)
retriever_3

# Define a prompt for the RAG (Retrieve and Generate) model
template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

# Create a prompt from the template
prompt = ChatPromptTemplate.from_template(template)

# Create a processing chain with the retriever, the RAG prompt, the language model, and a string output parser
chain = (
    {"context": retriever_3, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Invoke the chain with a user question
question = "How many runs did the captain score?"
response = chain.invoke(question)
print(response)

# Invoke the chain with a user question
question = "What is G7 and how is it different from G20?"
response = chain.invoke(question)
print(response)

question = "What is captain talking about?"
response = chain.invoke(question)
print(response)


###########

import logging

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

question = "What is captain talking about?"

unique_docs = retriever.invoke(question)
unique_docs
