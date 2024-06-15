from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import time


# Load the local PDF file
if local_path:
    # Create a loader for the PDF file
    loader = UnstructuredPDFLoader(file_path=local_path)
    # Load the data from the PDF file
    data = loader.load()
else:
    # If no file path is provided, print a message
    print("Upload a PDF file")

# Split the loaded data into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
chunks = text_splitter.split_documents(data)

# Create a vector database from the chunks
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),
    collection_name="local-ai",
)

# Import necessary classes
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

# Initialize a language model from Ollama
local_model = "mistral"
llm = ChatOllama(model=local_model)

# Define a prompt for the language model
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate three
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
)

# Initialize a retriever with the vector database and the language model
retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
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
chain.invoke(input("What is the document all about?"))

# Invoke the chain with another user question
chain.invoke(
    input(
        "What is the relation between the Demerit Points and Corresponding Suspension Points?"
    )
)

# Measure the time taken to invoke the chain with a third user question
start_time = time.time()
print(chain.invoke(input("what is the preamble mentioned in the document?")))
end_time = time.time()
elapsed_time = (end_time - start_time) / 60
print(f"Time taken: {elapsed_time} minutes")
