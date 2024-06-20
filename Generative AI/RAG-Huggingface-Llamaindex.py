############################### INstall necessary libraries ##############################################################
# pip install llama-index langchain pypdf langchain-community llama-index-embeddings-langchain llama-index-llms-huggingface
############################################################################################################################


# Import necessary libraries
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
    PromptTemplate,
    load_index_from_storage,
    StorageContext,
    SimpleDirectoryReader,
)
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.langchain import LangchainEmbedding
from google.colab import userdata
import os


############################### PDF Document for RAG  ###########################################################
# PDF version of the book “Handbook on Productivity” to query and test the performance of the RAG system.
# The PDF document is uploaded to the data folder
##########################################################################################################


# Get the Hugging Face API key from user data
sec_key = userdata.get("HF_API_KEY")

# Set the Hugging Face API key as an environment variable
HF_API_KEY = sec_key

# Define the directory where the index will be stored
PERSIST_DIR = "./storage"

# Check if the index already exists
if not os.path.exists(PERSIST_DIR):
    # If not, load the documents from the data directory
    documents = SimpleDirectoryReader("data").load_data()
    # Parse the documents into nodes
    parser = SimpleNodeParser()
    nodes = parser.get_nodes_from_documents(documents)
    # Initialize the Hugging Face inference API
    llm = HuggingFaceInferenceAPI(
        model_name="HuggingFaceH4/zephyr-7b-alpha", token=HF_API_KEY
    )
    # Initialize the embedding model
    embed_model = LangchainEmbedding(
        HuggingFaceInferenceAPIEmbeddings(
            api_key=HF_API_KEY, model_name="thenlper/gte-large"
        )
    )
    # Initialize the service context with the embedding model and the Hugging Face inference API
    service_context = ServiceContext.from_defaults(
        embed_model=embed_model, llm=llm, chunk_size=512
    )
    # Initialize the storage context for the vector store
    storage_context = StorageContext.from_defaults()
    # Initialize the index with the nodes, service context, and storage context
    index = VectorStoreIndex(
        nodes,
        service_context=service_context,
        storage_context=storage_context,
    )
    # Persist the index to the storage directory
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # If the index already exists, load it from the storage directory
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

# Initialize the query engine with the index
query_engine = index.as_query_engine()
# Query the engine
response = query_engine.query("what is the document all about?")
response = query_engine.query("what is green productivity?")
response = query_engine.query("Process-related Productivity Initiatives?")
# Print the response
print(response)

# Import necessary libraries for response synthesis and retrieval
from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever

# Initialize the response synthesizer with the service context
response_synthesizer = get_response_synthesizer(service_context=service_context)

# Initialize the vector retriever with the index and the number of top similar items to retrieve
vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=3)

# Initialize the query engine with the vector retriever and the response synthesizer
vector_query_engine = RetrieverQueryEngine(
    retriever=vector_retriever,
    response_synthesizer=response_synthesizer,
)

# Query the engine and print the response
query_engine = index.as_query_engine()
response = query_engine.query("what is the document all about?")


response = query_engine.query("what is green productivity?")
print(response)

response = query_engine.query("Process-related Productivity Initiatives?")
print(response)
