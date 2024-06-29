#######################################################################################################################################
# How to Evaluate RAG Pipeline with Open Source Models
# https://medium.com/@amitvkulkarni/how-to-evaluate-rag-pipeline-with-open-source-models-998e2a35ae57
#######################################################################################################################################

# Install the library using "pip install beyondllm"

# Import necessary modules from the beyondllm package.
from beyondllm import source, embeddings, retrieve, llms, generator

# Import Google Colab's drive module to mount Google Drive.
from google.colab import drive

# Mount Google Drive to access files stored there.
drive.mount("/content/gdrive")

# Import getpass for securely entering a password or secret key.
from getpass import getpass
import os

# Prompt the user to enter their Google API key securely and set it as an environment variable.
os.environ["GOOGLE_API_KEY"] = getpass()

# Use the 'source' module to load and preprocess a PDF document.
# The document is split into chunks of 100 units with an overlap of 10 units between chunks.
data = source.fit(
    "/content/data/Politics.pdf", dtype="pdf", chunk_size=100, chunk_overlap=10
)

# Print the processed data to verify its structure or content.
print(data)

# Initialize an automatic retriever with the processed data.
# This retriever will search for the top 3 most relevant chunks of text based on a query.
retriever = retrieve.auto_retriever(data, type="normal", top_k=3)

# Print the retriever object to verify its configuration.
print(retriever)

# Create a generation pipeline with a specific question and the configured retriever.
# This pipeline will use the retriever to find relevant information and generate an answer to the question.
pipeline = generator.Generate(
    question="What is G7 and how is it different from G20??", retriever=retriever
)

# Print the pipeline object to verify its configuration.
print(pipeline)

# Execute the pipeline to generate an answer to the question and print the result.
print(pipeline.call())

# Retrieve and print the relevancy score of the answer, indicating how relevant the generated answer is to the question.
print(pipeline.get_answer_relevancy())

# Retrieve and print the groundedness score of the answer, indicating how well the answer is grounded in the retrieved documents.
print(pipeline.get_groundedness())

# Retrieve and print the evaluations of the RAG (Retrieval-Augmented Generation) triad, providing insights into the retrieval and generation process.
print(pipeline.get_rag_triad_evals())

# Define a system prompt that instructs the AI on how to generate responses.
# This prompt limits the AI's response to a maximum of 25 words, focusing on expertise in retrieving information from documents.
system_prompt = """
You can AI assistant and an expert in retrieving information from the documents. Keep it to 25 words maximum
"""

# Create a new generation pipeline with the same question and retriever, but with an additional system prompt to guide the generation.
pipeline = generator.Generate(
    question="What is G7 and how is it different from G20??",
    retriever=retriever,
    system_prompt=system_prompt,
)

# Execute the new pipeline to generate an answer to the question, considering the system prompt, and print the result.
print(pipeline.call())
