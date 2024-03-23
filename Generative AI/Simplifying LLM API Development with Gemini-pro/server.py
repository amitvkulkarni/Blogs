from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langserve import add_routes
import uvicorn
import os


# Set the environment variable for the Google API key
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize FastAPI app
app = FastAPI(title="Server", version="1.0", description="A API Server")

# Initialize Google Generative AI model with Gemini-pro
model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

# Add route for Gemini chat
add_routes(app, model, path="/gemini")

# Define essay prompt template
brief_prompt = ChatPromptTemplate.from_template("write a brief on the {topic}")
# Add route for essay generation
add_routes(app, brief_prompt | model, path="/brief")

# Define poem prompt template
funny_prompt = ChatPromptTemplate.from_template("Tell me something funny about {topic}")
# Add route for poem generation
add_routes(app, funny_prompt | model, path="/funny")


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
