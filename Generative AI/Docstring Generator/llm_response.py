import os

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_llm_response(input):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(input)
    return response
