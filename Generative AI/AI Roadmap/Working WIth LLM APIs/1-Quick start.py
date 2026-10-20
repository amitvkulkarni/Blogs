# example_l1_gemini.py
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()


# def simple_gemini_chat(user_message: str) -> str:
#     """
#     Basic Gemini API call.

#     Args:
#         user_message: User's input text

#     Returns:
#         Model's response text
#     """
#     genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
#     model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
#     if model_name == "gemini-2.0-flash":
#         model_name = "gemini-2.5-flash"
#     model = genai.GenerativeModel(model_name)

#     response = model.generate_content(user_message)
#     return response.text


# # Usage
# if __name__ == "__main__":
#     result = simple_gemini_chat("What is machine learning?")
#     print(f"Response: {result}")


# example_l1_azure.py
import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()


def simple_azure_chat(user_message: str) -> str:
    """
    Basic Azure Foundry API call using OpenAI SDK.

    Args:
        user_message: User's input text

    Returns:
        Model's response text
    """
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_FOUNDRY_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-01-preview"),
        azure_endpoint=os.getenv("AZURE_FOUNDRY_ENDPOINT")
        or os.getenv("AZURE_OPENAI_ENDPOINT"),
    )

    response = client.chat.completions.create(
        model=os.getenv("AZURE_FOUNDRY_MODEL")
        or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini"),
        messages=[{"role": "user", "content": user_message}],
    )

    return response.choices[0].message.content


# Usage
if __name__ == "__main__":
    result = simple_azure_chat("How do neural networks work?")
    print(f"Response: {result}")


# # example_l1_openai.py
# import os
# from dotenv import load_dotenv
# from openai import OpenAI

# load_dotenv()

# def simple_openai_chat(user_message: str) -> str:
#     """
#     Basic OpenAI API call.

#     Args:
#         user_message: User's input text

#     Returns:
#         Model's response text
#     """
#     client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#     response = client.chat.completions.create(
#         model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
#         messages=[
#             {"role": "user", "content": user_message}
#         ]
#     )

#     return response.choices[0].message.content

# # Usage
# if __name__ == "__main__":
#     result = simple_openai_chat("What is machine learning?")
#     print(f"Response: {result}")
