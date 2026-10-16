# example_l2_structured_prompts.py
import os
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai

load_dotenv()

SYSTEM_PROMPT = """You are a helpful AI assistant specializing in machine learning.
Provide clear, concise explanations. Keep responses under 150 words.
If you don't know something, say so clearly."""

EXAMPLES = [
    {
        "question": "What is supervised learning?",
        "answer": "Supervised learning uses labeled data (input-output pairs) to train models. The model learns the mapping between inputs and outputs. Common examples: classification (predicting categories) and regression (predicting numbers).",
    },
    {
        "question": "What is unsupervised learning?",
        "answer": "Unsupervised learning finds patterns in unlabeled data. Common techniques: clustering (grouping similar data) and dimensionality reduction (simplifying data). No predefined correct answers.",
    },
]


# def structured_gemini_chat(user_question: str) -> str:
#     """Gemini with system instruction, few-shot examples, and robust error handling."""
#     api_key = os.getenv("GEMINI_API_KEY")
#     if not api_key:
#         return "Error: GEMINI_API_KEY environment variable is not set."

#     genai.configure(api_key=api_key)

#     examples_text = "\n\n".join(
#         [f"Q: {ex['question']}\nA: {ex['answer']}" for ex in EXAMPLES]
#     )

#     system_instruction = f"{SYSTEM_PROMPT}\n\nExamples:\n{examples_text}"

#     try:
#         model = genai.GenerativeModel(
#             "gemini-2.5-flash",
#             system_instruction=system_instruction,
#             generation_config=genai.GenerationConfig(
#                 temperature=0.5,
#                 top_p=0.9,
#             ),
#         )

#         response = model.generate_content(user_question)
#         return (
#             response.text
#             if response.text
#             else "Error: No response generated from Gemini API."
#         )
#     except Exception as e:
#         return f"Error executing Gemini API call: {e}"


def structured_azure_chat(user_question: str) -> str:
    """Azure with system prompt, few-shot examples, and robust error handling."""
    api_key = os.getenv("AZURE_FOUNDRY_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
    if not api_key:
        return "Error: Azure API key not set (AZURE_FOUNDRY_API_KEY or AZURE_OPENAI_API_KEY)."

    azure_endpoint = os.getenv("AZURE_FOUNDRY_ENDPOINT") or os.getenv(
        "AZURE_OPENAI_ENDPOINT"
    )
    if not azure_endpoint:
        return "Error: Azure endpoint not set (AZURE_FOUNDRY_ENDPOINT or AZURE_OPENAI_ENDPOINT)."

    try:
        from openai import AzureOpenAI

        client = AzureOpenAI(
            api_key=api_key,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-01-preview"),
            azure_endpoint=azure_endpoint,
        )

        examples_text = "\n\n".join(
            [f"Q: {ex['question']}\nA: {ex['answer']}" for ex in EXAMPLES]
        )

        system_content = f"{SYSTEM_PROMPT}\n\nExamples:\n{examples_text}"

        response = client.chat.completions.create(
            model=os.getenv("AZURE_FOUNDRY_MODEL")
            or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini"),
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_question},
            ],
        )

        return (
            response.choices[0].message.content
            if response.choices
            else "Error: No response from Azure API."
        )
    except Exception as e:
        return f"Error executing Azure API call: {e}"


# def structured_openai_chat(user_question: str) -> str:
#     """OpenAI with system prompt and few-shot examples."""
#     client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#     # Build few-shot examples in prompt
#     examples_text = "\n\n".join([
#         f"Q: {ex['question']}\nA: {ex['answer']}" for ex in EXAMPLES
#     ])

#     system_content = f"{SYSTEM_PROMPT}\n\nExamples:\n{examples_text}"

#     response = client.chat.completions.create(
#         model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
#         messages=[
#             {"role": "system", "content": system_content},
#             {"role": "user", "content": user_question}
#         ],
#         temperature=0.7,  # Balanced creativity
#         top_p=0.9         # Diversity in response
#     )

#     return response.choices[0].message.content


# Usage & Comparison
if __name__ == "__main__":
    question = "How do neural networks work?"

    # print("=== OpenAI ===")
    # print(structured_openai_chat(question))

    # print("\n=== Gemini ===")
    # print(structured_gemini_chat(question))

    print("\n=== Azure Foundry ===")
    print(structured_azure_chat(question))
