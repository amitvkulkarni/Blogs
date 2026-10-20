# _client.py
# Shared helper: builds an Azure OpenAI client and deployment name from .env.
import os
import sys
from dotenv import load_dotenv
from openai import AzureOpenAI

# Windows consoles default to cp1252 and can't print some model characters.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# load_dotenv() searches upward from this folder, so it finds "Code examples\.env".
load_dotenv()


def get_client(timeout: float = 60.0) -> AzureOpenAI:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if not endpoint or not api_key:
        raise RuntimeError("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set in .env")

    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-01-preview"),
        timeout=timeout,
    )


def get_deployment() -> str:
    return os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
