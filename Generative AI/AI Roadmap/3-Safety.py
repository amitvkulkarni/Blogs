# example_l3_safety.py
import os
from dotenv import load_dotenv
from openai import APIConnectionError, AzureOpenAI, RateLimitError
from typing import Optional

load_dotenv()


class SafeAIClient:
    """Wrapper providing safety guardrails for LLM interactions."""

    def validate_input(self, user_message: str) -> bool:
        """
        Input validation.

        Returns: True if valid, False otherwise
        """
        # Check length
        if len(user_message) > 5000:
            print("ERROR: Message too long (max 5000 chars)")
            return False

        # Check empty
        if not user_message.strip():
            print("ERROR: Message cannot be empty")
            return False

        # Check for suspicious patterns (block URLs in input for this example)
        if "http://" in user_message or "https://" in user_message:
            print("WARNING: URLs detected in input - proceeding with caution")

        return True

    def chat_with_retry(self, user_message: str, max_retries: int = 3) -> Optional[str]:
        """
        Call LLM with retry logic and error handling.

        Returns: Response text or None on failure
        """
        # Validate input
        if not self.validate_input(user_message):
            return None

        # Attempt with retries
        for attempt in range(max_retries):
            try:
                return self._call_azure(user_message)

            except RateLimitError as e:
                print(f"Rate limit hit (attempt {attempt+1}/{max_retries})")
                if attempt < max_retries - 1:
                    import time

                    wait_time = 2**attempt  # Exponential backoff
                    time.sleep(wait_time)
                    continue
                return None

            except APIConnectionError as e:
                print(f"Connection error (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    continue
                return None

            except Exception as e:
                print(f"Unexpected error: {type(e).__name__}: {e}")
                return None

    def _call_azure(self, message: str) -> str:
        """Safe Azure call."""
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini"),
            messages=[{"role": "user", "content": message}],
            timeout=30,
            max_completion_tokens=1000,
        )
        return response.choices[0].message.content or ""


# Usage
if __name__ == "__main__":
    # Test with Azure
    client = SafeAIClient()
    result = client.chat_with_retry("What is AI safety?")
    # result = client.chat_with_retry("Generate an image on AI safety?")
    if result:
        print(f"Azure Response: {result}")
    else:
        print("Failed to get response")
