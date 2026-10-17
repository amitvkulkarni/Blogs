# example_l3_safety.py
import os
import logging
import time
from dotenv import load_dotenv
from openai import APIConnectionError, AzureOpenAI, RateLimitError
from typing import Optional

load_dotenv()

# Configure logging with proper file handling
log_file = os.getenv("LOG_FILE", "ai_interactions.log")

# Create file handler with explicit configuration
file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")  # Append mode
file_handler.setLevel(logging.INFO)

# Create stream handler for console
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Configure root logger
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    handlers=[file_handler, stream_handler],
    force=True,
)

logger = logging.getLogger(__name__)

# Suppress sensitive information from third-party libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


def _sanitize_message(message: str, max_length: int = 50) -> str:
    """Sanitize message for logging (truncate to avoid sensitive data exposure)."""
    if len(message) > max_length:
        return f"{message[:max_length]}..."
    return message


class SafeAIClient:
    """Wrapper providing safety guardrails for LLM interactions."""

    def validate_input(self, user_message: str) -> bool:
        """
        Input validation.

        Returns: True if valid, False otherwise
        """
        # Check length
        if len(user_message) > 5000:
            logger.error("Message too long (max 5000 chars)")
            return False

        # Check empty
        if not user_message.strip():
            logger.error("Message cannot be empty")
            return False

        # Check for suspicious patterns (block URLs in input for this example)
        if "http://" in user_message or "https://" in user_message:
            logger.warning("URLs detected in input - proceeding with caution")

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
                wait_time = 2**attempt  # Exponential backoff
                logger.warning(
                    f"Rate limit hit (attempt {attempt+1}/{max_retries}) - waiting {wait_time}s"
                )
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                    continue
                return None

            except APIConnectionError as e:
                logger.error(
                    f"Connection error (attempt {attempt+1}/{max_retries}): Unable to connect to service"
                )
                if attempt < max_retries - 1:
                    continue
                return None

            except Exception as e:
                logger.error(f"Unexpected error: {type(e).__name__}")
                logger.debug(f"Error details: {e}")
                return None

    def _call_azure(self, message: str) -> str:
        """Safe Azure call."""
        start_time = time.time()
        logger.info(f"Processing message: {_sanitize_message(message)}")
        logger.debug("Initializing Azure OpenAI client")
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
        result = response.choices[0].message.content or ""
        elapsed_time = time.time() - start_time
        logger.debug(f"Azure API call completed successfully in {elapsed_time:.2f}s")
        logger.debug(f"Response length: {len(result)} characters")
        return result


# Usage
if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("QUERY SESSION START")
    logger.info("=" * 80)

    # Test with Azure
    logger.info("Starting SafeAIClient test")
    client = SafeAIClient()
    # result = client.chat_with_retry("What is AI safety?")
    # result = client.chat_with_retry("What are large  language models")
    result = client.chat_with_retry("Generate an image on AI safety?")
    if result:
        logger.info(
            f"Response received: {_sanitize_message(result)} ({len(result)} chars)"
        )
    else:
        logger.error("Failed to get response from Azure API")

    logger.info("=" * 80)
    logger.info("QUERY SESSION END")
    logger.info("=" * 80)

    # Flush all handlers to ensure logs are written to file
    for handler in logger.handlers:
        handler.flush()
