# example_l3_safety.py with Langchain Integration
import os
import logging
import time
from dotenv import load_dotenv
from openai import APIConnectionError, RateLimitError
from typing import Optional
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage

load_dotenv()

# Configure logging
log_file = os.getenv("LOG_FILE", "ai_interactions.log")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    handlers=[file_handler, stream_handler],
    force=True,
)

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


def _sanitize_message(message: str, max_length: int = 50) -> str:
    """Sanitize message for logging (truncate to avoid sensitive data exposure)."""
    if len(message) > max_length:
        return f"{message[:max_length]}..."
    return message


class SafeAIClient:
    """Wrapper providing safety guardrails for LLM interactions using Langchain."""

    def __init__(self):
        """Initialize Langchain Azure Chat OpenAI client."""
        self.llm = AzureChatOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini"),
            temperature=1,
        )
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content="You are a helpful and safe AI assistant."),
                ("human", "{user_message}"),
            ]
        )
        self.chain = self.prompt_template | self.llm | StrOutputParser()

    def validate_input(self, user_message: str) -> bool:
        """Validate user input for length and content."""
        if len(user_message) > 5000:
            logger.error("Message too long (max 5000 chars)")
            return False
        if not user_message.strip():
            logger.error("Message cannot be empty")
            return False
        if "http://" in user_message or "https://" in user_message:
            logger.warning("URLs detected in input - proceeding with caution")
        return True

    def chat_with_retry(self, user_message: str, max_retries: int = 3) -> Optional[str]:
        """Call LLM with retry logic and error handling."""
        if not self.validate_input(user_message):
            return None

        for attempt in range(max_retries):
            try:
                return self._call_langchain(user_message)
            except RateLimitError:
                wait_time = 2**attempt
                logger.warning(
                    f"Rate limit hit (attempt {attempt+1}/{max_retries}) - waiting {wait_time}s"
                )
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                return None
            except APIConnectionError:
                logger.error(
                    f"Connection error (attempt {attempt+1}/{max_retries}): Unable to connect"
                )
                if attempt < max_retries - 1:
                    continue
                return None
            except Exception as e:
                logger.error(f"Unexpected error: {type(e).__name__}: {e}")
                return None

    def _call_langchain(self, message: str) -> str:
        """Execute Langchain chain to get LLM response."""
        start_time = time.time()
        logger.info(f"Processing: {_sanitize_message(message)}")

        result = self.chain.invoke({"user_message": message})

        elapsed_time = time.time() - start_time
        logger.debug(f"Completed in {elapsed_time:.2f}s, response: {len(result)} chars")
        return result


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("LANGCHAIN QUERY SESSION START")
    logger.info("=" * 80)

    client = SafeAIClient()
    test_queries = [
        # "What is AI safety?",
        # "What are large language models?",
        "How does Langchain improve LLM applications?",
    ]

    for query in test_queries:
        logger.info("-" * 80)
        logger.info(f"Query: {query}")
        logger.info("-" * 80)

        result = client.chat_with_retry(query)
        if result:
            print("\n" + "=" * 80)
            print(f"Query: {query}")
            print("=" * 80)
            print(f"Response:\n{result}")
            print("=" * 80 + "\n")
            logger.info(f"Response: {_sanitize_message(result)} ({len(result)} chars)")
        else:
            logger.error(f"Failed to process: {query}")

    logger.info("=" * 80)
    logger.info("LANGCHAIN QUERY SESSION END")
    logger.info("=" * 80)

    for handler in logger.handlers:
        handler.flush()
