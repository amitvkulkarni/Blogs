# example_l3_safety.py with Langchain Integration and Conversation Memory
import os
import logging
import time
from dotenv import load_dotenv
from openai import APIConnectionError, RateLimitError
from typing import Optional
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

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
    """Wrapper providing safety guardrails for LLM interactions using Langchain with conversation memory."""

    def __init__(self):
        """Initialize Langchain Azure Chat OpenAI client with conversation memory."""
        self.llm = AzureChatOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini"),
            temperature=1,
        )
        self.chat_history = []
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
        """Execute Langchain chain to get LLM response with conversation memory."""
        start_time = time.time()
        logger.info(f"Processing: {_sanitize_message(message)}")

        # Build messages with conversation history
        messages = [SystemMessage(content="You are a helpful and safe AI assistant.")]
        messages.extend(self.chat_history)
        messages.append(HumanMessage(content=message))

        # Invoke LLM with messages
        result_message = self.llm.invoke(messages)

        # Extract content from message object
        result_text = (
            result_message.content
            if hasattr(result_message, "content")
            else str(result_message)
        )

        # Add user message and assistant response to chat history
        self.chat_history.append(HumanMessage(content=message))
        self.chat_history.append(AIMessage(content=result_text))

        elapsed_time = time.time() - start_time
        logger.debug(
            f"Completed in {elapsed_time:.2f}s, response: {len(result_text)} chars"
        )
        return result_text


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("LANGCHAIN INTERACTIVE CHATBOT SESSION START")
    logger.info("=" * 80)

    client = SafeAIClient()

    print("\n" + "=" * 80)
    print("Welcome to the Langchain Chatbot with Conversation Memory!")
    print("=" * 80)
    print("\nThis chatbot demonstrates how conversation memory works.")
    print("Try introducing yourself (e.g., 'My name is Alice')")
    print("Then ask questions about what you told it earlier.")
    print("Type 'exit' to end the conversation.\n")
    print("=" * 80 + "\n")

    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()

            # Check for exit command
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("\n" + "=" * 80)
                print("Thank you for chatting! Goodbye!")
                print("=" * 80 + "\n")
                logger.info("User ended the conversation")
                break

            # Validate and process user input
            if not user_input:
                print("Please enter a message.\n")
                continue

            logger.info("-" * 80)
            logger.info(f"User Input: {user_input}")
            logger.info("-" * 80)

            # Get response from AI with memory
            result = client.chat_with_retry(user_input)

            if result:
                print(f"\nAssistant: {result}\n")
                logger.info(
                    f"AI Response: {_sanitize_message(result)} ({len(result)} chars)"
                )
                logger.info(
                    f"Conversation History Length: {len(client.chat_history)} messages"
                )
            else:
                print("Assistant: Sorry, I encountered an error. Please try again.\n")
                logger.error(f"Failed to process user input: {user_input}")

        except KeyboardInterrupt:
            print("\n\n" + "=" * 80)
            print("Conversation interrupted by user. Goodbye!")
            print("=" * 80 + "\n")
            logger.info("Conversation interrupted by user (KeyboardInterrupt)")
            break
        except Exception as e:
            logger.error(
                f"Unexpected error in conversation loop: {type(e).__name__}: {e}"
            )
            print(f"An unexpected error occurred: {e}\n")

    logger.info("=" * 80)
    logger.info("LANGCHAIN INTERACTIVE CHATBOT SESSION END")
    logger.info("=" * 80)

    for handler in logger.handlers:
        handler.flush()
