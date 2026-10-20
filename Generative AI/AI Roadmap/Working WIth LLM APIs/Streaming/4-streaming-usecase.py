# 4-streaming-usecase.py
# Use case: a Python tutor that answers your question live.
# Combines the concepts from files 1-3:
#   - Render output as it arrives (1)
#   - Measure time to first token and total time (2)
#   - Accumulate the full answer and check finish_reason (3)
#   - Cancel with Ctrl+C, keep partial text, handle network/API errors (3)
import time
import openai
from _client import get_client, get_deployment


def ask_tutor(question: str) -> dict:
    client = get_client()
    start = time.perf_counter()
    first_token_at = None
    parts = []
    finish_reason = None
    interrupted = False
    error = None
    stream = None

    try:
        stream = client.chat.completions.create(
            model=get_deployment(),
            messages=[
                {
                    "role": "system",
                    "content": "You are a friendly Python tutor. Answer clearly in short paragraphs.",
                },
                {"role": "user", "content": question},
            ],
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.delta.content:
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                parts.append(choice.delta.content)
                print(choice.delta.content, end="", flush=True)  # render live
            if choice.finish_reason:
                finish_reason = choice.finish_reason
    except KeyboardInterrupt:
        interrupted = True  # user pressed Ctrl+C; keep what we have
    except openai.APITimeoutError:
        error = "The request timed out. Check your network and try again."
    except openai.APIConnectionError:
        error = "Could not reach Azure. Check your network connection."
    except openai.APIStatusError as exc:
        error = f"Azure returned an error ({exc.status_code}): {exc.message}"
    finally:
        if stream is not None:
            stream.close()  # always release the connection

    end = time.perf_counter()
    return {
        "text": "".join(parts),
        "finish_reason": finish_reason,
        "interrupted": interrupted,
        "error": error,
        "ttft": (first_token_at - start) if first_token_at else None,
        "total": end - start,
    }


def print_summary(result: dict) -> None:
    print("\n")
    if result["error"]:
        print(f"[Error] {result['error']}")
    elif result["interrupted"]:
        print("[Stopped] You cancelled the answer. Partial text was kept above.")
    elif result["finish_reason"] == "length":
        print("[Incomplete] The answer hit the token limit.")
    if not result["text"] and not result["error"]:
        print("(No visible text was generated.)")

    ttft = f"{result['ttft']:.2f} s" if result["ttft"] is not None else "n/a"
    print(
        f"Time to first token: {ttft} | Total time: {result['total']:.2f} s | "
        f"Characters: {len(result['text'])} | Finish reason: {result['finish_reason'] or 'n/a'}"
    )


if __name__ == "__main__":
    print("Ask a Python question (press Ctrl+C to stop the answer while it streams).")
    question = (
        input("Your question: ").strip()
        or "What is a list comprehension? Give one example."
    )
    print()
    print_summary(ask_tutor(question))
