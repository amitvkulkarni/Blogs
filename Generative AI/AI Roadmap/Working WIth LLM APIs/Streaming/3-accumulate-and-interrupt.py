# 3-accumulate-and-interrupt.py
# Topic 3: Accumulating and interrupting streams.
#   - Build the full response from chunks.
#   - Check finish_reason to know if the generation was complete.
#   - Stop early (user cancel) and keep the partial text.
#   - Handle network failures mid-stream and keep what arrived.
import openai
from _client import get_client, get_deployment


def accumulate(prompt: str, max_completion_tokens: int | None = None) -> dict:
    """Collect all chunks into one string and report how the stream ended."""
    client = get_client()
    parts = []
    finish_reason = None

    stream = client.chat.completions.create(
        model=get_deployment(),
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=max_completion_tokens,  # None = model default
        stream=True,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        if choice.delta.content:
            parts.append(choice.delta.content)
        if choice.finish_reason:
            finish_reason = choice.finish_reason  # "stop", "length", "content_filter"

    return {
        "text": "".join(parts),
        "finish_reason": finish_reason,
        "complete": finish_reason == "stop",
    }


def stream_with_cancel(prompt: str, stop_after_chars: int = 120) -> str:
    """Stop reading once enough text has arrived; closing the stream ends the HTTP call."""
    client = get_client()
    parts, received = [], 0

    stream = client.chat.completions.create(
        model=get_deployment(),
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    try:
        for chunk in stream:
            if not chunk.choices or not chunk.choices[0].delta.content:
                continue
            text = chunk.choices[0].delta.content
            parts.append(text)
            received += len(text)
            if received >= stop_after_chars:
                break  # simulated user cancel
    finally:
        stream.close()  # always release the connection

    return "".join(parts)


def stream_with_failure(prompt: str, timeout: float = 60.0) -> str:
    """Return whatever arrived before a network/API error."""
    parts = []
    try:
        client = get_client(timeout=timeout)
        stream = client.chat.completions.create(
            model=get_deployment(),
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                parts.append(chunk.choices[0].delta.content)
    except openai.APITimeoutError as exc:
        print(f"\n[timeout] {exc}")
    except openai.APIConnectionError as exc:
        print(f"\n[network error] {exc}")
    except openai.APIStatusError as exc:
        print(f"\n[API error {exc.status_code}] {exc.message}")
    except KeyboardInterrupt:
        print("\n[Ctrl+C] stopped by user")
    return "".join(parts)


if __name__ == "__main__":
    question = "List 8 interesting facts about the Python programming language."

    print("=== 1. Accumulate the full answer ===")
    result = accumulate(question)
    print(result["text"])
    print(f"\nfinish_reason={result['finish_reason']}, complete={result['complete']}")

    print("\n=== 2. Output cut short by max_completion_tokens (incomplete generation) ===")
    # Reasoning models count hidden reasoning tokens against this cap, so the visible text may be empty.
    short = accumulate(question, max_completion_tokens=150)
    print(short["text"] or "(No output: the max completion token limit was reached before any visible text was generated.)")
    print(f"\nfinish_reason={short['finish_reason']}, complete={short['complete']}")

    print("\n=== 3. Cancel after ~120 characters ===")
    partial = stream_with_cancel("Write a long story about a robot learning to paint.")
    print(partial + " ...[cancelled]")

    print("\n=== 4. Simulated network failure (timeout too small) ===")
    partial = stream_with_failure(question, timeout=0.001)
    print(f"Kept partial text: {partial!r}")
