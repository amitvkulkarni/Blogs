# 2-ttft-and-latency.py
# Topic 2: TTFT and perceived latency.
#   TTFT (time to first token): time until the first piece of text arrives.
#   Total latency: time until the stream ends.
#   Generation speed: output tokens per second after the first token.
import time
from _client import get_client, get_deployment


def measure_stream(prompt: str) -> None:
    client = get_client()
    start = time.perf_counter()
    first_token_at = None
    chars = 0

    stream = client.chat.completions.create(
        model=get_deployment(),
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        stream_options={"include_usage": True},  # last chunk carries token counts
    )

    usage = None
    for chunk in stream:
        if chunk.usage:
            usage = chunk.usage
        if not chunk.choices:
            continue
        text = chunk.choices[0].delta.content
        if text:
            if first_token_at is None:
                first_token_at = time.perf_counter()
            chars += len(text)
            print(text, end="", flush=True)
    end = time.perf_counter()
    print("\n")

    if first_token_at is None:
        print("No text received.")
        return

    ttft = first_token_at - start
    total = end - start
    print(f"TTFT (time to first token): {ttft:.2f} s")
    print(f"Total latency:              {total:.2f} s")
    print(f"Characters received:        {chars}")
    if usage:
        generation_time = end - first_token_at
        print(f"Output tokens:              {usage.completion_tokens}")
        if generation_time > 0:
            print(f"Generation speed:           {usage.completion_tokens / generation_time:.1f} tokens/s")


if __name__ == "__main__":
    measure_stream("Give me three tips for writing clean Python code.")
