# 1-rendering-stream.py
# Topic 1: Rendering output incrementally.
# Instead of waiting for the full answer, print each chunk as it arrives.
from _client import get_client, get_deployment


def stream_and_render(prompt: str) -> None:
    client = get_client()

    stream = client.chat.completions.create(
        model=get_deployment(),
        messages=[{"role": "user", "content": prompt}],
        stream=True,  # the API sends tokens as they are generated
    )

    for chunk in stream:
        # Some chunks (e.g., the first one with the role) have no text.
        if not chunk.choices:
            continue
        text = chunk.choices[0].delta.content
        if text:
            print(text, end="", flush=True)  # flush so the user sees it immediately
    print()


if __name__ == "__main__":
    stream_and_render("Explain what a Python generator is in 3 short bullet points.")
