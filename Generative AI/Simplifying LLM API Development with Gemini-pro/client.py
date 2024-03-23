import requests

response = requests.post(
    "http://localhost:8000/funny/invoke",
    json={"input": {"topic": "Smart phones"}},
)

print(response.json()["output"]["content"])


response = requests.post(
    "http://localhost:8000/brief/invoke",
    json={
        "input": {"topic": "Green house gas emission and its impact on climate change"}
    },
)

print(response.json()["output"]["content"])
