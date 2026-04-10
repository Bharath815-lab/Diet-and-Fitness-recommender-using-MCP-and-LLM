from ollama import Client

client = Client(host="http://localhost:11434")

messages = [
    {
        "role": "user",
        "content": "Why is the sky blue?"
    }
]

for part in client.chat(
    model="llama3.1",
    messages=messages,
    stream=True
):
    print(part["message"]["content"], end="", flush=True)
