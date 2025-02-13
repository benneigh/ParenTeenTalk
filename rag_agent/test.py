import openai, os

openai.api_key = "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"

try:
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello, world!"}]
    )
    print(response)
except openai.AuthenticationError as e:
    print("Authentication failed:", e)
