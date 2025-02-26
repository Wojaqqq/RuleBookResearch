import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
from dotenv import load_dotenv

load_dotenv()


chat_response = client.chat.completions.create(model="gpt-4o-mini",
messages=[
    {"role": "system", "content": f"Give me a warm welcome to my first APi call"},
    {"role": "user", "content": f"Is my first API call works? Hello?"}
])   

response_text = chat_response.choices[0].message.content
total_tokens = chat_response.usage.total_tokens

print(f"Response: {response_text}\nTokens_used: {total_tokens}")
