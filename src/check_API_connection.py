import os
from openai import OpenAI
from dotenv import load_dotenv

"""

Run this file to check if you have access to OPENAI API.

"""

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


if __name__ == "__main__":

    chat_response = client.chat.completions.create(model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": f"Give me a warm welcome to my test API call"},
        {"role": "user", "content": f"Is my API call works? Hello?"}
    ])   

    response_text = chat_response.choices[0].message.content
    total_tokens = chat_response.usage.total_tokens

    print(f"Response: {response_text}\nTokens_used: {total_tokens}")
