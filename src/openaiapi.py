import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()
client = Groq(
    api_key=os.getenv("GROQ_API_KEY"))

response = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "what is the capital of France?",
        }
    ],
    model="openai/gpt-oss-20b",
)

print(response.choices[0].message.content)