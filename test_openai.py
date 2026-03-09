from openai import OpenAI
import os

print("KEY:", os.getenv("OPENAI_API_KEY")[:10], "...")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

res = client.responses.create(
    model="gpt-5-mini",
    input="테스트"
)

print("SUCCESS")
print(res.output_text)