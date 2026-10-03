"""
test_gemini.py
Checks that your Gemini API key works before we build the chatbot.
Run from the project folder:  python test_gemini.py
"""
import tomllib
from google import genai

with open(".streamlit/secrets.toml", "rb") as f:
    key = tomllib.load(f)["GEMINI_API_KEY"]

client = genai.Client(api_key=key)

print("Flash models available to your key:")
for m in client.models.list():
    actions = getattr(m, "supported_actions", None) or []
    if "flash" in m.name and "generateContent" in actions:
        print("   ", m.name)

reply = client.models.generate_content(
    model="gemini-flash-latest",
    contents="Reply with exactly: Gemini is working",
)
print("\nTest reply:", reply.text)