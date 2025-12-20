# test_env.py
import os
from dotenv import load_dotenv

print("Current working directory:", os.getcwd())
print("Checking .env file...")

# Load .env
load_dotenv()

# Check các biến
env_vars = ["OPENAI_API_KEY", "GEMINI_API_KEY", "AWS_ACCESS_KEY_ID", "NEO4J_URI","NEO4J_USER", "NEO4J_PASSWORD"]
for var in env_vars:
    value = os.getenv(var)
    if value:
        print(f"✅ {var}: {value}")
    else:
        print(f"❌ {var}: Not found")