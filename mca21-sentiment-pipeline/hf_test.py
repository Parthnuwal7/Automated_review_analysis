# Test script: test_hf_token.py
from huggingface_hub import whoami
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

try:
    token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    if token:
        user_info = whoami(token=token)
        print(f"✅ HuggingFace token is valid!")
        print(f"👤 Logged in as: {user_info['name']}")
        print(f"📧 Email: {user_info.get('email', 'N/A')}")
    else:
        print("❌ No HuggingFace token found in environment")
except Exception as e:
    print(f"❌ Token validation failed: {e}")

# Test cache directories
cache_dirs = [
    "TRANSFORMERS_CACHE",
    "HF_HOME", 
    "HF_HUB_CACHE",
    "SENTENCE_TRANSFORMERS_HOME"
]

for cache_dir in cache_dirs:
    path = os.getenv(cache_dir)
    if path:
        print(f"✅ {cache_dir}: {path}")
    else:
        print(f"❌ {cache_dir}: Not set")
