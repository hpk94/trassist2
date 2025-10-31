#!/usr/bin/env python3
"""Quick test script to verify DeepSeek configuration"""

import os
from dotenv import load_dotenv
import litellm

# Load environment variables
load_dotenv()

print("=" * 60)
print("DeepSeek Configuration Test")
print("=" * 60)

# Check API keys
deepseek_key = os.getenv("DEEPSEEK_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")

print("\n1. Checking API Keys:")
print(f"   DEEPSEEK_API_KEY: {'✅ Set' if deepseek_key else '❌ Not found'}")
if deepseek_key:
    print(f"   Key starts with: {deepseek_key[:8]}...")
print(f"   OPENAI_API_KEY: {'✅ Set' if openai_key else '❌ Not found'}")
if openai_key:
    print(f"   Key starts with: {openai_key[:8]}...")

# Check model configuration
vision_model = os.getenv("LITELLM_VISION_MODEL", os.getenv("LITELLM_MODEL", "gpt-4o"))
text_model = os.getenv("LITELLM_TEXT_MODEL", os.getenv("LITELLM_MODEL", "gpt-4o"))

print("\n2. Model Configuration:")
print(f"   Vision Model (for charts): {vision_model}")
print(f"   Text Model (for gates): {text_model}")

# Test DeepSeek connection
print("\n3. Testing DeepSeek Connection:")
if not deepseek_key:
    print("   ❌ Cannot test - DEEPSEEK_API_KEY not set")
else:
    try:
        print("   Sending test request to DeepSeek...")
        response = litellm.completion(
            model="deepseek/deepseek-chat",
            messages=[{"role": "user", "content": "Reply with just 'OK' if you receive this."}],
            max_tokens=10
        )
        result = response.choices[0].message.content.strip()
        print(f"   ✅ DeepSeek responded: {result}")
        print("   ✅ DeepSeek is working correctly!")
    except Exception as e:
        print(f"   ❌ Error testing DeepSeek: {str(e)}")
        if "authentication" in str(e).lower() or "api key" in str(e).lower():
            print("   💡 Check your DEEPSEEK_API_KEY in .env file")

# Test OpenAI connection (if using GPT-4o)
if vision_model.startswith("gpt-") and openai_key:
    print("\n4. Testing OpenAI Connection:")
    try:
        print("   Sending test request to OpenAI...")
        response = litellm.completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Reply with just 'OK' if you receive this."}],
            max_tokens=10
        )
        result = response.choices[0].message.content.strip()
        print(f"   ✅ OpenAI responded: {result}")
        print("   ✅ OpenAI is working correctly!")
    except Exception as e:
        print(f"   ❌ Error testing OpenAI: {str(e)}")

# Summary
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)

all_good = True
if not deepseek_key:
    print("❌ Add DEEPSEEK_API_KEY to your .env file")
    all_good = False
if not openai_key and vision_model.startswith("gpt-"):
    print("❌ Add OPENAI_API_KEY to your .env file")
    all_good = False
if text_model != "deepseek/deepseek-chat":
    print("⚠️  LITELLM_TEXT_MODEL is not set to deepseek/deepseek-chat")
    print("   Add to .env: LITELLM_TEXT_MODEL=deepseek/deepseek-chat")
    all_good = False

if all_good:
    print("✅ Everything looks good!")
    print("\n🚀 Ready to run: python app.py")
    print("\n💰 Cost savings enabled:")
    print("   - Chart analysis: GPT-4o (~$18/month)")
    print("   - Trade gates: DeepSeek (~$0.42/month)")
    print("   - Total: ~$18.42/month (vs $36 with GPT-4o only)")
else:
    print("\n⚠️  Please fix the issues above and run this test again")

print("=" * 60)

