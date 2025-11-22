#!/usr/bin/env python3
"""Quick test script to verify Gemini configuration"""

import os
from dotenv import load_dotenv
import litellm

# Load environment variables
load_dotenv()

print("=" * 60)
print("Gemini Configuration Test")
print("=" * 60)

# Check API keys
gemini_key = os.getenv("GEMINI_API_KEY")

print("\n1. Checking API Keys:")
print(f"   GEMINI_API_KEY: {'✅ Set' if gemini_key else '❌ Not found'}")
if gemini_key:
    # Don't print the full key for security, just first few chars if available
    safe_preview = gemini_key[:8] if len(gemini_key) > 8 else "***"
    print(f"   Key starts with: {safe_preview}...")

# Check model configuration
# Usually we care if LITELLM_MODEL is set to gemini/gemini-1.5-pro or if we explicitly test it
# target_model = "gemini/gemini-2.5-flash-lite"
target_model = "gemini/gemini-3-pro-preview"

print("\n2. Testing Gemini Connection:")
if not gemini_key:
    print("   ❌ Cannot test - GEMINI_API_KEY not set")
else:
    try:
        print(f"   Sending test request to {target_model}...")
        response = litellm.completion(
            model=target_model,
            messages=[{"role": "user", "content": "Reply with just 'OK' if you receive this."}],
            max_tokens=10
        )
        
        # Debug output for full response
        # print(f"Full response: {response}")
        
        if response.choices and len(response.choices) > 0:
            message = response.choices[0].message
            content = message.content
            
            if content is not None:
                result = content.strip()
                print(f"   ✅ Gemini responded: {result}")
                print("   ✅ Gemini is working correctly!")
            else:
                print(f"   ⚠️  Gemini responded but content is None. Full message object: {message}")
                # Try to inspect if there's a refusal or safety block
                try:
                    print(f"   🔍 Safety ratings/other fields: {response.choices[0]}")
                except:
                    pass
        else:
            print(f"   ⚠️  No choices in response: {response}")
            
    except Exception as e:
        print(f"   ❌ Error testing Gemini: {str(e)}")
        if "authentication" in str(e).lower() or "api key" in str(e).lower():
            print("   💡 Check your GEMINI_API_KEY in .env file")

# Summary
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)

if gemini_key:
    print("✅ Gemini configuration looks good!")
else:
    print("❌ Please add GEMINI_API_KEY to your .env file")

print("=" * 60)
