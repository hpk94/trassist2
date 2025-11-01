#!/usr/bin/env python3
"""
Example: Upload a trading chart and receive it in Telegram

This script demonstrates how to upload a trading chart image
that will be automatically sent to your Telegram chat.
"""

import requests
import os
import sys

# Configuration
BASE_URL = "http://localhost:5001"
UPLOAD_ENDPOINT = f"{BASE_URL}/api/upload-image"

def upload_chart_with_telegram(image_path, auto_analyze=True, save_permanently=False):
    """
    Upload a trading chart image and send to Telegram
    
    Args:
        image_path: Path to the image file
        auto_analyze: Whether to automatically analyze the chart (default: True)
        save_permanently: Whether to save the image permanently (default: False)
    
    Returns:
        Response data from the server
    """
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"❌ Error: File not found: {image_path}")
        return None
    
    print(f"📤 Uploading: {image_path}")
    print(f"   Auto-analyze: {auto_analyze}")
    print(f"   Save permanently: {save_permanently}")
    print()
    
    try:
        # Prepare the multipart form data
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {
                'auto_analyze': 'true' if auto_analyze else 'false',
                'save_permanently': 'true' if save_permanently else 'false'
            }
            
            # Send the request
            response = requests.post(UPLOAD_ENDPOINT, files=files, data=data)
            response.raise_for_status()
            
            result = response.json()
            
            # Display results
            print("✅ Upload successful!")
            print()
            print("📊 Response:")
            print(f"   Success: {result.get('success')}")
            print(f"   Message: {result.get('message')}")
            
            if result.get('telegram_sent'):
                print(f"   ✅ Sent to Telegram!")
            else:
                print(f"   ⚠️  Not sent to Telegram (check your configuration)")
            
            print()
            print("📋 Metadata:")
            metadata = result.get('metadata', {})
            print(f"   Filename: {metadata.get('filename')}")
            print(f"   Size: {metadata.get('size_kb')} KB")
            print(f"   Timestamp: {metadata.get('timestamp')}")
            print(f"   Type: {metadata.get('file_type')}")
            
            if result.get('job_id'):
                print()
                print(f"🔄 Analysis started: {result.get('job_id')}")
                print(f"   Results will be sent to Telegram when complete")
            
            print()
            print("💬 Check your Telegram for:")
            print("   1. The uploaded chart image")
            if auto_analyze:
                print("   2. Analysis results (when complete)")
            
            return result
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Error: Cannot connect to server at {BASE_URL}")
        print(f"   Make sure the server is running:")
        print(f"   ./start_server.sh")
        return None
        
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
        if response.text:
            print(f"   Response: {response.text}")
        return None
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    """Main function"""
    
    # Check if image path was provided
    if len(sys.argv) < 2:
        print("Usage: python example_telegram_upload.py <image_path>")
        print()
        print("Example:")
        print("  python example_telegram_upload.py BTCUSDT.P_2025-09-02_22-55-40_545b4.png")
        print()
        
        # Try to find a default image
        default_images = [
            "BTCUSDT.P_2025-09-02_22-55-40_545b4.png",
            "BTCUSDT.P_2025-09-02_19-28-40_20635.png",
            "image_example.png"
        ]
        
        for img in default_images:
            if os.path.exists(img):
                print(f"Found default image: {img}")
                print(f"Using this image for demo...")
                print()
                image_path = img
                break
        else:
            print("❌ No default images found")
            return 1
    else:
        image_path = sys.argv[1]
    
    # Upload the image
    result = upload_chart_with_telegram(
        image_path=image_path,
        auto_analyze=True,      # Automatically analyze the chart
        save_permanently=False   # Don't save permanently (use temp file)
    )
    
    if result:
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())

