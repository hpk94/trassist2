#!/usr/bin/env python3
"""
Test script for the image upload API endpoints
"""

import requests
import os
import sys
from pathlib import Path

# Base URL for the API
BASE_URL = "http://localhost:5000"

def test_simple_upload(image_path):
    """Test simple image upload without any options"""
    print("\n" + "="*60)
    print("TEST 1: Simple Upload (temporary, no analysis)")
    print("="*60)
    
    url = f"{BASE_URL}/api/upload-image"
    
    with open(image_path, 'rb') as f:
        files = {'image': f}
        response = requests.post(url, files=files)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    return response.json()

def test_upload_with_analysis(image_path):
    """Test image upload with automatic analysis"""
    print("\n" + "="*60)
    print("TEST 2: Upload with Auto-Analysis")
    print("="*60)
    
    url = f"{BASE_URL}/api/upload-image"
    
    with open(image_path, 'rb') as f:
        files = {'image': f}
        data = {'auto_analyze': 'true'}
        response = requests.post(url, files=files, data=data)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    return response.json()

def test_upload_permanent(image_path):
    """Test image upload with permanent storage"""
    print("\n" + "="*60)
    print("TEST 3: Upload with Permanent Storage")
    print("="*60)
    
    url = f"{BASE_URL}/api/upload-image"
    
    with open(image_path, 'rb') as f:
        files = {'image': f}
        data = {'save_permanently': 'true'}
        response = requests.post(url, files=files, data=data)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    return response.json()

def test_upload_custom_filename(image_path):
    """Test image upload with custom filename"""
    print("\n" + "="*60)
    print("TEST 4: Upload with Custom Filename")
    print("="*60)
    
    url = f"{BASE_URL}/api/upload-image"
    
    with open(image_path, 'rb') as f:
        files = {'image': f}
        data = {
            'save_permanently': 'true',
            'filename': 'test_chart_custom_name'
        }
        response = requests.post(url, files=files, data=data)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    return response.json()

def test_upload_full_featured(image_path):
    """Test image upload with all features enabled"""
    print("\n" + "="*60)
    print("TEST 5: Upload with All Features")
    print("="*60)
    
    url = f"{BASE_URL}/api/upload-image"
    
    with open(image_path, 'rb') as f:
        files = {'image': f}
        data = {
            'auto_analyze': 'true',
            'save_permanently': 'true',
            'filename': 'test_full_featured_chart'
        }
        response = requests.post(url, files=files, data=data)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    return response.json()

def test_list_uploads():
    """Test listing uploaded images"""
    print("\n" + "="*60)
    print("TEST 6: List Uploaded Images")
    print("="*60)
    
    url = f"{BASE_URL}/api/list-uploads"
    response = requests.get(url)
    
    print(f"Status Code: {response.status_code}")
    data = response.json()
    
    if data.get('success'):
        print(f"\nFound {data['count']} uploaded images:")
        for img in data.get('images', []):
            print(f"  - {img['filename']} ({img['size_kb']} KB) uploaded at {img['uploaded_at']}")
    else:
        print(f"Error: {data.get('error')}")
    
    return data

def test_invalid_file():
    """Test uploading an invalid file type"""
    print("\n" + "="*60)
    print("TEST 7: Invalid File Type")
    print("="*60)
    
    url = f"{BASE_URL}/api/upload-image"
    
    # Create a temporary text file
    temp_file = 'temp_test.txt'
    with open(temp_file, 'w') as f:
        f.write('This is not an image')
    
    try:
        with open(temp_file, 'rb') as f:
            files = {'image': ('test.txt', f)}
            response = requests.post(url, files=files)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)

def test_no_file():
    """Test uploading without a file"""
    print("\n" + "="*60)
    print("TEST 8: No File Provided")
    print("="*60)
    
    url = f"{BASE_URL}/api/upload-image"
    response = requests.post(url)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

def check_server_running():
    """Check if the Flask server is running"""
    try:
        response = requests.get(f"{BASE_URL}/")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

def main():
    # Check if server is running
    if not check_server_running():
        print("ERROR: Flask server is not running!")
        print(f"Please start the server first with: python web_app.py")
        sys.exit(1)
    
    # Check if image file exists
    test_images = [
        'image_example.png',
        'BTCUSDT.P_2025-09-02_22-55-40_545b4.png',
        'BTCUSDT.P_2025-09-02_19-28-40_20635.png'
    ]
    
    image_path = None
    for img in test_images:
        if os.path.exists(img):
            image_path = img
            break
    
    if not image_path:
        print("ERROR: No test image found!")
        print(f"Please provide one of these images in the current directory:")
        for img in test_images:
            print(f"  - {img}")
        sys.exit(1)
    
    print("="*60)
    print("IMAGE UPLOAD API TEST SUITE")
    print("="*60)
    print(f"Using test image: {image_path}")
    print(f"Server URL: {BASE_URL}")
    
    try:
        # Run all tests
        test_simple_upload(image_path)
        test_upload_with_analysis(image_path)
        test_upload_permanent(image_path)
        test_upload_custom_filename(image_path)
        test_upload_full_featured(image_path)
        test_list_uploads()
        test_invalid_file()
        test_no_file()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED")
        print("="*60)
        
    except Exception as e:
        print(f"\nERROR during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()

