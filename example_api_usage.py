"""Example usage of the FastAPI trading chart analysis API."""

import requests
import json
from pathlib import Path

# API base URL
BASE_URL = "http://localhost:8001"

def test_health_check():
    """Test the health check endpoint."""
    print("Testing health check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def upload_and_analyze_image(image_path: str, max_cycles: int = None):
    """Upload and analyze a trading chart image."""
    print(f"Uploading and analyzing image: {image_path}")
    
    # Check if file exists
    if not Path(image_path).exists():
        print(f"Error: File {image_path} does not exist")
        return None
    
    # Prepare the request
    files = {"file": (image_path, open(image_path, "rb"), "image/png")}
    data = {}
    if max_cycles:
        data["max_cycles"] = max_cycles
    
    try:
        response = requests.post(f"{BASE_URL}/upload", files=files, data=data)
        files["file"][1].close()  # Close the file
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("Analysis completed successfully!")
            print(f"Analysis ID: {result['analysis_id']}")
            print(f"Results: {json.dumps(result['results'], indent=2)}")
            return result
        else:
            print(f"Error: {response.json()}")
            return None
            
    except Exception as e:
        print(f"Request failed: {e}")
        return None

def analyze_without_saving(image_path: str, max_cycles: int = None):
    """Analyze an image without saving it permanently."""
    print(f"Analyzing image without saving: {image_path}")
    
    # Check if file exists
    if not Path(image_path).exists():
        print(f"Error: File {image_path} does not exist")
        return None
    
    # Prepare the request
    files = {"file": (image_path, open(image_path, "rb"), "image/png")}
    data = {}
    if max_cycles:
        data["max_cycles"] = max_cycles
    
    try:
        response = requests.post(f"{BASE_URL}/analyze", files=files, data=data)
        files["file"][1].close()  # Close the file
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("Analysis completed successfully!")
            print(f"Analysis ID: {result['analysis_id']}")
            print(f"Results: {json.dumps(result['results'], indent=2)}")
            return result
        else:
            print(f"Error: {response.json()}")
            return None
            
    except Exception as e:
        print(f"Request failed: {e}")
        return None

def list_uploads():
    """List all uploaded files."""
    print("Listing uploaded files...")
    try:
        response = requests.get(f"{BASE_URL}/uploads")
        print(f"Status: {response.status_code}")
        print(f"Uploads: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    # Test the API
    print("=== Trading Chart Analysis API Test ===\n")
    
    # Test health check
    test_health_check()
    
    # Test with an existing image (if available)
    test_image = "image_example.png"
    if Path(test_image).exists():
        print("=== Testing Image Upload and Analysis ===")
        upload_and_analyze_image(test_image, max_cycles=5)
        print()
        
        print("=== Testing Analysis Without Saving ===")
        analyze_without_saving(test_image, max_cycles=5)
        print()
        
        print("=== Listing Uploads ===")
        list_uploads()
    else:
        print(f"Test image {test_image} not found. Please provide a valid image path.")
        print("You can test with any PNG, JPG, or JPEG image file.")
