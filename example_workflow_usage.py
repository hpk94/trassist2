"""Example usage of the Trading Chart Analysis Workflow API."""

import requests
import json
import time
from pathlib import Path

# API base URL
BASE_URL = "http://localhost:8002"

def test_health_check():
    """Test the health check endpoint."""
    print("🔍 Testing health check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def upload_chart(image_path: str):
    """Upload a chart and start analysis workflow."""
    print(f"📤 Uploading chart: {image_path}")
    
    if not Path(image_path).exists():
        print(f"❌ Error: File {image_path} does not exist")
        return None
    
    files = {"file": (image_path, open(image_path, "rb"), "image/png")}
    
    try:
        response = requests.post(f"{BASE_URL}/upload", files=files)
        files["file"][1].close()
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Chart uploaded and analyzed successfully!")
            print(f"📋 Analysis ID: {result['analysis_id']}")
            print(f"📊 Symbol: {result['results']['symbol']}")
            print(f"📈 Signal: {result['results']['opening_signal']['direction']}")
            print(f"🎯 Confidence: {result['results']['opening_signal']['confidence']}")
            return result['analysis_id']
        else:
            print(f"❌ Error: {response.json()}")
            return None
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return None

def get_analysis(analysis_id: str):
    """Retrieve analysis results by ID."""
    print(f"📋 Retrieving analysis: {analysis_id}")
    
    try:
        response = requests.get(f"{BASE_URL}/analysis/{analysis_id}")
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Analysis retrieved successfully!")
            print(f"📊 Symbol: {result['results']['symbol']}")
            print(f"⏰ Timeframe: {result['results']['timeframe']}")
            print(f"📈 Signal: {result['results']['opening_signal']['direction']}")
            print(f"🎯 Confidence: {result['results']['opening_signal']['confidence']}")
            print(f"💭 Reasoning: {result['results']['opening_signal']['reasoning']}")
            return result['results']
        else:
            print(f"❌ Error: {response.json()}")
            return None
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return None

def reanalyze_chart(analysis_id: str):
    """Re-analyze an existing chart."""
    print(f"🔄 Re-analyzing chart: {analysis_id}")
    
    try:
        response = requests.post(f"{BASE_URL}/analysis/{analysis_id}/reanalyze")
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Chart reanalyzed successfully!")
            print(f"📈 New Signal: {result['results']['opening_signal']['direction']}")
            print(f"🎯 New Confidence: {result['results']['opening_signal']['confidence']}")
            print(f"💭 New Reasoning: {result['results']['opening_signal']['reasoning']}")
            return result['results']
        else:
            print(f"❌ Error: {response.json()}")
            return None
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return None

def list_analyses():
    """List all analyses."""
    print("📋 Listing all analyses...")
    
    try:
        response = requests.get(f"{BASE_URL}/analyses")
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Found {len(result['analyses'])} analyses:")
            for i, analysis in enumerate(result['analyses'], 1):
                print(f"  {i}. ID: {analysis['analysis_id'][:8]}...")
                print(f"     Symbol: {analysis['symbol']}")
                print(f"     Signal: {analysis['signal_direction']}")
                print(f"     Confidence: {analysis['confidence']}")
                print(f"     Time: {analysis['timestamp']}")
                if analysis.get('reanalyzed_at'):
                    print(f"     Reanalyzed: {analysis['reanalyzed_at']}")
                print()
            return result['analyses']
        else:
            print(f"❌ Error: {response.json()}")
            return None
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return None

def delete_analysis(analysis_id: str):
    """Delete an analysis."""
    print(f"🗑️ Deleting analysis: {analysis_id}")
    
    try:
        response = requests.delete(f"{BASE_URL}/analysis/{analysis_id}")
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Analysis deleted successfully!")
            return True
        else:
            print(f"❌ Error: {response.json()}")
            return False
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

def demonstrate_workflow():
    """Demonstrate the complete workflow."""
    print("🚀 === Trading Chart Analysis Workflow Demo ===\n")
    
    # Step 1: Health check
    test_health_check()
    
    # Step 2: Upload chart
    test_image = "image_example.png"
    if not Path(test_image).exists():
        print(f"❌ Test image {test_image} not found. Please provide a valid image path.")
        return
    
    analysis_id = upload_chart(test_image)
    if not analysis_id:
        print("❌ Failed to upload chart. Exiting.")
        return
    
    print("\n" + "="*50 + "\n")
    
    # Step 3: Retrieve analysis
    get_analysis(analysis_id)
    
    print("\n" + "="*50 + "\n")
    
    # Step 4: Wait a moment, then reanalyze
    print("⏳ Waiting 2 seconds before reanalysis...")
    time.sleep(2)
    reanalyze_chart(analysis_id)
    
    print("\n" + "="*50 + "\n")
    
    # Step 5: List all analyses
    list_analyses()
    
    print("\n" + "="*50 + "\n")
    
    # Step 6: Ask user if they want to delete
    print("🤔 Would you like to delete this analysis? (This will remove the analysis and image files)")
    print("   Uncomment the line below to enable deletion:")
    print(f"   # delete_analysis('{analysis_id}')")
    
    # Uncomment the next line to actually delete the analysis
    # delete_analysis(analysis_id)

if __name__ == "__main__":
    demonstrate_workflow()
