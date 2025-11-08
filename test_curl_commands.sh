#!/bin/bash

# Test cURL commands for image upload API
# All commands on single lines to avoid line continuation issues

echo "🧪 Testing Image Upload API Endpoints"
echo "======================================"
echo ""

# Test 1: List uploads
echo "Test 1: List uploads"
curl -s http://localhost:5001/api/list-uploads | python -m json.tool
echo ""
echo ""

# Test 2: Simple upload
echo "Test 2: Simple upload (temporary storage, no analysis)"
curl -s -X POST http://localhost:5001/api/upload-image -F "image=@BTCUSDT.P_2025-09-02_19-28-40_20635.png" | python -m json.tool
echo ""
echo ""

# Test 3: Upload with auto-analysis
echo "Test 3: Upload with auto-analysis"
curl -s -X POST http://localhost:5001/api/upload-image -F "image=@BTCUSDT.P_2025-09-02_19-28-40_20635.png" -F "auto_analyze=true" | python -m json.tool
echo ""
echo ""

# Test 4: Upload with permanent storage
echo "Test 4: Upload with permanent storage"
curl -s -X POST http://localhost:5001/api/upload-image -F "image=@BTCUSDT.P_2025-09-02_19-28-40_20635.png" -F "save_permanently=true" | python -m json.tool
echo ""
echo ""

# Test 5: Upload with custom filename
echo "Test 5: Upload with custom filename"
curl -s -X POST http://localhost:5001/api/upload-image -F "image=@BTCUSDT.P_2025-09-02_19-28-40_20635.png" -F "save_permanently=true" -F "filename=test_btc_chart" | python -m json.tool
echo ""
echo ""

# Test 6: Full featured upload
echo "Test 6: Full featured upload (all options)"
curl -s -X POST http://localhost:5001/api/upload-image -F "image=@BTCUSDT.P_2025-09-02_19-28-40_20635.png" -F "auto_analyze=true" -F "save_permanently=true" -F "filename=btc_full_test" | python -m json.tool
echo ""
echo ""

echo "✅ All tests completed!"


