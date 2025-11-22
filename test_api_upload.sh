#!/bin/bash

# Test script for /api/upload-image endpoint
# Usage: ./test_api_upload.sh [image_path]

set -e

# Configuration
API_URL="${API_URL:-http://localhost:5000/api/upload-image}"
IMAGE_PATH="${1:-uploads/test_chart_custom_name.png}"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}Testing API Image Upload with Analysis${NC}"
echo -e "${YELLOW}========================================${NC}"
echo ""

# Check if image exists
if [ ! -f "$IMAGE_PATH" ]; then
    echo -e "${RED}Error: Image file not found: $IMAGE_PATH${NC}"
    echo "Usage: $0 [image_path]"
    exit 1
fi

echo -e "${GREEN}✓${NC} Image found: $IMAGE_PATH"
echo -e "${GREEN}✓${NC} API endpoint: $API_URL"
echo ""

# Test 1: Upload only (no analysis)
echo -e "${YELLOW}Test 1: Upload image without analysis${NC}"
echo "----------------------------------------"
response=$(curl -s -X POST "$API_URL" \
  -F "image=@$IMAGE_PATH" \
  -F "save_permanently=false")

echo "$response" | jq '.'
echo ""

# Test 2: Upload with auto-analysis
echo -e "${YELLOW}Test 2: Upload image with auto-analysis${NC}"
echo "----------------------------------------"
echo "This will trigger the full analysis pipeline:"
echo "  1. Image upload"
echo "  2. LLM chart analysis"
echo "  3. Market data fetching"
echo "  4. Signal validation polling"
echo "  5. Trade gate decision"
echo "  6. Telegram notifications"
echo ""
echo "⏳ Starting analysis (this may take several minutes)..."
echo ""

response=$(curl -s -X POST "$API_URL" \
  -F "image=@$IMAGE_PATH" \
  -F "auto_analyze=true" \
  -F "save_permanently=false")

echo "$response" | jq '.'

# Extract job_id
job_id=$(echo "$response" | jq -r '.job_id // empty')
telegram_sent=$(echo "$response" | jq -r '.telegram_sent // false')

echo ""
echo -e "${GREEN}✓${NC} Upload successful!"

if [ -n "$job_id" ]; then
    echo -e "${GREEN}✓${NC} Job ID: $job_id"
fi

if [ "$telegram_sent" = "true" ]; then
    echo -e "${GREEN}✓${NC} Image sent to Telegram"
else
    echo -e "${YELLOW}⚠${NC} Image not sent to Telegram (check configuration)"
fi

echo ""
echo -e "${YELLOW}========================================${NC}"
echo -e "${GREEN}Analysis is running in the background${NC}"
echo -e "${YELLOW}========================================${NC}"
echo ""
echo "You will receive Telegram notifications when:"
echo "  • Signal is invalidated (❌ TRADE INVALIDATED)"
echo "  • Trade is approved (🚀 VALID TRADE SIGNAL)"
echo "  • Any error occurs (❌ Analysis Error)"
echo ""
echo "Check the server logs for detailed progress:"
echo "  tail -f <server_log_file>"
echo ""
echo -e "${GREEN}Test complete!${NC}"




