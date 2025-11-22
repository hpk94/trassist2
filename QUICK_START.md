# Quick Start - Image Upload API

## 🚀 Getting Started in 5 Minutes

### Step 1: Ensure You're on the Feature Branch
```bash
git checkout feature/image-upload-endpoint
```

### Step 2: Start the Flask Server
```bash
python web_app.py
```

The server will start on `http://localhost:5000`

### Step 3: Choose Your Testing Method

#### Option A: Interactive Web Interface (Easiest)
Simply open the demo page in your browser:
```bash
open upload_example.html
```

**Features:**
- 📤 Drag & drop or click to upload
- ✅ Auto-analyze option
- 💾 Save permanently option
- 📝 Custom filename
- 📊 Live results display

#### Option B: Automated Test Suite
Run the comprehensive test script:
```bash
python test_upload_api.py
```

This will run 8 different test scenarios automatically.

#### Option C: Command Line (cURL)
```bash
# Simple upload
curl -X POST http://localhost:5000/api/upload-image \
  -F "image=@BTCUSDT.P_2025-09-02_22-55-40_545b4.png"

# Upload with auto-analysis
curl -X POST http://localhost:5000/api/upload-image \
  -F "image=@BTCUSDT.P_2025-09-02_22-55-40_545b4.png" \
  -F "auto_analyze=true"

# List all uploads
curl http://localhost:5000/api/list-uploads
```

---

## 📁 New Endpoints

### 1. POST `/api/upload-image`
Upload a trading chart image with options

**Parameters:**
- `image` (required) - The image file
- `auto_analyze` (optional) - Start analysis automatically
- `save_permanently` (optional) - Save to permanent storage
- `filename` (optional) - Custom filename

**Example Response:**
```json
{
  "success": true,
  "message": "Image uploaded successfully",
  "metadata": {
    "filename": "trading_chart_20250131_143025.png",
    "size_kb": 239.92,
    "timestamp": "20250131_143025"
  }
}
```

### 2. GET `/api/list-uploads`
List all permanently uploaded images

**Example Response:**
```json
{
  "success": true,
  "count": 3,
  "images": [
    {
      "filename": "trading_chart_20250131_143025.png",
      "size_kb": 239.92,
      "uploaded_at": "2025-01-31 14:30:25"
    }
  ]
}
```

---

## 📚 Documentation Files

| File | Description |
|------|-------------|
| `API_DOCUMENTATION.md` | Complete API reference with examples |
| `FEATURE_SUMMARY.md` | Overview of all changes and features |
| `QUICK_START.md` | This file - quick start guide |
| `test_upload_api.py` | Automated test suite |
| `upload_example.html` | Interactive web demo |

---

## ✨ Key Features

1. **Flexible Storage**
   - Temporary: Auto-deleted after processing
   - Permanent: Saved to `uploads/` directory

2. **Auto-Analysis**
   - Optionally trigger trading analysis automatically
   - Monitor progress via SSE endpoint

3. **Validation**
   - File type validation (PNG, JPG, GIF, BMP, WebP)
   - File size limit (16MB)
   - Secure filename handling

4. **Developer-Friendly**
   - RESTful JSON API
   - Comprehensive error messages
   - Consistent response format

---

## 🧪 Testing Examples

### Python Example
```python
import requests

url = "http://localhost:5000/api/upload-image"
files = {'image': open('chart.png', 'rb')}
data = {
    'auto_analyze': 'true',
    'save_permanently': 'true'
}

response = requests.post(url, files=files, data=data)
print(response.json())
```

### JavaScript Example
```javascript
const formData = new FormData();
formData.append('image', fileInput.files[0]);
formData.append('auto_analyze', 'true');

fetch('http://localhost:5000/api/upload-image', {
  method: 'POST',
  body: formData
})
.then(res => res.json())
.then(data => console.log(data));
```

---

## 🔧 Troubleshooting

### Server not running?
```bash
# Make sure you're in the project directory
cd /Users/hpk/trassist2

# Activate virtual environment if needed
source venv/bin/activate

# Start the server
python web_app.py
```

### Test image not found?
The test script looks for these images:
- `image_example.png`
- `BTCUSDT.P_2025-09-02_22-55-40_545b4.png`
- `BTCUSDT.P_2025-09-02_19-28-40_20635.png`

Make sure at least one exists in the project directory.

### Port already in use?
Change the port in `web_app.py` or kill the existing process:
```bash
lsof -ti:5000 | xargs kill -9
```

---

## 📊 What Happens Next?

1. **Upload**: Image is validated and saved
2. **Storage**: Stored temporarily or permanently based on option
3. **Analysis** (if enabled): Background thread processes the image
4. **Progress**: Monitor via `/progress` SSE endpoint
5. **Results**: View in `llm_outputs/` directory

---

## 🎯 Next Steps

1. **Test the endpoint** using any of the three methods above
2. **Read the full docs** in `API_DOCUMENTATION.md`
3. **Integrate** into your application
4. **Customize** the HTML demo for your needs

---

## 📞 Need Help?

- **API Reference**: See `API_DOCUMENTATION.md`
- **Feature Details**: See `FEATURE_SUMMARY.md`
- **Run Tests**: `python test_upload_api.py`
- **Try Demo**: Open `upload_example.html`

---

## 🎉 You're All Set!

The image upload API is ready to use. Start with the web interface for a quick visual test, then integrate the API into your application.

**Happy coding! 🚀**




