# API Documentation - Image Upload Endpoints

## Overview

This document describes the image upload API endpoints available in the Trading Assistant application.

## Endpoints

### 1. Upload Image with Analysis (Original Endpoint)

**Endpoint:** `POST /upload`

**Description:** Upload a trading chart image and automatically start analysis.

**Parameters:**
- `image` (File, Required): The trading chart image file

**Supported File Types:**
- PNG (.png)
- JPEG (.jpg, .jpeg)
- GIF (.gif)
- BMP (.bmp)
- WebP (.webp)

**Response:**
```json
{
  "success": true,
  "message": "Analysis started",
  "job_id": "analysis_1234567890"
}
```

**Example cURL:**
```bash
curl -X POST http://localhost:5000/upload \
  -F "image=@/path/to/chart.png"
```

---

### 2. Enhanced Image Upload API (New Endpoint)

**Endpoint:** `POST /api/upload-image`

**Description:** Enhanced image upload endpoint with optional analysis and permanent storage options.

**Parameters:**
- `image` (File, Required): The trading chart image file
- `auto_analyze` (Boolean, Optional): Whether to automatically start analysis (default: `false`)
- `save_permanently` (Boolean, Optional): Whether to save the image permanently (default: `false`)
- `filename` (String, Optional): Custom filename for the saved image

**Supported File Types:**
- PNG (.png)
- JPEG (.jpg, .jpeg)
- GIF (.gif)
- BMP (.bmp)
- WebP (.webp)

**File Size Limit:** 16MB

**Response (without auto_analyze):**
```json
{
  "success": true,
  "message": "Image uploaded successfully",
  "metadata": {
    "filename": "trading_chart_20250131_143025.png",
    "size_bytes": 245678,
    "size_kb": 239.92,
    "timestamp": "20250131_143025",
    "file_type": "png",
    "saved_permanently": false
  }
}
```

**Response (with auto_analyze and save_permanently):**
```json
{
  "success": true,
  "message": "Image uploaded and analysis started",
  "image_path": "/path/to/uploads/trading_chart_20250131_143025.png",
  "job_id": "analysis_20250131_143025",
  "metadata": {
    "filename": "trading_chart_20250131_143025.png",
    "size_bytes": 245678,
    "size_kb": 239.92,
    "timestamp": "20250131_143025",
    "file_type": "png",
    "saved_permanently": true
  }
}
```

**Error Response:**
```json
{
  "success": false,
  "error": "Error message describing what went wrong"
}
```

**Example cURL Commands:**

1. **Simple upload (temporary):**
```bash
curl -X POST http://localhost:5000/api/upload-image \
  -F "image=@/path/to/chart.png"
```

2. **Upload with auto-analysis:**
```bash
curl -X POST http://localhost:5000/api/upload-image \
  -F "image=@/path/to/chart.png" \
  -F "auto_analyze=true"
```

3. **Upload with permanent storage:**
```bash
curl -X POST http://localhost:5000/api/upload-image \
  -F "image=@/path/to/chart.png" \
  -F "save_permanently=true"
```

4. **Upload with custom filename:**
```bash
curl -X POST http://localhost:5000/api/upload-image \
  -F "image=@/path/to/chart.png" \
  -F "save_permanently=true" \
  -F "filename=btc_analysis_chart"
```

5. **Full featured upload:**
```bash
curl -X POST http://localhost:5000/api/upload-image \
  -F "image=@/path/to/chart.png" \
  -F "auto_analyze=true" \
  -F "save_permanently=true" \
  -F "filename=btc_analysis_chart"
```

**Example Python:**
```python
import requests

# Simple upload
url = "http://localhost:5000/api/upload-image"
files = {'image': open('chart.png', 'rb')}
response = requests.post(url, files=files)
print(response.json())

# Upload with options
data = {
    'auto_analyze': 'true',
    'save_permanently': 'true',
    'filename': 'my_chart'
}
response = requests.post(url, files=files, data=data)
print(response.json())
```

**Example JavaScript:**
```javascript
// Using fetch API
const formData = new FormData();
formData.append('image', fileInput.files[0]);
formData.append('auto_analyze', 'true');
formData.append('save_permanently', 'true');

fetch('http://localhost:5000/api/upload-image', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => console.log(data))
.catch(error => console.error('Error:', error));
```

---

### 3. List Uploaded Images

**Endpoint:** `GET /api/list-uploads`

**Description:** List all uploaded images stored in the uploads directory.

**Parameters:** None

**Response:**
```json
{
  "success": true,
  "images": [
    {
      "filename": "trading_chart_20250131_143025.png",
      "filepath": "/path/to/uploads/trading_chart_20250131_143025.png",
      "size_bytes": 245678,
      "size_kb": 239.92,
      "size_mb": 0.23,
      "uploaded_at": "2025-01-31 14:30:25",
      "modified_at": "2025-01-31 14:30:25"
    }
  ],
  "count": 1
}
```

**Example cURL:**
```bash
curl http://localhost:5000/api/list-uploads
```

**Example Python:**
```python
import requests

response = requests.get("http://localhost:5000/api/list-uploads")
data = response.json()

if data['success']:
    print(f"Found {data['count']} uploaded images")
    for image in data['images']:
        print(f"- {image['filename']} ({image['size_kb']} KB)")
```

---

## Progress Monitoring

After uploading an image with `auto_analyze=true`, you can monitor the analysis progress using Server-Sent Events (SSE):

**Endpoint:** `GET /progress`

**Example JavaScript:**
```javascript
const eventSource = new EventSource('http://localhost:5000/progress');

eventSource.onmessage = function(event) {
  const data = JSON.parse(event.data);
  console.log(`[${data.timestamp}] ${data.message}`);
};

eventSource.onerror = function(error) {
  console.error('EventSource error:', error);
  eventSource.close();
};
```

---

## Error Codes

| HTTP Status | Description |
|-------------|-------------|
| 200 | Success |
| 400 | Bad Request (missing file, invalid file type, etc.) |
| 500 | Internal Server Error |

---

## Notes

1. **Temporary vs Permanent Storage:**
   - Temporary: Images are stored in system temp directory and deleted after processing
   - Permanent: Images are stored in `uploads/` directory and persist

2. **File Naming:**
   - Default: `trading_chart_YYYYMMDD_HHMMSS.<ext>`
   - Custom: Sanitized version of provided filename with original extension

3. **Analysis Job ID:**
   - Format: `analysis_YYYYMMDD_HHMMSS`
   - Can be used to track analysis progress

4. **Maximum File Size:**
   - Default: 16MB
   - Can be configured via Flask's `MAX_CONTENT_LENGTH` setting

---

## Integration Examples

### Web Form Upload
```html
<!DOCTYPE html>
<html>
<head>
    <title>Upload Trading Chart</title>
</head>
<body>
    <h1>Upload Trading Chart</h1>
    <form id="uploadForm">
        <input type="file" id="imageFile" accept="image/*" required>
        <br><br>
        <label>
            <input type="checkbox" id="autoAnalyze"> Auto-analyze
        </label>
        <label>
            <input type="checkbox" id="savePermanently"> Save permanently
        </label>
        <br><br>
        <button type="submit">Upload</button>
    </form>
    <div id="result"></div>

    <script>
        document.getElementById('uploadForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = new FormData();
            formData.append('image', document.getElementById('imageFile').files[0]);
            formData.append('auto_analyze', document.getElementById('autoAnalyze').checked ? 'true' : 'false');
            formData.append('save_permanently', document.getElementById('savePermanently').checked ? 'true' : 'false');
            
            try {
                const response = await fetch('http://localhost:5000/api/upload-image', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                document.getElementById('result').innerHTML = 
                    `<pre>${JSON.stringify(data, null, 2)}</pre>`;
            } catch (error) {
                document.getElementById('result').innerHTML = 
                    `<p style="color: red;">Error: ${error.message}</p>`;
            }
        });
    </script>
</body>
</html>
```

---

## Testing

You can test the endpoints using the provided test scripts:

```bash
# Test simple upload
curl -X POST http://localhost:5000/api/upload-image \
  -F "image=@test_chart.png"

# Test with all features
curl -X POST http://localhost:5000/api/upload-image \
  -F "image=@test_chart.png" \
  -F "auto_analyze=true" \
  -F "save_permanently=true" \
  -F "filename=test_btc_chart"

# List uploads
curl http://localhost:5000/api/list-uploads
```

---

## Troubleshooting

### Common Issues

1. **"No image file provided"**
   - Ensure the form field name is `image`
   - Check that the file is actually being sent

2. **"Invalid file type"**
   - Verify the file extension is one of the supported types
   - Check the file MIME type

3. **"Server error"**
   - Check server logs for detailed error messages
   - Verify sufficient disk space
   - Ensure write permissions for uploads directory

4. **File size too large**
   - Check the file size (max 16MB by default)
   - Consider compressing the image before upload


