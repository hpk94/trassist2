# Apple Shortcuts - Quick Start Guide

## 5-Minute Setup

### Step 1: Get Your Server Address (1 min)

On your Mac, run:
```bash
ifconfig | grep "inet " | grep -v 127.0.0.1
```

Or use ngrok for external access:
```bash
ngrok http 5000
```

Your URL will be:
- **Local**: `http://192.168.1.XXX:5000/api/upload-image`
- **Ngrok**: `https://abc123.ngrok.io/api/upload-image`

---

### Step 2: Create Shortcut (3 min)

Open **Shortcuts App** on iPhone/iPad:

1. **Tap "+" → Add Action**

2. **Add "Select Photos"**
   - Turn OFF "Select Multiple"

3. **Add "Get Contents of URL"**
   - URL: `http://YOUR_IP:5000/api/upload-image`
   - Method: **POST**
   - Request Body: **Form**
   - Add field:
     - Key: `image`
     - Type: **File**
     - Value: [Select Photos]
   - Add field (optional):
     - Key: `auto_analyze`
     - Type: **Text**
     - Value: `true`

4. **Add "Show Notification"**
   - Title: "Upload Complete"
   - Body: [Contents of URL]

5. **Name it**: "Upload Trading Chart"

---

### Step 3: Test (1 min)

1. Tap your shortcut
2. Select a photo
3. Check notification for success

---

## Visual Flow

```
📱 Shortcuts App
    ↓
[Select Photos] → Choose image
    ↓
[Get Contents of URL]
  • URL: http://YOUR_IP:5000/api/upload-image
  • Method: POST
  • Body: Form
    - image: [Selected Photo]
    - auto_analyze: true
    ↓
[Show Notification] → "✅ Upload Complete"
```

---

## Configuration Options

### Minimal (Upload Only)
```
Form Fields:
- image: [Photo]
```

### With Analysis (Recommended)
```
Form Fields:
- image: [Photo]
- auto_analyze: true
- save_permanently: true
```

### With Custom Name
```
Form Fields:
- image: [Photo]
- auto_analyze: true
- save_permanently: true
- filename: btc_chart
```

---

## Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| Can't connect | Check server is running: `flask run` |
| Wrong network | iPhone must be on same WiFi as Mac |
| 404 error | Use `/api/upload-image` (not `/upload`) |
| Form error | Field name must be `image` (lowercase) |

---

## Add to Share Sheet

1. Tap shortcut settings (⋯)
2. Enable **"Show in Share Sheet"**
3. Select **"Images"**

Now in Photos app:
1. Select chart screenshot
2. Tap Share button
3. Tap "Upload Trading Chart"

---

## Success Response

```json
{
  "success": true,
  "message": "Image uploaded and analysis started",
  "job_id": "analysis_20251031_143025"
}
```

---

## Quick Test

From iPhone Safari, visit:
```
http://YOUR_IP:5000/api/list-uploads
```

If you see JSON, your server is accessible! ✅

---

## Ready to Use!

🎯 **From Photos**: Share → Upload Trading Chart
🏠 **From Home**: Add to Home Screen for quick access  
🗣️ **With Siri**: "Hey Siri, Upload Trading Chart"

---

For detailed setup, see [APPLE_SHORTCUTS_SETUP.md](APPLE_SHORTCUTS_SETUP.md)



