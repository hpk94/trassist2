# Apple Shortcuts Setup Guide

## Overview
This guide will help you create an Apple Shortcuts shortcut to upload trading chart images to your Trading Assistant API endpoint from your iPhone, iPad, or Mac.

## Prerequisites
1. Your Flask server must be running and accessible from your device
2. You need the server's IP address or domain name
3. If running locally, your device must be on the same network

## Server Setup

### Find Your Server Address

#### On Mac (where your server is running):
```bash
# Get your local IP address
ifconfig | grep "inet " | grep -v 127.0.0.1
```

Or check System Settings > Network > Your Connection > Details

Your server URL will be: `http://YOUR_IP:5000/api/upload-image`

For example: `http://192.168.1.100:5000/api/upload-image`

### Make Server Accessible

If your server is only accessible locally, you have two options:

**Option 1: Use ngrok (Recommended for external access)**
```bash
# Install ngrok from https://ngrok.com
ngrok http 5000
```

This will give you a public URL like: `https://abc123.ngrok.io/api/upload-image`

**Option 2: Local Network Only**
Use your Mac's local IP address (make sure your iPhone/iPad is on the same WiFi network)

---

## Creating the Apple Shortcut

### Step 1: Open Shortcuts App
1. Open the **Shortcuts** app on your iPhone/iPad
2. Tap the **"+"** button in the top right corner
3. Tap **"Add Action"**

### Step 2: Select Photos
1. Search for **"Select Photos"** or **"Take Photo"**
   - **"Select Photos"**: Choose existing images from your photo library
   - **"Take Photo"**: Take a new photo with the camera
2. Tap to add it to your shortcut

For this guide, we'll use **"Select Photos"**:
- Set **"Select Multiple"** to OFF (unless you want to upload multiple images)
- Enable **"Select Photos"** to allow selection at runtime

### Step 3: Add the Upload Action
1. Tap the **"+"** button below your first action
2. Search for **"Get Contents of URL"**
3. Tap to add it
4. Configure it as follows:

#### Basic Configuration:
- **Method**: `POST`
- **URL**: `http://YOUR_IP:5000/api/upload-image` (replace with your actual server URL)

#### Headers (Add these by tapping "Headers" and "Add new field"):
- No specific headers required (Content-Type is auto-set for multipart/form-data)

#### Request Body:
1. Tap **"Request Body"** dropdown
2. Select **"Form"**
3. Tap **"Add new field"**
4. Add the following fields:

**Required Field:**
- **Key**: `image`
- **Type**: `File`
- **Value**: Tap the field and select **"Select Photos"** from the variables

**Optional Fields** (add these if you want auto-analysis and permanent storage):

- **Key**: `auto_analyze`
- **Type**: `Text`
- **Value**: `true` or `false`

- **Key**: `save_permanently`
- **Type**: `Text`
- **Value**: `true` or `false`

- **Key**: `filename`
- **Type**: `Text`
- **Value**: `my_chart` (any custom name)

### Step 4: Show Result (Optional)
1. Tap the **"+"** button
2. Search for **"Show Result"**
3. Tap to add it
4. Tap the field and select **"Contents of URL"** from the variables

This will show you the API response after upload.

### Step 5: Show Notification (Alternative)
Instead of "Show Result", you can use "Show Notification":
1. Search for **"Show Notification"**
2. Set the title to: `Upload Complete`
3. Set the body to: `Contents of URL` (from variables)

### Step 6: Name Your Shortcut
1. Tap the shortcut name at the top (e.g., "New Shortcut")
2. Rename it to something like: **"Upload Trading Chart"**

### Step 7: Add to Share Sheet (Optional)
1. Tap the settings icon (three dots) in the top right
2. Scroll down to **"Share Sheet Types"**
3. Enable **"Images"** and **"Photos"**
4. This allows you to run the shortcut from the share menu in Photos

### Step 8: Add to Home Screen or Widget (Optional)
1. Tap the settings icon
2. Scroll down to **"Add to Home Screen"**
3. Customize the icon and name
4. Tap **"Add"**

---

## Example Shortcut Configuration

### Minimal Configuration (Just Upload)
```
[Select Photos]
  ↓
[Get Contents of URL]
  - URL: http://192.168.1.100:5000/api/upload-image
  - Method: POST
  - Request Body: Form
    - image: [Select Photos]
  ↓
[Show Notification]
  - Title: "Upload Complete"
  - Body: [Contents of URL]
```

### Full Configuration (With Analysis)
```
[Select Photos]
  ↓
[Get Contents of URL]
  - URL: http://192.168.1.100:5000/api/upload-image
  - Method: POST
  - Request Body: Form
    - image: [Select Photos]
    - auto_analyze: true
    - save_permanently: true
    - filename: chart
  ↓
[Show Notification]
  - Title: "Analysis Started"
  - Body: [Contents of URL]
```

### Advanced: Take Photo and Upload
```
[Take Photo]
  ↓
[Get Contents of URL]
  - URL: http://192.168.1.100:5000/api/upload-image
  - Method: POST
  - Request Body: Form
    - image: [Taken Photo]
    - auto_analyze: true
    - save_permanently: true
  ↓
[Get Dictionary from Input]
  ↓
[Get Value for Key: "success"]
  ↓
[If "success" equals "true"]
  - Show Notification: "✅ Upload Successful"
[Otherwise]
  - Show Notification: "❌ Upload Failed"
[End If]
```

---

## Testing Your Shortcut

### Test from Shortcuts App:
1. Open the Shortcuts app
2. Tap your new shortcut
3. Select a photo when prompted
4. Check the notification for success/error

### Test from Share Sheet:
1. Open Photos app
2. Select a screenshot of a trading chart
3. Tap the share button
4. Scroll down and tap **"Upload Trading Chart"** (or your shortcut name)
5. Check the notification

### Test from Home Screen:
1. Tap the shortcut icon on your home screen
2. Follow the prompts

---

## Troubleshooting

### "Cannot Connect to Server"
- **Check server is running**: On your Mac, run `curl http://localhost:5000/api/upload-image` 
- **Check IP address**: Make sure you're using the correct IP
- **Check network**: iPhone/iPad must be on same WiFi network (for local network)
- **Check firewall**: Mac firewall might be blocking connections

### "Invalid Response"
- Check the server logs in your terminal
- Verify the URL is correct
- Make sure the API endpoint is `/api/upload-image` (not `/upload`)

### "Request Failed"
- Check that the form field name is exactly `image` (lowercase)
- Verify the request body type is set to **"Form"**
- Check that the image variable is properly connected

### Testing Server Accessibility
From your iPhone/iPad, open Safari and go to:
```
http://YOUR_IP:5000/api/list-uploads
```

If you see a JSON response, your server is accessible.

---

## Advanced Features

### Add Input Parameters
You can make the shortcut ask for custom filename:

1. Add **"Ask for Input"** action before the upload
   - Prompt: "Enter chart name"
   - Input Type: Text
2. Use **"Provided Input"** as the value for the `filename` field

### Handle Response
Parse the JSON response to show specific information:

1. After **"Get Contents of URL"**, add **"Get Dictionary from Input"**
2. Add **"Get Dictionary Value"** with key `job_id` or `metadata`
3. Use in notification

### Upload Multiple Photos
1. Change **"Select Photos"** to allow multiple
2. Add **"Repeat with Each"** action
3. Place upload actions inside the repeat loop

### Conditional Logic
Add conditions based on response:
```
[Get Dictionary Value: "success"]
  ↓
[If "success" is true]
  - Show Notification: "✅ Success"
[Otherwise]
  - Show Notification: "❌ Failed"
  - Get Dictionary Value: "error"
  - Show Alert with error message
[End If]
```

---

## Security Considerations

1. **Local Network Only**: If using local IP, only works on your home network
2. **HTTPS Recommended**: For external access, use HTTPS (ngrok provides this)
3. **Authentication**: Consider adding API key authentication if exposing publicly
4. **VPN**: For secure remote access, set up VPN to your home network

---

## Quick Reference

### Endpoint Details
- **URL**: `http://YOUR_IP:5000/api/upload-image`
- **Method**: POST
- **Content-Type**: multipart/form-data
- **Required Field**: `image` (file)
- **Optional Fields**: 
  - `auto_analyze` (boolean)
  - `save_permanently` (boolean)
  - `filename` (string)

### Response Format
```json
{
  "success": true,
  "message": "Image uploaded and analysis started",
  "job_id": "analysis_20251031_143025",
  "metadata": {
    "filename": "trading_chart_20251031_143025.png",
    "size_kb": 239.92,
    "timestamp": "20251031_143025",
    "saved_permanently": true
  }
}
```

---

## Example Use Cases

### Use Case 1: Quick Screenshot Upload
1. Take screenshot of trading chart on TradingView
2. Open Photos, find screenshot
3. Share > Upload Trading Chart
4. Receive notification when complete

### Use Case 2: Auto-Analysis from Chart
1. Tap home screen shortcut
2. Select chart image
3. Shortcut uploads with `auto_analyze=true`
4. Receive notification with analysis job ID
5. Check results in your Flask app

### Use Case 3: Batch Upload
1. Configure shortcut for multiple photos
2. Select 5-10 chart screenshots
3. Shortcut uploads each one sequentially
4. Receive summary notification

---

## Next Steps

After setting up the shortcut:

1. **Test with a sample image** to verify it works
2. **Add to home screen** for quick access
3. **Share with Share Sheet** for Photos integration
4. **Create variants** for different use cases (with/without analysis)
5. **Set up Siri** by naming the shortcut and using voice commands

---

## Support

If you encounter issues:

1. Check your Flask server logs
2. Verify the URL is accessible from your device
3. Test with curl first (from terminal)
4. Check Shortcuts app for error messages
5. Review the API_DOCUMENTATION.md for endpoint details

---

## Alternative: Using Siri

Once your shortcut is created, you can run it with Siri:

1. Say: **"Hey Siri, Upload Trading Chart"**
2. Siri will prompt you to select a photo
3. Select your chart
4. Siri will upload and show the result

---

## Bonus: QR Code for Easy Setup

You can create a QR code that opens your shortcut setup:

1. Go to https://www.icloud.com/shortcuts
2. Create the shortcut there
3. Get the share link
4. Convert to QR code
5. Scan on other devices to install

---

## Summary

You now have a complete guide to:
✅ Create an Apple Shortcut for uploading images
✅ Configure it to work with your trading assistant API
✅ Add advanced features like auto-analysis
✅ Share and use it across your iOS devices
✅ Troubleshoot common issues

Happy trading! 📈


