# Simple Flutter Integration Options

## Option 1: Batch Upload (EASIEST) 🎯

**Perfect for: Recording a sign gesture and then getting prediction**

### Flutter Implementation
```dart
class ASLBatchService {
  final String baseUrl = 'http://your-api-url.com';
  List<File> frameBuffer = [];
  
  // Collect frames while user is signing
  void addFrame(File frame) {
    frameBuffer.add(frame);
    
    // Optional: limit buffer size
    if (frameBuffer.length > 60) {
      frameBuffer.removeAt(0);
    }
  }
  
  // Send all frames when user finishes signing
  Future<PredictionResult?> predictFromFrames() async {
    if (frameBuffer.length < 10) {
      throw Exception('Need at least 10 frames');
    }
    
    var request = http.MultipartRequest('POST', Uri.parse('$baseUrl/predict_batch'));
    
    // Add all frames
    for (int i = 0; i < frameBuffer.length; i++) {
      request.files.add(
        await http.MultipartFile.fromPath('files', frameBuffer[i].path)
      );
    }
    
    var response = await request.send();
    if (response.statusCode == 200) {
      var responseData = await response.stream.bytesToString();
      var data = json.decode(responseData);
      
      // Clear buffer after prediction
      frameBuffer.clear();
      
      return PredictionResult.fromJson(data);
    }
    return null;
  }
}
```

### Usage in Your App
```dart
// While recording
void onCameraFrame(CameraImage image) {
  File imageFile = convertToFile(image); // Your conversion logic
  aslService.addFrame(imageFile);
}

// When user stops signing
void onStopSigning() async {
  try {
    var result = await aslService.predictFromFrames();
    if (result != null) {
      showPrediction(result.arabic, result.english);
    }
  } catch (e) {
    showError("Need more frames for prediction");
  }
}
```

---

## Option 2: Auto-Session (SIMPLE) 🔄

**Perfect for: Real-time continuous prediction**

### Flutter Implementation
```dart
class ASLSimpleService {
  final String baseUrl = 'http://your-api-url.com';
  final String userId = 'device_${DateTime.now().millisecondsSinceEpoch}'; // Simple device ID
  
  Future<PredictionResult?> predictFromFrame(File imageFile) async {
    var request = http.MultipartRequest('POST', Uri.parse('$baseUrl/predict_simple'));
    
    // Just add your user ID - no session management needed!
    request.headers['X-User-ID'] = userId;
    request.files.add(await http.MultipartFile.fromPath('file', imageFile.path));
    
    var response = await request.send();
    if (response.statusCode == 200) {
      var responseData = await response.stream.bytesToString();
      return PredictionResult.fromJson(json.decode(responseData));
    }
    return null;
  }
  
  // Optional: Clear buffer when starting new sign
  Future<void> clearBuffer() async {
    // The API will auto-timeout old sessions, but you can manually clear if needed
    await http.delete(Uri.parse('$baseUrl/session/$userId'));
  }
}
```

### Usage in Your App
```dart
// Send frames continuously
Timer.periodic(Duration(milliseconds: 200), (timer) async {
  if (isRecording) {
    File currentFrame = getCurrentFrame(); // Your camera logic
    var result = await aslService.predictFromFrame(currentFrame);
    
    if (result != null && result.isReliable) {
      showPrediction(result.arabic, result.english);
    } else if (result != null) {
      showStatus("Collecting frames... ${result.framesInBuffer}/10");
    }
  }
});
```

---

## Option 3: Client-Side Buffering (ZERO API CHANGES) 📱

**Keep your original API, manage sequences in Flutter**

```dart
class ASLClientBufferService {
  final String baseUrl = 'http://your-api-url.com';
  List<File> frameBuffer = [];
  
  void addFrame(File frame) {
    frameBuffer.add(frame);
    if (frameBuffer.length > 60) frameBuffer.removeAt(0);
    
    // Predict every few frames
    if (frameBuffer.length >= 10 && frameBuffer.length % 5 == 0) {
      _predictFromBuffer();
    }
  }
  
  void _predictFromBuffer() async {
    // Take recent frames and send as batch
    List<File> recentFrames = frameBuffer.sublist(
      math.max(0, frameBuffer.length - 20), 
      frameBuffer.length
    );
    
    // Use your batch endpoint
    // ... same as Option 1
  }
}
```

---

## Comparison

| Approach | Complexity | Real-time | Best For |
|----------|------------|-----------|----------|
| **Batch Upload** | ⭐ Easiest | ❌ No | Record → Predict workflow |
| **Auto-Session** | ⭐⭐ Simple | ✅ Yes | Continuous real-time prediction |
| **Client Buffer** | ⭐⭐⭐ Medium | ✅ Yes | Maximum control |

## Recommendation

**Start with Option 1 (Batch Upload)** - it's the simplest and will work perfectly for most ASL apps where users:
1. Start recording a sign
2. Perform the sign gesture  
3. Stop recording
4. Get the prediction

If you need real-time predictions, use **Option 2 (Auto-Session)** with just a device ID.
