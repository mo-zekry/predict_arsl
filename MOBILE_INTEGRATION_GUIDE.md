# Mobile Integration Guide for ASL Transformer API

## Problem Solved

The previous implementation was treating each API request independently, causing the model to see only one real frame padded with 59 zeros instead of a proper sequence of 60 frames showing continuous motion. This resulted in poor predictions because the transformer model was trained on continuous sequences, not single isolated frames.

## How the Fixed API Works

### Session-Based Frame Buffering
- Each mobile app session gets its own frame buffer
- Frames accumulate across multiple API calls
- The buffer maintains up to 60 frames in a rolling window
- Predictions become reliable after accumulating at least 10 real frames

### API Changes

#### 1. Start a Session (Optional but Recommended)
```http
POST /start_session
```
Returns a `session_id` that you should use for all subsequent requests.

#### 2. Send Frames with Session ID
```http
POST /predict
Headers:
  X-Session-ID: your-session-id-here
Body:
  file: image.jpg
```

#### 3. Monitor Session Status
```http
GET /session/{session_id}/status
```

## Flutter Implementation Example

### 1. Initialize Session
```dart
class ASLPredictionService {
  String? sessionId;
  final String baseUrl = 'http://your-api-url.com';

  Future<void> startSession() async {
    final response = await http.post(Uri.parse('$baseUrl/start_session'));
    if (response.statusCode == 200) {
      final data = json.decode(response.body);
      sessionId = data['session_id'];
    }
  }
}
```

### 2. Send Frames Continuously
```dart
Future<PredictionResult?> predictFromFrame(File imageFile) async {
  if (sessionId == null) {
    await startSession();
  }

  var request = http.MultipartRequest('POST', Uri.parse('$baseUrl/predict'));

  // Add session header
  request.headers['X-Session-ID'] = sessionId!;

  // Add image file
  request.files.add(await http.MultipartFile.fromPath('file', imageFile.path));

  var response = await request.send();
  if (response.statusCode == 200) {
    var responseData = await response.stream.bytesToString();
    var data = json.decode(responseData);

    return PredictionResult(
      predIdx: data['pred_idx'],
      confidence: data['confidence'],
      arabic: data['arabic'],
      english: data['english'],
      framesInBuffer: data['frames_in_buffer'],
      isReliable: data['is_prediction_reliable'],
    );
  }
  return null;
}
```

### 3. Handle Prediction Results
```dart
class PredictionResult {
  final int predIdx;
  final double confidence;
  final String arabic;
  final String english;
  final int framesInBuffer;
  final bool isReliable;

  PredictionResult({
    required this.predIdx,
    required this.confidence,
    required this.arabic,
    required this.english,
    required this.framesInBuffer,
    required this.isReliable,
  });
}

void handlePredictionResult(PredictionResult result) {
  if (!result.isReliable) {
    // Show "Collecting frames..." or progress indicator
    showStatus("Collecting frames... (${result.framesInBuffer}/10 minimum)");
  } else {
    // Show actual prediction
    showPrediction(result.arabic, result.english, result.confidence);
  }
}
```

## Key Improvements

1. **Session Management**: Each user/session maintains its own frame buffer
2. **Reliable Predictions**: Only return predictions when enough real frames are collected
3. **Rolling Buffer**: Maintains the most recent 60 frames, discarding older ones
4. **Memory Management**: Automatic cleanup of expired sessions
5. **Better Feedback**: API responses include buffer status and reliability indicators

## Best Practices for Mobile Apps

1. **Start a session** when the user begins sign recognition
2. **Send frames continuously** (e.g., every 100-200ms) while recording
3. **Check `is_prediction_reliable`** before showing predictions to users
4. **Display buffer status** to help users understand when the system is ready
5. **Clear sessions** when users finish or switch between different sign recording sessions

## Testing the Fix

1. Start the API server
2. Create a session and note the session_id
3. Send 15-20 frames with the same session_id
4. Observe that `is_prediction_reliable` becomes true after 10 frames
5. Compare prediction quality with the previous single-frame approach

This implementation should now provide much more accurate predictions that match the quality you observed on your friend's laptop with the live video feed.
