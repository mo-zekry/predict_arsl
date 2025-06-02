# WebSocket Integration Guide for ASL Prediction API

## Overview

WebSockets provide the **BEST** approach for real-time ASL prediction because they offer:

- ⚡ **Lower latency** (no HTTP overhead)
- 🔄 **Real-time streaming** of camera frames
- 📱 **Better mobile app integration**
- 🎯 **Automatic session management**
- 💪 **Higher throughput** for continuous frames

## Quick Start

### 1. Start the Server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 2. Test with Demo Page
Visit: `http://localhost:8000/ws_demo`

### 3. Connect to WebSocket
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/your_client_id');
```

## WebSocket Protocol

### Connection
- **Endpoint:** `/ws/{client_id}`
- **Client ID:** Any unique identifier (device ID, user ID, etc.)

### Message Format

#### Send Frame (Client → Server)
```json
{
  "type": "frame",
  "data": "base64_encoded_image_data"
}
```

#### Receive Prediction (Server → Client)
```json
{
  "type": "prediction",
  "prediction": {
    "pred_idx": 5,
    "confidence": 0.89,
    "arabic": "السلام عليكم",
    "english": "Hello",
    "session_id": "client123",
    "frames_in_buffer": 15,
    "is_prediction_reliable": true
  },
  "timestamp": 1735819200.123
}
```

#### Other Messages
```json
// Ping (heartbeat)
{"type": "ping"}
// Response: {"type": "pong", "timestamp": 1735819200.123}

// Get status
{"type": "get_status"}
// Response: {"type": "status", "status": {...}, "timestamp": 1735819200.123}

// Error
{"type": "error", "error": "Error message", "timestamp": 1735819200.123}
```

## Implementation Examples

### JavaScript/Web
```javascript
class ASLWebSocketClient {
    constructor(serverUrl, clientId) {
        this.serverUrl = serverUrl;
        this.clientId = clientId;
        this.ws = null;
    }

    connect() {
        this.ws = new WebSocket(`${this.serverUrl}/ws/${this.clientId}`);

        this.ws.onopen = () => {
            console.log('Connected to ASL prediction server');
        };

        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleMessage(data);
        };

        this.ws.onclose = () => {
            console.log('Disconnected from server');
        };
    }

    sendFrame(imageBlob) {
        const reader = new FileReader();
        reader.onload = () => {
            const base64Data = reader.result.split(',')[1];
            this.ws.send(JSON.stringify({
                type: 'frame',
                data: base64Data
            }));
        };
        reader.readAsDataURL(imageBlob);
    }

    handleMessage(data) {
        switch(data.type) {
            case 'prediction':
                this.onPrediction(data.prediction);
                break;
            case 'error':
                this.onError(data.error);
                break;
            case 'pong':
                console.log('Server alive');
                break;
        }
    }

    onPrediction(prediction) {
        console.log(`Prediction: ${prediction.arabic} (${prediction.english})`);
        console.log(`Confidence: ${(prediction.confidence * 100).toFixed(1)}%`);
    }

    onError(error) {
        console.error('Prediction error:', error);
    }
}

// Usage
const client = new ASLWebSocketClient('ws://localhost:8000', 'user123');
client.connect();
```

### Flutter/Dart
```dart
import 'dart:convert';
import 'dart:typed_data';
import 'package:web_socket_channel/web_socket_channel.dart';

class ASLWebSocketClient {
  late WebSocketChannel _channel;
  final String serverUrl;
  final String clientId;

  ASLWebSocketClient(this.serverUrl, this.clientId);

  void connect() {
    _channel = WebSocketChannel.connect(
      Uri.parse('$serverUrl/ws/$clientId'),
    );

    _channel.stream.listen(
      (data) {
        final message = jsonDecode(data);
        _handleMessage(message);
      },
      onError: (error) {
        print('WebSocket error: $error');
      },
      onDone: () {
        print('WebSocket connection closed');
      },
    );
  }

  void sendFrame(Uint8List imageBytes) {
    final base64Image = base64Encode(imageBytes);
    final message = {
      'type': 'frame',
      'data': base64Image,
    };
    _channel.sink.add(jsonEncode(message));
  }

  void _handleMessage(Map<String, dynamic> message) {
    switch (message['type']) {
      case 'prediction':
        _onPrediction(message['prediction']);
        break;
      case 'error':
        _onError(message['error']);
        break;
    }
  }

  void _onPrediction(Map<String, dynamic> prediction) {
    print('Arabic: ${prediction['arabic']}');
    print('English: ${prediction['english']}');
    print('Confidence: ${(prediction['confidence'] * 100).toStringAsFixed(1)}%');
  }

  void _onError(String error) {
    print('Error: $error');
  }

  void disconnect() {
    _channel.sink.close();
  }
}
```

### Python Client
```python
import asyncio
import websockets
import json
import base64

class ASLWebSocketClient:
    def __init__(self, server_url, client_id):
        self.server_url = server_url
        self.client_id = client_id
        self.websocket = None

    async def connect(self):
        uri = f"{self.server_url}/ws/{self.client_id}"
        self.websocket = await websockets.connect(uri)
        print(f"Connected to {uri}")

        # Listen for messages
        async for message in self.websocket:
            data = json.loads(message)
            await self.handle_message(data)

    async def send_frame(self, image_bytes):
        base64_data = base64.b64encode(image_bytes).decode('utf-8')
        message = {
            "type": "frame",
            "data": base64_data
        }
        await self.websocket.send(json.dumps(message))

    async def handle_message(self, data):
        if data['type'] == 'prediction':
            prediction = data['prediction']
            print(f"Arabic: {prediction['arabic']}")
            print(f"English: {prediction['english']}")
            print(f"Confidence: {prediction['confidence']:.2%}")
        elif data['type'] == 'error':
            print(f"Error: {data['error']}")

# Usage
async def main():
    client = ASLWebSocketClient("ws://localhost:8000", "python_client")
    await client.connect()

# asyncio.run(main())
```

## Mobile App Integration

### Camera Frame Streaming
```dart
// In your Flutter camera app
import 'package:camera/camera.dart';

class CameraStreamingPage extends StatefulWidget {
  @override
  _CameraStreamingPageState createState() => _CameraStreamingPageState();
}

class _CameraStreamingPageState extends State<CameraStreamingPage> {
  CameraController? _cameraController;
  ASLWebSocketClient? _wsClient;
  Timer? _frameTimer;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
    _initializeWebSocket();
  }

  void _initializeCamera() async {
    final cameras = await availableCameras();
    _cameraController = CameraController(
      cameras.first,
      ResolutionPreset.medium,
    );
    await _cameraController!.initialize();
    setState(() {});

    // Start streaming frames at 10 FPS
    _frameTimer = Timer.periodic(Duration(milliseconds: 100), (timer) {
      _captureAndSendFrame();
    });
  }

  void _initializeWebSocket() {
    _wsClient = ASLWebSocketClient(
      'ws://your-server.com:8000',
      'flutter_${DateTime.now().millisecondsSinceEpoch}',
    );
    _wsClient!.connect();
  }

  void _captureAndSendFrame() async {
    if (_cameraController != null && _wsClient != null) {
      try {
        final image = await _cameraController!.takePicture();
        final imageBytes = await image.readAsBytes();
        _wsClient!.sendFrame(imageBytes);
      } catch (e) {
        print('Error capturing frame: $e');
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      return Center(child: CircularProgressIndicator());
    }

    return Scaffold(
      body: CameraPreview(_cameraController!),
    );
  }

  @override
  void dispose() {
    _frameTimer?.cancel();
    _cameraController?.dispose();
    _wsClient?.disconnect();
    super.dispose();
  }
}
```

## Performance Tips

### 1. Frame Rate Optimization
- **Recommended:** 5-15 FPS for good predictions
- **Mobile:** 8-10 FPS is optimal for battery life
- **High-end:** Up to 15 FPS for smoother experience

### 2. Image Quality
- **Resolution:** 480x640 or 640x480 is sufficient
- **Format:** JPEG with 80% quality for balance
- **Processing:** Resize before base64 encoding

### 3. Connection Management
```javascript
// Reconnection logic
class RobustASLClient extends ASLWebSocketClient {
    constructor(serverUrl, clientId) {
        super(serverUrl, clientId);
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
    }

    connect() {
        super.connect();

        this.ws.onclose = () => {
            console.log('Connection lost, attempting to reconnect...');
            this.attemptReconnect();
        };
    }

    attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            setTimeout(() => {
                this.connect();
            }, 1000 * this.reconnectAttempts); // Exponential backoff
        }
    }
}
```

## Advantages Over HTTP

| Feature | HTTP API | WebSocket API |
|---------|----------|---------------|
| Latency | High (100-300ms) | Low (5-50ms) |
| Overhead | HTTP headers per request | Minimal after handshake |
| Session Management | Manual with IDs | Automatic with connection |
| Real-time | Polling required | Native streaming |
| Mobile Battery | Higher consumption | Lower consumption |
| Scalability | Limited by connections | Better with persistent connections |

## Security Considerations

### 1. Authentication
```javascript
// Add authentication in connection headers
const ws = new WebSocket('ws://localhost:8000/ws/client123', [], {
    headers: {
        'Authorization': 'Bearer your-jwt-token'
    }
});
```

### 2. Rate Limiting
- Server automatically limits frames per second
- Client should respect server responses
- Implement client-side throttling

### 3. Data Validation
- All frames are validated server-side
- Invalid frames are skipped, not rejected
- Error messages provide helpful feedback

## Troubleshooting

### Common Issues

1. **Connection Refused**
   - Check server is running: `uvicorn main:app --host 0.0.0.0 --port 8000`
   - Verify firewall settings
   - Test with demo page first

2. **No Predictions**
   - Need minimum 10 frames for reliable predictions
   - Check image format (JPEG/PNG)
   - Verify base64 encoding

3. **High Latency**
   - Reduce image resolution
   - Lower frame rate
   - Check network connection

4. **Memory Issues**
   - Server automatically cleans up old sessions
   - Client should reconnect periodically
   - Monitor frame buffer size

### Testing Commands
```bash
# Test server health
curl http://localhost:8000/health

# Check WebSocket with wscat
npm install -g wscat
wscat -c ws://localhost:8000/ws/test123
```

## Conclusion

WebSockets are the **recommended approach** for real-time ASL prediction because they provide:

- 🚀 **3-10x lower latency** than HTTP
- 📱 **Better mobile app integration**
- 🔋 **Lower battery consumption**
- 🎯 **Automatic session management**
- 📊 **Higher prediction accuracy** with smoother frame sequences

Start with the demo page, then integrate into your mobile app using the examples above!
