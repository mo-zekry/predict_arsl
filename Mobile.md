To achieve **real-time prediction** with your FastAPI backend, your Flutter app should continuously capture frames from the camera and send them to the `/predict` endpoint, displaying the prediction as soon as a response is received.

Below is a **minimal Flutter example** using [`camera`](https://pub.dev/packages/camera) and `http` packages. This code captures frames from the camera, sends them every 300ms, and displays the prediction.

````dart
import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';

class RealTimePredictor extends StatefulWidget {
  @override
  _RealTimePredictorState createState() => _RealTimePredictorState();
}

class _RealTimePredictorState extends State<RealTimePredictor> {
  CameraController? _controller;
  late List<CameraDescription> cameras;
  String prediction = '';
  Timer? _timer;
  bool _sending = false;

  @override
  void initState() {
    super.initState();
    initCamera();
  }

  Future<void> initCamera() async {
    cameras = await availableCameras();
    _controller = CameraController(
      cameras[0],
      ResolutionPreset.low,
      enableAudio: false,
    );
    await _controller!.initialize();
    setState(() {});
    startStreaming();
  }

  void startStreaming() {
    _timer = Timer.periodic(Duration(milliseconds: 300), (_) => sendFrame());
  }

  Future<void> sendFrame() async {
    if (_controller == null || !_controller!.value.isInitialized || _sending) return;
    _sending = true;
    try {
      final XFile file = await _controller!.takePicture();
      final bytes = await file.readAsBytes();

      final uri = Uri.parse('http://YOUR_API_HOST:8000/predict'); // Change to your API URL
      final request = http.MultipartRequest('POST', uri)
        ..files.add(http.MultipartFile.fromBytes('file', bytes, filename: 'frame.jpg'));

      final response = await request.send();
      final respStr = await response.stream.bytesToString();

      if (response.statusCode == 200) {
        final data = jsonDecode(respStr);
        setState(() {
          prediction =
              "Prediction: ${data['english']} (${data['arabic']})\nConfidence: ${data['confidence']}";
        });
      } else {
        setState(() {
          prediction = "Error: ${response.statusCode}";
        });
      }
    } catch (e) {
      setState(() {
        prediction = "Error: $e";
      });
    }
    _sending = false;
  }

  @override
  void dispose() {
    _timer?.cancel();
    _controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_controller == null || !_controller!.value.isInitialized) {
      return Center(child: CircularProgressIndicator());
    }
    return Scaffold(
      appBar: AppBar(title: Text('Real-Time ASL Predictor')),
      body: Column(
        children: [
          AspectRatio(
            aspectRatio: _controller!.value.aspectRatio,
            child: CameraPreview(_controller!),
          ),
          SizedBox(height: 16),
          Text(prediction, textAlign: TextAlign.center),
        ],
      ),
    );
  }
}
````

**Instructions:**
- Replace `YOUR_API_HOST` with your FastAPI server's IP/domain.
- Add `camera`, `http`, and `path_provider` to your `pubspec.yaml`.
- Add camera permissions to your `AndroidManifest.xml` and `Info.plist` for Android/iOS.
- Use this widget as your app's home or as a screen.

**This code captures a frame every 300ms, sends it to your API, and displays the prediction.**
You can adjust the interval for performance and accuracy.

Let me know if you need a full `main.dart` or more integration help!