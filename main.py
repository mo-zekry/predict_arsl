"""
ASL Transformer ONNX API for Mobile
==================================
A FastAPI-based backend for real-time Arabic Sign Language (ASL) prediction using a transformer model in ONNX format. Designed for integration with mobile apps (e.g., Flutter) for live camera-based sign recognition.

Author: Graduation Project Team
Date: 2025-06-01
"""

import io
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Header, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import onnxruntime as ort
import json
import os
import uuid
import time
import base64
from typing import List, Dict, Optional
from collections import defaultdict

# Paths (relative to predict_arsl)
ONNX_MODEL_PATH = os.path.join(os.path.dirname(__file__), "asl_transformer.onnx")
IDX2AR_PATH = os.path.join(os.path.dirname(__file__), "idx2ar.json")
IDX2ENG_PATH = os.path.join(os.path.dirname(__file__), "idx2eng.json")

# Load label mappings
try:
    with open(IDX2AR_PATH, "r", encoding="utf-8") as f:
        idx2ar = {int(k): v for k, v in json.load(f).items()}
    with open(IDX2ENG_PATH, "r", encoding="utf-8") as f:
        idx2eng = {int(k): v for k, v in json.load(f).items()}
except Exception as e:
    raise RuntimeError(f"Failed to load label files: {e}")

# ONNX session
try:
    session = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
except Exception as e:
    raise RuntimeError(f"Failed to load ONNX model: {e}")

SEQUENCE_LENGTH = 60
FEATURE_DIM = 126
SESSION_TIMEOUT = 300  # 5 minutes timeout for inactive sessions
MIN_FRAMES_FOR_PREDICTION = 10  # Minimum real frames before making predictions

# Session-based frame buffers with timestamps
class SessionData:
    def __init__(self):
        self.frame_buffer: List[np.ndarray] = []
        self.last_updated: float = time.time()

    def update_timestamp(self):
        self.last_updated = time.time()

    def is_expired(self) -> bool:
        return time.time() - self.last_updated > SESSION_TIMEOUT

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_sessions: Dict[str, SessionData] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.connection_sessions[client_id] = SessionData()
        print(f"Client {client_id} connected via WebSocket")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.connection_sessions:
            del self.connection_sessions[client_id]
        print(f"Client {client_id} disconnected")

    async def send_personal_message(self, message: dict, client_id: str):
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            await websocket.send_json(message)

    def get_session(self, client_id: str) -> Optional[SessionData]:
        return self.connection_sessions.get(client_id)

# Global session storage and WebSocket manager
session_buffers: Dict[str, SessionData] = {}
manager = ConnectionManager()

def cleanup_expired_sessions():
    """Remove expired sessions to prevent memory leaks"""
    expired_sessions = [session_id for session_id, data in session_buffers.items() if data.is_expired()]
    for session_id in expired_sessions:
        del session_buffers[session_id]

app = FastAPI(
    title="ASL Transformer Prediction API",
    description="API for real-time Arabic Sign Language prediction using ONNX transformer model. Designed for mobile camera integration.",
    version="1.0.0",
    contact={
        "name": "Graduation Project Team",
        "email": "your-email@example.com"
    },
)

# CORS: Allow all origins for demo; restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionResponse(BaseModel):
    pred_idx: int
    confidence: float
    arabic: str
    english: str
    session_id: str
    frames_in_buffer: int
    is_prediction_reliable: bool  # True when we have enough real frames

class WebSocketMessage(BaseModel):
    type: str  # "frame", "ping", "get_status"
    data: Optional[str] = None  # base64 encoded image for "frame" type
    client_id: Optional[str] = None
    timestamp: Optional[float] = None

class WebSocketResponse(BaseModel):
    type: str  # "prediction", "status", "error", "pong"
    prediction: Optional[PredictionResponse] = None
    status: Optional[dict] = None
    error: Optional[str] = None
    timestamp: float

@app.post("/predict", response_model=PredictionResponse, summary="Predict ASL sign from image frame", tags=["Prediction"])
async def predict_api(
    file: UploadFile = File(...),
    session_id: Optional[str] = Header(None, alias="X-Session-ID")
):
    """
    Accepts a single image frame (as JPEG/PNG) and returns the predicted ASL sign.
    The API maintains a rolling buffer of frames per session to form a sequence for the transformer model.

    IMPORTANT: Send frames continuously from your mobile app with the same session_id
    to build up a proper sequence. The model needs multiple frames to see motion patterns.

    Args:
        file: Uploaded image file (frame from mobile camera)
        session_id: Optional session identifier in X-Session-ID header. If not provided, a new session is created.
    Returns:
        PredictionResponse: predicted class index, confidence, Arabic and English labels, plus session info
    """
    # Clean up expired sessions periodically
    cleanup_expired_sessions()

    # Handle session management
    if not session_id:
        session_id = str(uuid.uuid4())

    if session_id not in session_buffers:
        session_buffers[session_id] = SessionData()

    session_data = session_buffers[session_id]
    session_data.update_timestamp()

    try:
        image_bytes = await file.read()
        npimg = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Invalid image file.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {e}")

    # Extract keypoints and add to session buffer
    keypoints = extract_keypoints_api(frame)
    session_data.frame_buffer.append(keypoints)

    # Maintain rolling buffer of SEQUENCE_LENGTH frames
    if len(session_data.frame_buffer) > SEQUENCE_LENGTH:
        session_data.frame_buffer.pop(0)

    # Determine if we can make a reliable prediction
    frames_count = len(session_data.frame_buffer)
    is_reliable = frames_count >= MIN_FRAMES_FOR_PREDICTION

    pred_idx, confidence = -1, 0.0

    if is_reliable:
        # We have enough real frames - create sequence for prediction
        sequence = preprocess_sequence_api(session_data.frame_buffer)
        pred_idx, confidence = predict_api_onnx(sequence)

    arabic = idx2ar.get(pred_idx, "Unknown" if pred_idx != -1 else "Collecting frames...")
    english = idx2eng.get(pred_idx, "Unknown" if pred_idx != -1 else "Collecting frames...")

    return PredictionResponse(
        pred_idx=pred_idx,
        confidence=confidence,
        arabic=arabic,
        english=english,
        session_id=session_id,
        frames_in_buffer=frames_count,
        is_prediction_reliable=is_reliable
    )

def extract_keypoints_api(frame: np.ndarray) -> np.ndarray:
    """
    Extracts keypoints/features from the input frame.
    For demo: resizes and flattens the image to a fixed feature vector.
    Replace with real keypoint extraction for production.
    """
    resized = cv2.resize(frame, (21, 2))
    keypoints = resized.flatten()[:FEATURE_DIM].astype(np.float32)
    if keypoints.shape[0] < FEATURE_DIM:
        keypoints = np.pad(keypoints, (0, FEATURE_DIM - keypoints.shape[0]), 'constant')
    return keypoints

def preprocess_sequence_api(seq: List[np.ndarray]) -> np.ndarray:
    """
    Pads or trims the sequence to SEQUENCE_LENGTH for model input.
    For sequences shorter than SEQUENCE_LENGTH, we pad with zeros at the beginning
    to keep the most recent frames at the end (which is more important for current prediction).
    """
    if len(seq) < SEQUENCE_LENGTH:
        # Pad at the beginning to keep recent frames at the end
        padding = [np.zeros(FEATURE_DIM, dtype=np.float32)] * (SEQUENCE_LENGTH - len(seq))
        seq = padding + seq
    else:
        # Take the most recent SEQUENCE_LENGTH frames
        seq = seq[-SEQUENCE_LENGTH:]
    return np.array(seq, dtype=np.float32)

def predict_api_onnx(sequence: np.ndarray) -> (int, float):
    """
    Runs ONNX model inference and returns predicted index and confidence.
    """
    input_array = np.expand_dims(sequence, axis=0).astype(np.float32)
    ort_inputs = {input_name: input_array}
    ort_outs = session.run([output_name], ort_inputs)
    logits = ort_outs[0]
    probabilities = softmax_api(logits[0])
    pred_idx = int(np.argmax(probabilities))
    confidence = float(probabilities[pred_idx])
    return pred_idx, confidence

def softmax_api(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax for model output.
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

@app.post("/start_session", tags=["Session"])
def start_new_session():
    """
    Creates a new session for frame sequence tracking.
    Returns a session_id that should be sent in X-Session-ID header with subsequent predict requests.
    """
    session_id = str(uuid.uuid4())
    session_buffers[session_id] = SessionData()
    return {
        "session_id": session_id,
        "message": "New session created. Include this session_id in X-Session-ID header for predict requests.",
        "min_frames_needed": MIN_FRAMES_FOR_PREDICTION,
        "sequence_length": SEQUENCE_LENGTH
    }

@app.get("/session/{session_id}/status", tags=["Session"])
def get_session_status(session_id: str):
    """
    Get the current status of a session including frame count and buffer state.
    """
    if session_id not in session_buffers:
        raise HTTPException(status_code=404, detail="Session not found")

    session_data = session_buffers[session_id]
    frames_count = len(session_data.frame_buffer)

    return {
        "session_id": session_id,
        "frames_in_buffer": frames_count,
        "is_reliable": frames_count >= MIN_FRAMES_FOR_PREDICTION,
        "frames_needed_for_reliable_prediction": max(0, MIN_FRAMES_FOR_PREDICTION - frames_count),
        "buffer_full": frames_count >= SEQUENCE_LENGTH,
        "last_updated": session_data.last_updated
    }

@app.delete("/session/{session_id}", tags=["Session"])
def clear_session(session_id: str):
    """
    Clear/reset a session's frame buffer or delete the session entirely.
    """
    if session_id not in session_buffers:
        raise HTTPException(status_code=404, detail="Session not found")

    del session_buffers[session_id]
    return {"message": f"Session {session_id} cleared successfully"}

@app.get("/", tags=["Health"])
def root():
    """
    Health check endpoint. Returns API status and documentation link.
    """
    return {
        "message": "ASL Transformer ONNX API for Mobile is running.",
        "docs_url": "/docs",
        "websocket_endpoint": "/ws/{client_id}",
        "predict_endpoint": "/predict",
        "session_management": {
            "start_session": "/start_session",
            "session_status": "/session/{session_id}/status",
            "clear_session": "/session/{session_id}"
        },
        "recommended": "Use WebSocket endpoint /ws/{client_id} for best performance!",
        "websocket_protocol": {
            "connect": "ws://your-server/ws/your_client_id",
            "send_frame": '{"type": "frame", "data": "base64_encoded_image"}',
            "send_ping": '{"type": "ping"}',
            "get_status": '{"type": "get_status"}'
        }
    }

@app.get("/ws_demo", tags=["Demo"])
def websocket_demo():
    """
    Returns HTML demo page for testing WebSocket functionality.
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ASL WebSocket Demo</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { max-width: 800px; margin: 0 auto; }
            .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
            .connected { background-color: #d4edda; }
            .disconnected { background-color: #f8d7da; }
            .prediction { background-color: #e2e3e5; padding: 15px; margin: 10px 0; border-radius: 5px; }
            button { padding: 10px 20px; margin: 5px; }
            #fileInput { margin: 10px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>ASL WebSocket Demo</h1>
            <div id="status" class="status disconnected">Disconnected</div>

            <button onclick="connect()">Connect</button>
            <button onclick="disconnect()">Disconnect</button>
            <button onclick="ping()">Ping</button>
            <button onclick="getStatus()">Get Status</button>
            <br>

            <input type="file" id="fileInput" accept="image/*">
            <button onclick="sendImage()">Send Image</button>

            <div id="predictions"></div>

            <h3>Raw Messages:</h3>
            <div id="messages" style="height: 200px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px;"></div>
        </div>

        <script>
            let ws = null;
            let clientId = 'demo_' + Math.random().toString(36).substr(2, 9);

            function connect() {
                ws = new WebSocket(`ws://localhost:8000/ws/${clientId}`);

                ws.onopen = function(event) {
                    document.getElementById('status').textContent = `Connected (Client ID: ${clientId})`;
                    document.getElementById('status').className = 'status connected';
                    addMessage('Connected to WebSocket');
                };

                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    addMessage(`Received: ${JSON.stringify(data, null, 2)}`);

                    if (data.type === 'prediction' && data.prediction) {
                        showPrediction(data.prediction);
                    }
                };

                ws.onclose = function(event) {
                    document.getElementById('status').textContent = 'Disconnected';
                    document.getElementById('status').className = 'status disconnected';
                    addMessage('Disconnected from WebSocket');
                };

                ws.onerror = function(error) {
                    addMessage(`Error: ${error}`);
                };
            }

            function disconnect() {
                if (ws) {
                    ws.close();
                }
            }

            function ping() {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({type: 'ping'}));
                    addMessage('Sent ping');
                }
            }

            function getStatus() {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({type: 'get_status'}));
                    addMessage('Requested status');
                }
            }

            function sendImage() {
                const fileInput = document.getElementById('fileInput');
                const file = fileInput.files[0];

                if (!file) {
                    alert('Please select an image file');
                    return;
                }

                if (!(ws && ws.readyState === WebSocket.OPEN)) {
                    alert('Please connect to WebSocket first');
                    return;
                }

                const reader = new FileReader();
                reader.onload = function(e) {
                    const base64Data = e.target.result.split(',')[1]; // Remove data:image/...;base64,
                    ws.send(JSON.stringify({
                        type: 'frame',
                        data: base64Data
                    }));
                    addMessage('Sent image frame');
                };
                reader.readAsDataURL(file);
            }

            function addMessage(message) {
                const messagesDiv = document.getElementById('messages');
                const timestamp = new Date().toLocaleTimeString();
                messagesDiv.innerHTML += `<div>[${timestamp}] ${message}</div>`;
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }

            function showPrediction(prediction) {
                const predictionsDiv = document.getElementById('predictions');
                const predictionHtml = `
                    <div class="prediction">
                        <h4>Prediction Result</h4>
                        <p><strong>Arabic:</strong> ${prediction.arabic}</p>
                        <p><strong>English:</strong> ${prediction.english}</p>
                        <p><strong>Confidence:</strong> ${(prediction.confidence * 100).toFixed(2)}%</p>
                        <p><strong>Frames in buffer:</strong> ${prediction.frames_in_buffer}</p>
                        <p><strong>Reliable:</strong> ${prediction.is_prediction_reliable ? 'Yes' : 'No'}</p>
                    </div>
                `;
                predictionsDiv.innerHTML = predictionHtml + predictionsDiv.innerHTML;
            }
        </script>
    </body>
    </html>
    """
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=html_content)

@app.get("/health", tags=["Health"])
def health_check():
    """
    Returns API health status.
    """
    return {"status": "ok"}

@app.post("/predict_batch", response_model=PredictionResponse, summary="Predict ASL sign from multiple image frames", tags=["Prediction"])
async def predict_batch_api(files: List[UploadFile] = File(...)):
    """
    SIMPLEST APPROACH: Send multiple frames at once (10-60 frames).
    No session management needed - just collect frames in your Flutter app and send them all together.

    Args:
        files: List of image files (10-60 frames from mobile camera)
    Returns:
        PredictionResponse: predicted class index, confidence, Arabic and English labels
    """
    if len(files) < MIN_FRAMES_FOR_PREDICTION:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {MIN_FRAMES_FOR_PREDICTION} frames, got {len(files)}"
        )

    if len(files) > SEQUENCE_LENGTH:
        # Take the most recent frames if too many
        files = files[-SEQUENCE_LENGTH:]

    frame_sequence = []

    for file in files:
        try:
            image_bytes = await file.read()
            npimg = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            if frame is None:
                continue  # Skip invalid frames

            keypoints = extract_keypoints_api(frame)
            frame_sequence.append(keypoints)
        except Exception as e:
            continue  # Skip problematic frames

    if len(frame_sequence) < MIN_FRAMES_FOR_PREDICTION:
        raise HTTPException(status_code=400, detail="Not enough valid frames after processing")

    # Prepare sequence for model
    sequence = preprocess_sequence_api(frame_sequence)
    pred_idx, confidence = predict_api_onnx(sequence)

    arabic = idx2ar.get(pred_idx, "Unknown")
    english = idx2eng.get(pred_idx, "Unknown")

    return PredictionResponse(
        pred_idx=pred_idx,
        confidence=confidence,
        arabic=arabic,
        english=english,
        session_id="batch",
        frames_in_buffer=len(frame_sequence),
        is_prediction_reliable=True
    )

@app.post("/predict_simple", response_model=PredictionResponse, summary="Predict ASL sign with automatic session", tags=["Prediction"])
async def predict_simple_api(
    file: UploadFile = File(...),
    user_id: Optional[str] = Header(None, alias="X-User-ID")
):
    """
    SIMPLE APPROACH: Automatic session management based on user_id.
    Just send a simple user identifier (like device ID) and frames accumulate automatically.

    Args:
        file: Single image file
        user_id: Simple user identifier (device ID, username, etc.) in X-User-ID header
    Returns:
        PredictionResponse: predicted class index, confidence, Arabic and English labels
    """
    # Auto-generate user_id if not provided (uses IP-based identification)
    if not user_id:
        # Simple fallback - you could use device ID or any simple identifier
        user_id = "default_user"

    # Clean up expired sessions
    cleanup_expired_sessions()

    # Auto-create session if needed
    if user_id not in session_buffers:
        session_buffers[user_id] = SessionData()

    session_data = session_buffers[user_id]
    session_data.update_timestamp()

    try:
        image_bytes = await file.read()
        npimg = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Invalid image file.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {e}")

    # Extract keypoints and add to buffer
    keypoints = extract_keypoints_api(frame)
    session_data.frame_buffer.append(keypoints)

    # Maintain rolling buffer
    if len(session_data.frame_buffer) > SEQUENCE_LENGTH:
        session_data.frame_buffer.pop(0)

    frames_count = len(session_data.frame_buffer)
    is_reliable = frames_count >= MIN_FRAMES_FOR_PREDICTION

    pred_idx, confidence = -1, 0.0

    if is_reliable:
        sequence = preprocess_sequence_api(session_data.frame_buffer)
        pred_idx, confidence = predict_api_onnx(sequence)

    arabic = idx2ar.get(pred_idx, "Unknown" if pred_idx != -1 else f"Collecting... {frames_count}/{MIN_FRAMES_FOR_PREDICTION}")
    english = idx2eng.get(pred_idx, "Unknown" if pred_idx != -1 else f"Collecting... {frames_count}/{MIN_FRAMES_FOR_PREDICTION}")

    return PredictionResponse(
        pred_idx=pred_idx,
        confidence=confidence,
        arabic=arabic,
        english=english,
        session_id=user_id,
        frames_in_buffer=frames_count,
        is_prediction_reliable=is_reliable
    )

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time ASL prediction.

    This is the RECOMMENDED approach for mobile apps!

    Protocol:
    1. Connect to /ws/{your_client_id}
    2. Send frames as JSON: {"type": "frame", "data": "base64_image_data"}
    3. Receive predictions as JSON: {"type": "prediction", "prediction": {...}}

    Benefits over HTTP:
    - Lower latency (no HTTP overhead)
    - Real-time streaming
    - Automatic session management
    - Better mobile app integration
    """
    await manager.connect(websocket, client_id)

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            message = WebSocketMessage(**data)

            if message.type == "frame" and message.data:
                # Process frame
                response = await process_websocket_frame(message.data, client_id)
                await manager.send_personal_message(response.dict(), client_id)

            elif message.type == "ping":
                # Heartbeat
                response = WebSocketResponse(
                    type="pong",
                    timestamp=time.time()
                )
                await manager.send_personal_message(response.dict(), client_id)

            elif message.type == "get_status":
                # Session status
                session_data = manager.get_session(client_id)
                if session_data:
                    frames_count = len(session_data.frame_buffer)
                    status = {
                        "frames_in_buffer": frames_count,
                        "is_reliable": frames_count >= MIN_FRAMES_FOR_PREDICTION,
                        "frames_needed": max(0, MIN_FRAMES_FOR_PREDICTION - frames_count),
                        "buffer_full": frames_count >= SEQUENCE_LENGTH
                    }
                    response = WebSocketResponse(
                        type="status",
                        status=status,
                        timestamp=time.time()
                    )
                    await manager.send_personal_message(response.dict(), client_id)

    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        error_response = WebSocketResponse(
            type="error",
            error=str(e),
            timestamp=time.time()
        )
        await manager.send_personal_message(error_response.dict(), client_id)
        manager.disconnect(client_id)

async def process_websocket_frame(base64_data: str, client_id: str) -> WebSocketResponse:
    """
    Process a single frame received via WebSocket.
    """
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(base64_data)
        npimg = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if frame is None:
            return WebSocketResponse(
                type="error",
                error="Invalid image data",
                timestamp=time.time()
            )

        # Get session data
        session_data = manager.get_session(client_id)
        if not session_data:
            return WebSocketResponse(
                type="error",
                error="Session not found",
                timestamp=time.time()
            )

        session_data.update_timestamp()

        # Extract keypoints and add to buffer
        keypoints = extract_keypoints_api(frame)
        session_data.frame_buffer.append(keypoints)

        # Maintain rolling buffer
        if len(session_data.frame_buffer) > SEQUENCE_LENGTH:
            session_data.frame_buffer.pop(0)

        frames_count = len(session_data.frame_buffer)
        is_reliable = frames_count >= MIN_FRAMES_FOR_PREDICTION

        pred_idx, confidence = -1, 0.0

        if is_reliable:
            sequence = preprocess_sequence_api(session_data.frame_buffer)
            pred_idx, confidence = predict_api_onnx(sequence)

        arabic = idx2ar.get(pred_idx, "Unknown" if pred_idx != -1 else f"Collecting... {frames_count}/{MIN_FRAMES_FOR_PREDICTION}")
        english = idx2eng.get(pred_idx, "Unknown" if pred_idx != -1 else f"Collecting... {frames_count}/{MIN_FRAMES_FOR_PREDICTION}")

        prediction = PredictionResponse(
            pred_idx=pred_idx,
            confidence=confidence,
            arabic=arabic,
            english=english,
            session_id=client_id,
            frames_in_buffer=frames_count,
            is_prediction_reliable=is_reliable
        )

        return WebSocketResponse(
            type="prediction",
            prediction=prediction,
            timestamp=time.time()
        )

    except Exception as e:
        return WebSocketResponse(
            type="error",
            error=f"Processing error: {str(e)}",
            timestamp=time.time()
        )
