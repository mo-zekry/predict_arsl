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
from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import onnxruntime as ort
import json
import os
import uuid
import time
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

# Global session storage
session_buffers: Dict[str, SessionData] = {}

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
        "predict_endpoint": "/predict",
        "session_management": {
            "start_session": "/start_session",
            "session_status": "/session/{session_id}/status",
            "clear_session": "/session/{session_id}"
        },
        "important_note": "Use session management to maintain frame sequences across requests!"
    }

@app.get("/health", tags=["Health"])
def health_check():
    """
    Returns API health status.
    """
    return {"status": "ok"}
