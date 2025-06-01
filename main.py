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
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import onnxruntime as ort
import json
import os
from typing import List

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

# Buffer to store sequence per client (for demo, global, for production use session/user id)
frame_buffer: List[np.ndarray] = []

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

@app.post("/predict", response_model=PredictionResponse, summary="Predict ASL sign from image frame", tags=["Prediction"])
async def predict_api(file: UploadFile = File(...)):
    """
    Accepts a single image frame (as JPEG/PNG) and returns the predicted ASL sign.
    The API maintains a rolling buffer of frames to form a sequence for the transformer model.

    Args:
        file: Uploaded image file (frame from mobile camera)
    Returns:
        PredictionResponse: predicted class index, confidence, Arabic and English labels
    """
    try:
        image_bytes = await file.read()
        npimg = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Invalid image file.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {e}")

    keypoints = extract_keypoints_api(frame)
    frame_buffer.append(keypoints)
    if len(frame_buffer) > SEQUENCE_LENGTH:
        frame_buffer.pop(0)
    pred_idx, confidence = -1, 0.0
    if len(frame_buffer) >= 5:
        sequence = preprocess_sequence_api(frame_buffer)
        pred_idx, confidence = predict_api_onnx(sequence)
    arabic = idx2ar.get(pred_idx, "Unknown")
    english = idx2eng.get(pred_idx, "Unknown")
    return PredictionResponse(pred_idx=pred_idx, confidence=confidence, arabic=arabic, english=english)

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
    """
    if len(seq) < SEQUENCE_LENGTH:
        padding = [np.zeros(FEATURE_DIM, dtype=np.float32)] * (SEQUENCE_LENGTH - len(seq))
        seq = seq + padding
    else:
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

@app.get("/", tags=["Health"])
def root():
    """
    Health check endpoint. Returns API status and documentation link.
    """
    return {
        "message": "ASL Transformer ONNX API for Mobile is running.",
        "docs_url": "/docs",
        "predict_endpoint": "/predict"
    }

@app.get("/health", tags=["Health"])
def health_check():
    """
    Returns API health status.
    """
    return {"status": "ok"}
