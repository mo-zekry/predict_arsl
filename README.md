# ASL Transformer Prediction API

A FastAPI backend for real-time Arabic Sign Language (ASL) prediction using a transformer model in ONNX format. Designed for mobile camera integration (e.g., Flutter apps).

---

## Features
- **/predict**: Upload a camera frame, get ASL sign prediction (Arabic & English).
- **/health**: Health check endpoint.
- **/docs**: Interactive API documentation (Swagger UI).
- CORS enabled for mobile/web integration.
- Modular, production-ready Python code.

---

## Setup & Installation

1. **Clone the repository** and navigate to the `predict_arsl` directory:
   ```sh
   cd predict_arsl
   ```

2. **Install dependencies** (recommended: use a virtual environment):
   ```sh
   pip install -r requirements.txt
   ```

3. **Ensure the following files are present in `predict_arsl/`:**
   - `asl_transformer.onnx` (ONNX model)
   - `idx2ar.json` (Arabic label mapping)
   - `idx2eng.json` (English label mapping)

4. **Run the API server:**
   ```sh
   uvicorn main:app --reload
   ```
   The API will be available at `http://127.0.0.1:8000/`

---

## API Endpoints

### 1. Health Check
- **GET /** or **GET /health**
- **Response:**
  ```json
  { "status": "ok" }
  ```

### 2. Predict ASL Sign
- **POST /predict**
- **Request:**
  - `file`: Image file (frame from mobile camera, JPEG/PNG)
- **Response:**
  ```json
  {
    "pred_idx": 12,
    "confidence": 0.98,
    "arabic": "مرحبا",
    "english": "Hello"
  }
  ```
- **Description:**
  Upload a single frame. The API maintains a rolling buffer of frames for sequence-based prediction. For best results, send frames in real-time from the mobile app.

---

## Example (Python)
```python
import requests

url = "http://127.0.0.1:8000/predict"
with open("frame.jpg", "rb") as f:
    files = {"file": ("frame.jpg", f, "image/jpeg")}
    response = requests.post(url, files=files)
    print(response.json())
```

---

## Integration Notes
- **Flutter:** Use `http.MultipartRequest` to send camera frames to `/predict`.
- **Buffer:** The API uses a global buffer for demo. For production, use per-user/session buffers.
- **Keypoints:** The current implementation uses image resizing as a placeholder. Replace `extract_keypoints_api` with real keypoint extraction for higher accuracy.

---

## Authors
Graduation Project Team 2025

## License
[MIT](LICENSE)
