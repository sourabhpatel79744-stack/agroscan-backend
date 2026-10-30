"""
AgroScan Backend - FastAPI
Serves the trained crop disease model, Hindi advisory content, and scan history.

Setup:
1. pip install fastapi uvicorn python-multipart pillow
   (torch/torchvision already installed from training)
2. Make sure these files are in the same folder as this script:
     - disease_model_best.pth   (trained model)
     - advisory_data.json       (Hindi advisory content)
3. Run: uvicorn backend_main:app --reload
4. Open http://127.0.0.1:8000/docs in your browser to test it interactively (Swagger UI)
"""

import os
import json
import uuid
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ----------------------------
# Config
# ----------------------------
CHECKPOINT_PATH = "disease_model_v2.pth"
ADVISORY_PATH = "advisory_data.json"
HISTORY_PATH = "scan_history.json"   # simple local JSON "database" for now
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

# ----------------------------
# App setup
# ----------------------------
app = FastAPI(title="AgroScan API", description="Crop disease detection and Hindi advisory API")

# Allow the React frontend (running on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this to your actual frontend URL before deploying
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Load model once at startup
# ----------------------------
print(f"Loading model on {DEVICE}...")
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, len(CLASS_NAMES))

if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(f"{CHECKPOINT_PATH} not found - make sure it's in the same folder as this script.")

model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()
print("Model loaded successfully.")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ----------------------------
# Load advisory content
# ----------------------------
if os.path.exists(ADVISORY_PATH):
    with open(ADVISORY_PATH, "r", encoding="utf-8") as f:
        advisory_data = json.load(f)
else:
    advisory_data = {}
    print(f"WARNING: {ADVISORY_PATH} not found - advisory lookups will return defaults only.")

# ----------------------------
# Simple local JSON "history store" (swap for Firebase/MongoDB later)
# ----------------------------
def load_history():
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_history(history):
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

# ----------------------------
# Response models
# ----------------------------
class PredictionResponse(BaseModel):
    scan_id: str
    predicted_class: str
    confidence: float
    disease_name_hindi: str
    disease_name_english: str
    precautions_hindi: str
    treatment_hindi: str
    recovery_time_hindi: str
    timestamp: str

class HistoryEntry(BaseModel):
    scan_id: str
    predicted_class: str
    confidence: float
    disease_name_hindi: str
    timestamp: str

# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def root():
    return {
        "message": "AgroScan API is running",
        "endpoints": ["/predict (POST)", "/advisory/{disease_id} (GET)", "/history (GET)"],
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    # Read and preprocess the image
    try:
        image_bytes = await file.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the uploaded image. Try a different file.")

    input_tensor = preprocess(pil_image).unsqueeze(0).to(DEVICE)

    # Run prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)

    predicted_class = CLASS_NAMES[pred_idx.item()]
    confidence_value = round(confidence.item() * 100, 2)

    # Look up advisory content (fall back to default if not yet filled in)
    advisory = advisory_data.get(predicted_class, advisory_data.get("_default", {
        "disease_name_hindi": predicted_class,
        "disease_name_english": predicted_class,
        "precautions_hindi": "जानकारी जल्द उपलब्ध होगी।",
        "treatment_hindi": "जल्द अपडेट किया जाएगा।",
        "recovery_time_hindi": "जल्द अपडेट किया जाएगा।",
    }))

    scan_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()

    # Log to history
    history = load_history()
    history.append({
        "scan_id": scan_id,
        "predicted_class": predicted_class,
        "confidence": confidence_value,
        "disease_name_hindi": advisory.get("disease_name_hindi", predicted_class),
        "timestamp": timestamp,
    })
    save_history(history)

    return PredictionResponse(
        scan_id=scan_id,
        predicted_class=predicted_class,
        confidence=confidence_value,
        disease_name_hindi=advisory.get("disease_name_hindi", predicted_class),
        disease_name_english=advisory.get("disease_name_english", predicted_class),
        precautions_hindi=advisory.get("precautions_hindi", ""),
        treatment_hindi=advisory.get("treatment_hindi", ""),
        recovery_time_hindi=advisory.get("recovery_time_hindi", ""),
        timestamp=timestamp,
    )

@app.get("/advisory/{disease_id}")
def get_advisory(disease_id: str):
    advisory = advisory_data.get(disease_id)
    if advisory is None:
        raise HTTPException(status_code=404, detail=f"No advisory found for '{disease_id}'.")
    return advisory

@app.get("/history")
def get_history(limit: Optional[int] = 20):
    history = load_history()
    # Most recent first
    return list(reversed(history))[:limit]

@app.get("/classes")
def get_classes():
    """Returns the full list of disease classes the model can predict."""
    return {"classes": CLASS_NAMES, "count": len(CLASS_NAMES)}
