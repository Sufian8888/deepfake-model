"""
Model API Server for Deepfake Detection
This service runs the PyTorch model and provides REST API endpoints
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import torch
import cv2
import numpy as np
from torchvision import transforms
import timm
import torch.nn as nn
from pathlib import Path
import os
import tempfile
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Model architecture (must match training)
class DeepfakeDetector(nn.Module):
    def __init__(self):
        super(DeepfakeDetector, self).__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0)
        self.classifier = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output

# Global model runtime state
model = None
device = None
loaded_model_key = None
loaded_model_path = None


def get_available_model_files():
    """Return available checkpoint files in model_output as {key, label, path}."""
    model_dir = Path(__file__).parent / "model_output"
    if not model_dir.exists():
        return []

    files = []
    for pth_file in sorted(model_dir.glob("*.pth")):
        files.append({
            "key": pth_file.stem,
            "label": pth_file.name,
            "path": pth_file,
        })
    return files


def resolve_model_path(model_key: str | None):
    available = get_available_model_files()
    if not available:
        return None

    if model_key:
        for model_file in available:
            if model_file["key"] == model_key:
                return model_file["path"]

    # Prefer final_model if present, otherwise first file.
    for model_file in available:
        if model_file["key"] == "final_model":
            return model_file["path"]

    return available[0]["path"]


def extract_state_dict(checkpoint):
    """Handle different checkpoint formats."""
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
    return checkpoint


def load_model(model_key: str | None = None):
    """Load selected trained model checkpoint into memory."""
    global model, device, loaded_model_key, loaded_model_path

    try:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {device}")

        model_path = resolve_model_path(model_key)

        if model_path is None:
            print("⚠️ No model files found in model_output")
            print("⚠️ Running in demo mode with random predictions")
            model = None
            loaded_model_key = None
            loaded_model_path = None
            return False

        if loaded_model_path == str(model_path) and model is not None:
            return True

        detector = DeepfakeDetector()
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        state_dict = extract_state_dict(checkpoint)
        detector.load_state_dict(state_dict, strict=False)
        detector.to(device)
        detector.eval()

        model = detector
        loaded_model_key = model_path.stem
        loaded_model_path = str(model_path)
        print(f"✅ Model loaded successfully from {model_path}")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("⚠️ Running in demo mode with random predictions")
        model = None
        loaded_model_key = None
        loaded_model_path = None
        return False


# Load model on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    load_model(os.getenv("DEFAULT_MODEL_KEY", "final_model"))
    yield
    # Shutdown
    pass

# Create FastAPI app
app = FastAPI(
    title="Deepfake Detection Model API", 
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for annotated frames
analysis_results_dir = os.path.join(os.path.dirname(__file__), 'analysis_results')
os.makedirs(analysis_results_dir, exist_ok=True)
app.mount("/model/analysis_results", StaticFiles(directory=analysis_results_dir), name="analysis_results")

def extract_frames(video_path, num_frames=5, frame_rate=30):
    """Extract frames from video"""
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    frame_count = 0
    extracted = 0
    
    while extracted < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_rate == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            extracted += 1
        
        frame_count += 1
    
    cap.release()
    
    if len(frames) == 0:
        raise ValueError("No frames extracted from video")
    
    return frames

def preprocess_frame(frame):
    """Preprocess a single frame for the model"""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(frame)

def save_annotated_frames(video_path, raw_frames, predictions):
    """Save annotated frames with face detection"""
    import cv2
    
    # Create results folder
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join(os.path.dirname(__file__), 'analysis_results', f"{video_name}_{timestamp}")
    frames_folder = os.path.join(output_folder, 'frames')
    os.makedirs(frames_folder, exist_ok=True)
    
    # Load face detector
    haarcascade_path = os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier(haarcascade_path)
    
    annotated_paths = []
    frame_details = []
    
    for i, (raw_frame, pred_score) in enumerate(zip(raw_frames, predictions)):
        # Detect faces
        gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Create annotated frame
        annotated = raw_frame.copy()
        h, w = annotated.shape[:2]
        
        label = "FAKE" if pred_score > 0.5 else "REAL"
        confidence = float(pred_score * 100 if pred_score > 0.5 else (1 - pred_score) * 100)
        
        color = (0, 0, 255) if label == "FAKE" else (0, 255, 0)
        
        # Draw face boxes
        for (x, y, fw, fh) in faces:
            cv2.rectangle(annotated, (x, y), (x+fw, y+fh), color, 4)
            text = f"{label} FACE"
            cv2.rectangle(annotated, (x, y-30), (x+150, y), color, -1)
            cv2.putText(annotated, text, (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add header overlay
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), color, -1)
        cv2.addWeighted(overlay, 0.3, annotated, 0.7, 0, annotated)
        
        # Add text
        cv2.putText(annotated, f"Frame {i+1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(annotated, f"{label}: {confidence:.1f}%", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        # Save frame
        frame_filename = f"frame_{i+1:02d}_{label}.jpg"
        frame_path = os.path.join(frames_folder, frame_filename)
        cv2.imwrite(frame_path, annotated)
        
        annotated_paths.append(frame_path)
        frame_details.append({
            "frame_num": i + 1,
            "label": label,
            "confidence": confidence,
            "raw_score": float(pred_score),
            "is_suspicious": bool(pred_score > 0.5),
            "faces_detected": int(len(faces))
        })
    
    return output_folder, annotated_paths, frame_details

def analyze_video_with_model(video_path, model_key: str | None = None):
    """Analyze video using the selected model."""
    global model, device, loaded_model_key

    logger.info(f"🎬 Starting analysis for: {video_path}")

    # Switch model when requested by client.
    if model_key and model_key != loaded_model_key:
        logger.info(f"🔄 Switching model to: {model_key}")
        load_model(model_key)

    if model is None:
        # Demo mode - return random results
        logger.warning("⚠️ Model not loaded, using demo mode")
        import random
        is_fake = random.choice([True, False])
        confidence = random.uniform(60, 95)
        
        return {
            "is_deepfake": is_fake,
            "confidence_score": confidence,
            "analysis_details": {
                "mode": "demo",
                "facial_consistency": random.uniform(50, 95),
                "audio_sync": random.uniform(50, 95),
                "artifacts_detected": random.choice([True, False]),
                "frame_analysis": {
                    "total_frames": random.randint(100, 500),
                    "suspicious_frames": random.randint(0, 50)
                },
                "annotated_frames": []
            }
        }
    
    # Extract frames (these are RGB frames for processing)
    logger.info("📹 Extracting frames...")
    frames = extract_frames(video_path)
    logger.info(f"✅ Extracted {len(frames)} frames")
    
    # Extract raw frames again (BGR for OpenCV annotation)
    logger.info("🎞️ Extracting raw frames for annotation...")
    raw_frames = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        logger.error(f"❌ Cannot open video: {video_path}")
        raise ValueError(f"Cannot open video: {video_path}")
    
    frame_count = 0
    extracted = 0
    num_frames = 5  # Reduced from 10 to 5 to save memory
    frame_rate = 30  # Increased from 15 to 30 (skip more frames)
    
    while extracted < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_rate == 0:
            raw_frames.append(frame)  # Keep as BGR
            extracted += 1
        
        frame_count += 1
    
    cap.release()
    logger.info(f"✅ Extracted {len(raw_frames)} raw frames")
    
    # Ensure we have the same number of frames
    if len(raw_frames) != len(frames):
        min_len = min(len(raw_frames), len(frames))
        raw_frames = raw_frames[:min_len]
        frames = frames[:min_len]
    
    # Preprocess frames
    logger.info("🔧 Preprocessing frames...")
    processed_frames = torch.stack([preprocess_frame(frame) for frame in frames])
    processed_frames = processed_frames.to(device)
    logger.info(f"✅ Preprocessed frames shape: {processed_frames.shape}")
    
    # Run inference
    logger.info("🧠 Running model inference...")
    with torch.no_grad():
        predictions = model(processed_frames)
        predictions = predictions.cpu().numpy().flatten()
    logger.info(f"✅ Inference complete. Predictions: {predictions}")
    
    # Save annotated frames
    output_folder, annotated_paths, frame_details = save_annotated_frames(video_path, raw_frames, predictions)
    
    # Calculate metrics
    avg_prediction = float(np.mean(predictions))
    is_deepfake = bool(avg_prediction > 0.5)
    confidence_score = float(avg_prediction * 100 if is_deepfake else (1 - avg_prediction) * 100)
    
    # Count suspicious frames
    suspicious_count = int(np.sum(predictions > 0.5))
    fake_frames = suspicious_count
    real_frames = len(predictions) - suspicious_count
    
    # Calculate consistency score based on variance
    consistency_score = float((1 - np.std(predictions)) * 100)
    
    # Detailed analysis
    analysis_details = {
        "mode": "model",
        "model_key": loaded_model_key,
        "model_file": os.path.basename(loaded_model_path) if loaded_model_path else None,
        "facial_consistency": consistency_score,
        "temporal_consistency": float(100 - (np.std(predictions) * 150)),  # Inverse of variance
        "artifacts_detected": bool(suspicious_count > len(frames) * 0.3),
        "frame_analysis": {
            "total_frames": len(frames),
            "suspicious_frames": suspicious_count,
            "fake_frames": fake_frames,
            "real_frames": real_frames,
            "frame_scores": predictions.tolist(),
            "frame_details": frame_details
        },
        "annotated_frames": [os.path.relpath(p, os.path.dirname(os.path.dirname(__file__))) for p in annotated_paths],
        "output_folder": output_folder
    }
    
    return {
        "is_deepfake": is_deepfake,
        "confidence_score": confidence_score,
        "analysis_details": analysis_details
    }

@app.get("/")
async def root():
    return {
        "message": "Deepfake Detection Model API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model is not None,
        "loaded_model_key": loaded_model_key,
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "not initialized",
        "loaded_model_key": loaded_model_key,
        "loaded_model_path": loaded_model_path,
    }

@app.get("/models")
async def list_models():
    available = get_available_model_files()
    return {
        "models": [
            {"key": model_file["key"], "label": model_file["label"]}
            for model_file in available
        ]
    }

@app.post("/analyze")
async def analyze_video(
    file: UploadFile = File(...),
    model_key: str = Form("final_model"),
):
    """Analyze a video file for deepfake detection"""
    
    logger.info(f"📹 Analyzing video: {file.filename} with model: {model_key}")
    
    # Validate file type
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.webm')):
        logger.error(f"❌ Invalid file type: {file.filename}")
        raise HTTPException(status_code=400, detail="Invalid file type. Only video files are supported.")
    
    # Save uploaded file temporarily
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"temp_video_{datetime.now().timestamp()}.mp4")
    
    try:
        # Save file
        logger.info(f"💾 Saving temp file: {temp_path}")
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        logger.info(f"✅ Temp file saved: {len(content)} bytes")
        
        try:
            # Analyze video with selected model key
            logger.info(f"🔍 Starting video analysis...")
            result = analyze_video_with_model(temp_path, model_key=model_key)
            logger.info(f"✅ Analysis complete: deepfake={result['is_deepfake']}")
            
            # Ensure result can be serialized
            import json
            json_test = json.dumps(result)
            logger.info(f"✅ Result is valid JSON: {len(json_test)} bytes")
            
            return result
        except MemoryError as e:
            logger.error(f"❌ Out of memory: {str(e)}")
            # Return error response instead of crashing
            return {
                "is_deepfake": False,
                "confidence_score": 0.0,
                "error": "Model ran out of memory",
                "analysis_details": {"mode": "error", "reason": "memory"}
            }
        except Exception as e:
            logger.error(f"❌ Analysis error: {type(e).__name__}: {str(e)}", exc_info=True)
            raise
        
    except Exception as e:
        logger.error(f"❌ Error analyzing video: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error analyzing video: {str(e)}")
    
    finally:
        # Cleanup
        logger.info(f"🧹 Cleaning up temp file: {temp_path}")
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.info(f"✅ Temp file deleted")
            except Exception as e:
                logger.error(f"❌ Error deleting temp file: {e}")


@app.post("/analyze-path")
async def analyze_video_path(video_path: str, model_key: str = "final_model"):
    """Analyze a video file by path"""
    
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    
    try:
        result = analyze_video_with_model(video_path, model_key=model_key)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing video: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 5000))
    print("🚀 Starting Deepfake Detection Model API...")
    print(f"📝 API will be available at: http://0.0.0.0:{port}")
    print(f"📝 API Documentation: http://0.0.0.0:{port}/docs")
    uvicorn.run(app, host="0.0.0.0", port=port)
