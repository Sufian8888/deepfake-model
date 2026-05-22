"""
Model API Server for Deepfake Detection
This service runs the PyTorch model and provides REST API endpoints
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import asyncio
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

# Reduce default Torch thread usage to lower memory/CPU pressure on small instances
try:
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
except Exception:
    pass

# Model architectures must match the checkpoint that is being loaded.
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


class ConvNeXtDetector(nn.Module):
    def __init__(self, num_classes=6):
        super(ConvNeXtDetector, self).__init__()
        self.backbone = timm.create_model('convnext_tiny', pretrained=False, num_classes=num_classes)

    def forward(self, x):
        return self.backbone(x)

# Global model runtime state
model = None
device = None
loaded_model_key = None
loaded_model_path = None
loaded_model_architecture = None

CLASS_NAMES_6 = ['Deepfakes', 'Face2Face', 'FaceShifter', 'FaceSwap', 'NeuralTextures', 'original']
FAKE_CLASS_NAMES_6 = set(CLASS_NAMES_6[:-1])
REAL_CLASS_NAMES_6 = {CLASS_NAMES_6[-1]}


def get_available_model_files():
    """Return available checkpoint files in model_output or root directory as {key, label, path}."""
    files = []
    
    # Check model_output subdirectory first (local development)
    model_dir = Path(__file__).parent / "model_output"
    if model_dir.exists():
        for pth_file in sorted(model_dir.glob("*.pth")):
            files.append({
                "key": pth_file.stem,
                "label": pth_file.name,
                "path": pth_file,
            })
    
    # Also check root directory (Hugging Face Spaces deployment)
    root_dir = Path(__file__).parent
    for pth_file in sorted(root_dir.glob("*.pth")):
        # Avoid duplicates
        if not any(f["path"] == pth_file for f in files):
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


def remap_state_dict_keys(state_dict: dict) -> dict:
    """Fix state dict key mismatch by adding/removing 'backbone.' prefix."""
    # Check if keys need 'backbone.' prefix added
    if state_dict and not any(k.startswith("backbone.") for k in list(state_dict.keys())[:5]):
        # Keys don't have 'backbone.' prefix, add it to ALL keys
        logger.info("🔄 Remapping state dict: adding 'backbone.' prefix to ALL keys...")
        new_state_dict = {}
        for key, value in state_dict.items():
            new_state_dict[f"backbone.{key}"] = value
        return new_state_dict
    
    return state_dict


def infer_model_architecture(model_key: str | None, state_dict: dict) -> str:
    """Infer checkpoint architecture from the model key or checkpoint tensor names."""
    if model_key and "deepfake_master_model" in model_key:
        return "convnext_tiny"

    state_keys = list(state_dict.keys())
    if any(key.startswith(("stages.", "downsample_layers.", "head.")) for key in state_keys):
        return "convnext_tiny"

    return "efficientnet_b0"


def infer_num_classes(architecture: str, state_dict: dict) -> int:
    """Infer classifier width from the checkpoint tensors."""
    candidate_keys = ["head.weight", "head.fc.weight", "classifier.6.weight", "classifier.weight"]
    for key in candidate_keys:
        tensor = state_dict.get(key)
        if tensor is not None and hasattr(tensor, "shape") and len(tensor.shape) > 0:
            return int(tensor.shape[0])

    return 6 if architecture == "convnext_tiny" else 1


def normalize_model_outputs(outputs: torch.Tensor, architecture: str | None):
    """Convert raw model outputs into fake probabilities, predicted classes, and confidence values."""
    if outputs.ndim == 0:
        outputs = outputs.view(1)

    if outputs.ndim == 1:
        outputs = outputs.unsqueeze(-1)

    # Multi-class ConvNeXt checkpoints should use softmax and map the last class to real.
    if outputs.ndim == 2 and outputs.shape[-1] > 1:
        probabilities = torch.softmax(outputs, dim=-1)
        predicted_classes = torch.argmax(probabilities, dim=-1)

        if probabilities.shape[-1] == 6:
            fake_probabilities = probabilities[:, :5].sum(dim=-1)
            predicted_class_names = [CLASS_NAMES_6[idx] for idx in predicted_classes.tolist()]
            return fake_probabilities, predicted_class_names, predicted_classes, probabilities

        frame_confidence, _ = torch.max(probabilities, dim=-1)
        predicted_class_names = [str(idx) for idx in predicted_classes.tolist()]
        return frame_confidence, predicted_class_names, predicted_classes, probabilities

    scores = outputs.squeeze(-1)

    # Binary checkpoints may already emit probabilities via Sigmoid; only sigmoid raw logits.
    if torch.any((scores < 0) | (scores > 1)):
        scores = torch.sigmoid(scores)

    scores = torch.clamp(scores, 0.0, 1.0)
    predicted_class_names = ["fake" if float(score) >= 0.5 else "real" for score in scores.tolist()]
    return scores, predicted_class_names, None, None


def build_detector(architecture: str, num_classes: int):
    if architecture == "convnext_tiny":
        logger.info(f"🏗️ Creating ConvNeXtDetector architecture with {num_classes} classes...")
        return ConvNeXtDetector(num_classes=num_classes)

    logger.info("🏗️ Creating DeepfakeDetector architecture...")
    return DeepfakeDetector()


def load_model(model_key: str | None = None):
    """Load selected trained model checkpoint into memory."""
    global model, device, loaded_model_key, loaded_model_path, loaded_model_architecture

    try:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {device}")

        available_models = get_available_model_files()
        logger.info(f"📁 Available models: {[m['key'] for m in available_models]}")
        
        model_path = resolve_model_path(model_key)

        if model_path is None:
            logger.error("⚠️ No model files found in model_output")
            logger.warning("⚠️ Running in demo mode with random predictions")
            model = None
            loaded_model_key = None
            loaded_model_path = None
            loaded_model_architecture = None
            return False

        logger.info(f"🔍 Loading model from: {model_path}")

        if loaded_model_path == str(model_path) and model is not None:
            logger.info(f"✅ Model already loaded: {model_path}")
            return True

        logger.info(f"📦 Loading checkpoint: {model_path}")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        logger.info(f"✅ Checkpoint loaded, extracting state dict...")
        
        state_dict = extract_state_dict(checkpoint)
        state_dict = remap_state_dict_keys(state_dict)  # Fix key mismatch
        architecture = infer_model_architecture(model_key, state_dict)
        num_classes = infer_num_classes(architecture, state_dict)
        logger.info(f"🔎 Inferred architecture: {architecture}, num_classes: {num_classes}")

        detector = build_detector(architecture, num_classes)
        logger.info(f"🔄 Applying state dict ({len(state_dict)} parameters)...")
        detector.load_state_dict(state_dict, strict=True)
        detector.to(device)
        detector.eval()

        model = detector
        loaded_model_key = model_key or model_path.stem
        loaded_model_path = str(model_path)
        loaded_model_architecture = architecture
        logger.info(f"✅ Model loaded successfully from {model_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Error loading model: {type(e).__name__}: {str(e)}", exc_info=True)
        logger.warning("⚠️ Running in demo mode with random predictions")
        model = None
        loaded_model_key = None
        loaded_model_path = None
        loaded_model_architecture = None
        return False


# Load model on startup (load in background to avoid long blocking startup)
async def _async_load_model(default_key: str | None):
    try:
        await asyncio.to_thread(load_model, default_key)
    except Exception:
        logger.exception("Failed to load model in background")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start background model loading so server can bind to port immediately
    default_key = os.getenv("DEFAULT_MODEL_KEY", "deepfake_master_model")
    logger.info(f"🔁 Scheduling background model load: {default_key}")
    asyncio.create_task(_async_load_model(default_key))
    yield
    # Shutdown (no-op)
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

def extract_frames(video_path, num_frames=15, frame_rate=30):
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

class GradCAM:
    """Grad-CAM implementation for ConvNeXt models"""
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        
        # Hook into the last stage of ConvNeXt
        if hasattr(model, 'stages'):
            target = model.stages[-1]
            target.register_forward_hook(self._save_activation)
            target.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, x, class_idx=None):
        """Generate Grad-CAM heatmap"""
        self.model.eval()
        output = self.model(x)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        output[0, class_idx].backward(retain_graph=True)
        
        # Compute heatmap
        if self.gradients is not None and self.activations is not None:
            grads = self.gradients.mean(dim=[2, 3], keepdim=True)
            cam = (grads * self.activations).sum(dim=1).squeeze()
            cam = torch.relu(cam)
            cam = cam.cpu().detach().numpy()
            
            if cam.max() > 0:
                cam = (cam - cam.min()) / (cam.max() - cam.min())
            return cam
        
        return np.zeros((x.shape[2] // 32, x.shape[3] // 32))

def overlay_gradcam(frame_bgr, cam):
    """Overlay Grad-CAM heatmap on frame"""
    h, w = frame_bgr.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap_colored = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(frame_bgr, 0.55, heatmap_colored, 0.45, 0)

def save_annotated_frames(video_path, raw_frames, fake_probabilities, predicted_class_names=None, gradcam_maps=None):
    """Save annotated frames with face detection and Grad-CAM heatmap overlay"""
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
    
    for i, (raw_frame, fake_prob) in enumerate(zip(raw_frames, fake_probabilities)):
        # Start with Grad-CAM overlay if available
        if gradcam_maps is not None and i < len(gradcam_maps):
            annotated = overlay_gradcam(raw_frame, gradcam_maps[i])
        else:
            annotated = raw_frame.copy()
        
        # Detect faces
        gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        h, w = annotated.shape[:2]
        
        is_fake = bool(float(fake_prob) >= 0.5)
        label = "FAKE" if is_fake else "REAL"
        confidence = float(max(float(fake_prob), 1.0 - float(fake_prob)) * 100)
        pred_class = predicted_class_names[i] if predicted_class_names and i < len(predicted_class_names) else ("fake" if is_fake else "real")
        
        color = (0, 0, 255) if label == "FAKE" else (0, 255, 0)
        
        # Draw face boxes on top of heatmap
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
        cv2.putText(annotated, f"{label} | {pred_class} | {confidence:.1f}%", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        
        # Save frame
        frame_filename = f"frame_{i+1:02d}_{label}.jpg"
        frame_path = os.path.join(frames_folder, frame_filename)
        cv2.imwrite(frame_path, annotated)
        
        annotated_paths.append(frame_path)
        frame_details.append({
            "frame_num": i + 1,
            "label": label,
            "confidence": confidence,
            "raw_score": float(fake_prob),
            "pred_class": pred_class,
            "prob_fake": float(fake_prob),
            "prob_real": float(1.0 - float(fake_prob)),
            "is_suspicious": is_fake,
            "faces_detected": int(len(faces))
        })
    
    return output_folder, annotated_paths, frame_details

def analyze_video_with_model(video_path, model_key: str | None = None):
    """Analyze video using the selected model."""
    global model, device, loaded_model_key, loaded_model_architecture

    logger.info(f"🎬 Starting analysis for: {video_path}")

    # Switch model when requested by client.
    if model_key and model_key != loaded_model_key:
        logger.info(f"🔄 Switching model to: {model_key}")
        load_model(model_key)

    if model is None:
        # Demo mode - but still extract frames for stats
        logger.warning("⚠️ Model not loaded, using demo mode")
        
        # Try to extract frames for analysis
        try:
            logger.info("📹 Extracting frames for demo mode...")
            frames = extract_frames(video_path, num_frames=15)
            logger.info(f"✅ Extracted {len(frames)} frames")
            num_extracted_frames = len(frames)
        except Exception as e:
            logger.error(f"❌ Cannot extract frames: {str(e)}")
            num_extracted_frames = 0
        
        import random
        is_fake = random.choice([True, False])
        confidence = random.uniform(60, 95)
        
        # Build frame-level demo data
        fake_count = random.randint(0, max(1, num_extracted_frames // 2))
        real_count = random.randint(0, max(1, num_extracted_frames // 2))
        suspicious_count = random.randint(0, max(1, num_extracted_frames // 2))
        
        frame_details_demo = [
            {
                "frame_number": i,
                "timestamp": float(i / 30.0),
                "is_fake": random.choice([True, False]),
                "is_suspicious": random.choice([True, False]),
                "confidence_score": random.uniform(40, 95),
            }
            for i in range(num_extracted_frames)
        ]
        
        frame_analysis_demo = {
            "total_frames": num_extracted_frames,
            "fake_frames": fake_count,
            "real_frames": real_count,
            "suspicious_frames": suspicious_count,
            "frame_details": frame_details_demo
        }
        
        return {
            "is_deepfake": is_fake,
            "confidence_score": confidence,
            "analysis_details": {
                "mode": "demo",
                "reason": "Model not loaded on server",
                "facial_consistency": random.uniform(50, 95),
                "audio_sync": random.uniform(50, 95),
                "artifacts_detected": random.choice([True, False]),
                "frame_analysis": frame_analysis_demo,
                "annotated_frames": []  # Empty in demo mode
            },
            "frame_analysis": frame_analysis_demo
        }
    
    # Extract frames (these are RGB frames for processing)
    logger.info("📹 Extracting frames...")
    frames = extract_frames(video_path, num_frames=15)
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
    num_frames = 15  # Match local report sampling
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = max(1, total_video_frames // num_frames) if total_video_frames > 0 else 30
    
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
    
    # Initialize Grad-CAM
    logger.info("📊 Initializing Grad-CAM for heatmap generation...")
    gradcam = GradCAM(model)
    gradcam_maps = []
    raw_outputs_list = []
    
    # Process each frame for inference + Grad-CAM
    logger.info("🧠 Running model inference with Grad-CAM for each frame...")
    try:
        for i, frame in enumerate(frames):
            try:
                # Preprocess single frame
                frame_tensor = preprocess_frame(frame)
                frame_batch = frame_tensor.unsqueeze(0).to(device)
                
                # Run inference with gradient tracking enabled
                with torch.set_grad_enabled(True):
                    output = model(frame_batch)
                    raw_outputs_list.append(output.detach().cpu())
                
                # Compute Grad-CAM with backward pass
                cam = gradcam.generate(frame_batch)
                gradcam_maps.append(cam)
                logger.info(f"   ✅ Frame {i+1}/{len(frames)}: inference + Grad-CAM")
                
            except Exception as e:
                logger.warning(f"   ⚠️ Frame {i+1} failed: {str(e)}")
                gradcam_maps.append(None)
                with torch.no_grad():
                    frame_tensor = preprocess_frame(frame)
                    frame_batch = frame_tensor.unsqueeze(0).to(device)
                    output = model(frame_batch)
                    raw_outputs_list.append(output.detach().cpu())
        
        # Combine outputs
        raw_outputs = torch.cat(raw_outputs_list, dim=0)
        fake_probabilities, predicted_class_names, predicted_classes, class_probabilities = normalize_model_outputs(
            raw_outputs,
            loaded_model_architecture,
        )
        predictions = fake_probabilities.numpy().flatten()
        logger.info(f"✅ Inference + Grad-CAM complete for {len(frames)} frames")
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {type(e).__name__}: {e}", exc_info=True)
        raise
    
    # Save annotated frames with Grad-CAM heatmaps
    output_folder, annotated_paths, frame_details = save_annotated_frames(
        video_path,
        raw_frames,
        predictions,
        predicted_class_names=predicted_class_names,
        gradcam_maps=gradcam_maps,
    )
    logger.info(f"✅ Saved {len(annotated_paths)} annotated frames with Grad-CAM to {output_folder}")
    
    # Log frame paths for debugging
    for i, path in enumerate(annotated_paths):
        logger.info(f"   Frame {i+1}: {path}")
    
    # Calculate metrics
    avg_prediction = float(np.mean(predictions))
    is_deepfake = bool(avg_prediction >= 0.5)
    confidence_score = float(max(avg_prediction, 1.0 - avg_prediction) * 100)
    
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
        "annotated_frames": [
            "/model/analysis_results/" + os.path.relpath(p, os.path.join(os.path.dirname(__file__), "analysis_results")).replace("\\", "/")
            for p in annotated_paths
        ],
        "output_folder": output_folder,
        "report_summary": {
            "final_label": "FAKE" if is_deepfake else "REAL",
            "final_confidence": round(confidence_score, 2),
            "avg_prob_fake": round(avg_prediction, 4),
            "fake_frames": fake_frames,
            "real_frames": real_frames,
            "total_frames": len(frames),
            "frame_breakdown": frame_details,
        }
    }
    
    # Build frame-level analysis for database storage
    frame_analysis_db = {
        "total_frames": len(frames),
        "fake_frames": fake_frames,
        "real_frames": real_frames,
        "suspicious_frames": suspicious_count,
        "frame_details": [
            {
                "frame_number": idx,
                "timestamp": float(idx / 30.0),  # Assuming 30 FPS
                "is_fake": bool(predictions[idx] >= 0.5),
                "is_suspicious": bool(predictions[idx] >= 0.5),
                "confidence_score": float(max(float(predictions[idx]), 1.0 - float(predictions[idx])) * 100),
                "analysis_details": frame_details[idx] if idx < len(frame_details) else {}
            }
            for idx in range(len(frames))
        ]
    }
    
    return {
        "is_deepfake": is_deepfake,
        "confidence_score": confidence_score,
        "analysis_details": analysis_details,
        "frame_analysis": frame_analysis_db
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
    
    # WAIT for model to load if not already loaded
    global model, loaded_model_key
    if model is None:
        logger.warning(f"⏳ Model not loaded yet. Waiting...")
        for attempt in range(30):  # Wait up to 30 seconds
            if model is not None:
                logger.info(f"✅ Model loaded after {attempt} attempts")
                break
            await asyncio.sleep(1)
        
        if model is None:
            logger.error(f"❌ Model still not loaded after 30 seconds. Proceeding with demo mode.")
    
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
    logger.info("🚀 Starting Deepfake Detection Model API...")
    logger.info(f"📝 API will be available at: http://0.0.0.0:{port}")
    logger.info(f"📝 API Documentation: http://0.0.0.0:{port}/docs")
    uvicorn.run(app, host="0.0.0.0", port=port)
