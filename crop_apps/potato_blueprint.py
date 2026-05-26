# crop_apps/rice_blueprint.py
from flask import Blueprint, render_template, request, jsonify
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as models, torchvision.transforms as transforms
from PIL import Image
import numpy as np, os, json
from werkzeug.utils import secure_filename

potato_bp = Blueprint("potato", __name__, url_prefix="/potato")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ✅ Upload folder
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads", "potato")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ Paths (relative)
POTATO_MODEL_PATH = os.path.join(BASE_DIR, "models", "best_resnet18.pth")
POTATO_CLASSES_PATH = os.path.join(BASE_DIR, "models", "class_names.npy")
DISEASE_KNOWLEDGE_FILE = os.path.join(BASE_DIR, "disease_knowledge", "potato_disease.json")
# --- Load classes
def load_classes():
    try:
        return np.load(POTATO_CLASSES_PATH, allow_pickle=True).tolist()
    except:
        return ["Potato_healthy"]

# --- Load knowledge
def load_knowledge():
    try:
        with open(DISEASE_KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

# --- Load model
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(POTATO_CLASSES))
    state_dict = torch.load(POTATO_MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

POTATO_CLASSES = load_classes()
DISEASE_KNOWLEDGE = load_knowledge()
try:
    potato_model = load_model()
except Exception as e:
    print(f"❌ Model load error: {e}")
    potato_model = None

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict(img_path):
    if potato_model is None:
        return "Model not loaded", 0

    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = potato_model(tensor)
        probs = F.softmax(outputs, dim=1)[0]
        idx = torch.argmax(probs).item()
        return POTATO_CLASSES[idx], probs[idx].item() * 100

def get_disease_info(name):
    return DISEASE_KNOWLEDGE.get(name, {
        "description": "No information available.",
        "cause": "Unknown",
        "treatment": ["Consult expert"],
        "prevention": ["Good agricultural practices"]
    })

@potato_bp.route("/")
def index():
    return render_template("crop_index.html",
                           crop_name="Potato",
                           classes=POTATO_CLASSES,
                           total_classes=len(POTATO_CLASSES))

@potato_bp.route("/upload", methods=["POST"])
def upload_file():
    file = request.files.get("file")
    if not file: return jsonify({"error": "No file uploaded"}), 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    disease, confidence = predict(filepath)
    info = get_disease_info(disease)

    return jsonify({
        "crop": "Potato",
        "prediction": disease,
        "confidence": f"{confidence:.2f}%",
        "disease_info": info,
        "image_path": f"/static/uploads/potato/{filename}"
    })

@potato_bp.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "crop": "Potato",
        "model_loaded": potato_model is not None,
        "classes_count": len(POTATO_CLASSES)
    })
