from flask import Blueprint, render_template, request
import numpy as np
import pandas as pd
import pickle
import os

crop_bp = Blueprint('crop', __name__, url_prefix='/crop')

# ==========================
# Load Model and Encoder
# ==========================


# Base directory (Farmer_assistant folder)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
model_path = os.path.join(BASE_DIR, 'models', 'RandomForest_crop.pkl')
encoder_path = os.path.join(BASE_DIR, 'models', 'crop_encoder.pkl')
dataset_path = os.path.join(BASE_DIR, 'models', 'crop_recommendation.csv')

try:
    with open(model_path, "rb") as file:
        model = pickle.load(file)
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Load encoder
try:
    with open(encoder_path, "rb") as file:
        crop_encoder = pickle.load(file)
except Exception as e:
    print(f"❌ Error loading crop encoder: {e}")
    crop_encoder = None

# Load dataset
if os.path.exists(dataset_path):
    df = pd.read_csv(dataset_path)
else:
    print(f"❌ Dataset not found at {dataset_path}")
    df = None

# ==========================
# Helper Functions
# ==========================
def get_crop_features(crop_name: str):
    """Return average feature values for the given crop name."""
    if df is None:
        return None
    crop_data = df[df['label'].str.lower() == crop_name.lower()]
    if crop_data.empty:
        return None
    return crop_data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']].mean().to_dict()


def calculate_fertilizer_requirements(crop_name: str, soil_data: dict, land_area: float):
    """
    Calculate fertilizer requirements based on soil report, 
    ideal crop conditions, and land area.
    Also recommend real fertilizers (Urea, DAP, MOP, etc.).
    """
    ideal = get_crop_features(crop_name)
    if ideal is None:
        return None, None, None

    requirements = {}
    fertilizer_recommendations = {}

    # --- Nitrogen (N) ---
    n_diff = max(0, ideal['N'] - soil_data.get('N', 0))
    n_needed = n_diff * land_area
    requirements['N'] = n_needed
    if n_needed > 0:
        # Urea (46% N)
        urea_needed = n_needed / 0.46
        fertilizer_recommendations['Nitrogen'] = f"Add {urea_needed:.2f} kg Urea"

    # --- Phosphorus (P) ---
    p_diff = max(0, ideal['P'] - soil_data.get('P', 0))
    p_needed = p_diff * land_area
    requirements['P'] = p_needed
    if p_needed > 0:
        # DAP (20% P effective)
        dap_needed = p_needed / 0.20
        fertilizer_recommendations['Phosphorus'] = f"Add {dap_needed:.2f} kg DAP"

    # --- Potassium (K) ---
    k_diff = max(0, ideal['K'] - soil_data.get('K', 0))
    k_needed = k_diff * land_area
    requirements['K'] = k_needed
    if k_needed > 0:
        # MOP (50% K effective)
        mop_needed = k_needed / 0.50
        fertilizer_recommendations['Potassium'] = f"Add {mop_needed:.2f} kg MOP"

    # --- pH Adjustment ---
    current_ph = soil_data.get("ph", ideal["ph"])
    ph_suggestion = ""
    if current_ph < 6.0:
        ph_suggestion = f"Soil is acidic (pH={current_ph:.2f}). Add agricultural lime to raise pH."
        fertilizer_recommendations['pH'] = "Apply Lime (CaCO₃)"
    elif current_ph > 7.5:
        ph_suggestion = f"Soil is alkaline (pH={current_ph:.2f}). Add elemental sulfur or gypsum to lower pH."
        fertilizer_recommendations['pH'] = "Apply Gypsum or Sulfur"
    else:
        ph_suggestion = f"Soil pH ({current_ph:.2f}) is within optimal range."

    return requirements, ph_suggestion, fertilizer_recommendations

# ==========================
# Routes
# ==========================
@crop_bp.route('/', methods=['GET'])
def home():
    return render_template('index.html', prediction_text='', ideal_conditions_text='')


@crop_bp.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input data
        N = request.form['N']
        P = request.form['P']
        K = request.form['K']
        temperature = request.form['temperature']
        humidity = request.form['humidity']
        ph = request.form['ph']
        rainfall = request.form['rainfall']

        # Validate inputs
        try:
            n, p, k, temperature_f, humidity_f, ph_f, rainfall_f = map(
                float, [N, P, K, temperature, humidity, ph, rainfall]
            )
        except ValueError:
            return render_template('index.html', prediction_text="❌ Error: All inputs must be numeric.", ideal_conditions_text='')

        # Create DataFrame
        input_data = pd.DataFrame(
            [[n, p, k, temperature_f, humidity_f, ph_f, rainfall_f]],
            columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        )

        if model is None or crop_encoder is None:
            return render_template('index.html', prediction_text="❌ Error: Model or encoder not loaded correctly.", ideal_conditions_text='')

        # Get top 3 crops
        probabilities = model.predict_proba(input_data)[0]
        top_indices = np.argsort(probabilities)[::-1][:3]
        top_crops_encoded = model.classes_[top_indices]
        top_crops = crop_encoder.inverse_transform(top_crops_encoded)
        top_probs = probabilities[top_indices]

        recommended = [f"{crop} ({prob*100:.1f}%)" for crop, prob in zip(top_crops, top_probs)]
        result_text = "🌱 Recommended Crops: " + ", ".join(recommended)

        return render_template('index.html', prediction_text=result_text, ideal_conditions_text='')

    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return render_template('index.html', prediction_text="❌ Error: Something went wrong with prediction.", ideal_conditions_text='')


@crop_bp.route('/ideal_conditions', methods=['POST'])
def ideal_conditions_route():
    crop_name = request.form['crop_name']
    conditions = get_crop_features(crop_name)

    if conditions:
        result_text = (f"🌾 Ideal conditions for {crop_name.capitalize()}: "
                       f"N: {conditions['N']:.2f}, P: {conditions['P']:.2f}, K: {conditions['K']:.2f}, "
                       f"Temperature: {conditions['temperature']:.2f}°C, Humidity: {conditions['humidity']:.2f}%, "
                       f"pH: {conditions['ph']:.2f}, Rainfall: {conditions['rainfall']:.2f} mm")
    else:
        result_text = f"❌ No ideal conditions found for crop: {crop_name}"

    return render_template('index.html', prediction_text=result_text, ideal_conditions_text='')


@crop_bp.route('/fertilizer_calc', methods=['POST'])
def fertilizer_calc():
    try:
        crop_name = request.form['crop_name']
        land_area = float(request.form['land_area'])

        # Soil report inputs
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        ph = float(request.form['ph'])

        soil_report = {'N': N, 'P': P, 'K': K, 'ph': ph}

        requirements, ph_suggestion, fert_recos = calculate_fertilizer_requirements(crop_name, soil_report, land_area)

        if requirements:
            result_text = (f"🌱 Fertilizer recommendation for {crop_name.capitalize()} on {land_area} ha:<br>"
                           f"➤ Nitrogen (N) deficit: {requirements['N']:.2f} kg<br>"
                           f"➤ Phosphorus (P) deficit: {requirements['P']:.2f} kg<br>"
                           f"➤ Potassium (K) deficit: {requirements['K']:.2f} kg<br><br>"
                           f"💡 pH Suggestion: {ph_suggestion}<br><br>"
                           f"🧪 Recommended Fertilizers:<br>"
                           + "<br>".join([f"• {v}" for v in fert_recos.values()]))
        else:
            result_text = f"❌ Could not calculate requirements for {crop_name}"

        return render_template('index.html', prediction_text=result_text, ideal_conditions_text='')

    except Exception as e:
        print(f"❌ Error in fertilizer calculation: {e}")
        return render_template('index.html', prediction_text="❌ Error: Fertilizer calculation failed.", ideal_conditions_text='')