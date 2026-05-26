# crop_routes.py
from flask import Blueprint, render_template

# import all crop blueprints
from crop_apps.rice_blueprint import rice_bp
from crop_apps.potato_blueprint import potato_bp
from crop_apps.wheat_blueprint import wheat_bp
from crop_apps.sugarcane_blueprint import sugarcane_bp
from crop_apps.tomato_blueprint import tomato_bp
from crop_apps.corn_blueprint import corn_bp

# ✅ Blueprint banao (Flask nahi)
disease_bp = Blueprint('disease', __name__)

# ✅ Blueprints register karo (yahin karna hai)
disease_bp.register_blueprint(rice_bp)
disease_bp.register_blueprint(potato_bp)
disease_bp.register_blueprint(wheat_bp)
disease_bp.register_blueprint(sugarcane_bp)
disease_bp.register_blueprint(tomato_bp)
disease_bp.register_blueprint(corn_bp)

# ✅ dashboard route
@disease_bp.route('/')
def index():
    crops = {
        "rice": {"name": "Rice", "url": "/disease/rice"},
        "potato": {"name": "Potato", "url": "/disease/potato"},
        "wheat": {"name": "Wheat", "url": "/disease/wheat"},
        "sugarcane": {"name": "Sugarcane", "url": "/disease/sugarcane"},
        "tomato": {"name": "Tomato", "url": "/disease/tomato"},
        "corn": {"name": "Corn", "url": "/disease/corn"},
    }
    return render_template("main_index.html", crops=crops)

