from flask import Flask, render_template, request, jsonify
import requests  # To make HTTP requests to the weather API
from app.crop_routes import crop_bp  # Crop recommendation module (data 2)
from app.disease_routes import disease_bp
app = Flask(__name__)


# Register blueprints for different modules
app.register_blueprint(crop_bp, url_prefix='/crop')  # URL prefix for crop-related routes
app.register_blueprint(disease_bp, url_prefix='/disease')  # URL prefix for crop-related routes

# Root route to redirect to /crop
@app.route('/')
def index():
    return render_template('main.html')  # Redirect to the 'home' route of the crop blueprint

@app.route('/weather')
def weather():
    return render_template('weather.html')

@app.route('/prices')
def prices():
    return render_template('prices.html')
# Debug: print the registered routes
print("\nRegistered Routes:")
for rule in app.url_map.iter_rules():
    print(rule)

if __name__ == '__main__':
    app.run(debug=True)
