import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Load the model
MODEL_PATH = "liner_model.pkl"

def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None

model = load_model()

# Feature order expected by the model
FEATURE_NAMES = [
    "number of bedrooms",
    "number of bathrooms",
    "living area",
    "lot area",
    "number of floors",
    "waterfront present",
    "number of views",
    "condition of the house",
    "grade of the house",
    "Area of the house(excluding basement)",
    "Area of the basement",
    "Built Year",
    "Renovation Year",
    "lot_area_renov",
    "Number of schools nearby",
    "Distance from the airport"
]

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>House Price Predictor</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background-color: #f8f9fa; }
        .card { border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    </style>
</head>
<body>
    <div class="container py-5">
        <div class="row justify-content-center">
            <div class="col-lg-10">
                <div class="card p-4">
                    <h2 class="text-center mb-4">🏠 House Price Prediction</h2>
                    
                    {% if error %}
                        <div class="alert alert-danger">{{ error }}</div>
                    {% endif %}
                    
                    {% if prediction is not none %}
                        <div class="alert alert-success text-center">
                            <h4>Estimated House Price:</h4>
                            <h2 class="fw-bold">${{ "%.2f"|format(prediction) }}</h2>
                        </div>
                    {% endif %}

                    <form method="POST" action="/">
                        <div class="row g-3">
                            <div class="col-md-3">
                                <label class="form-label">Bedrooms</label>
                                <input type="number" step="1" name="number of bedrooms" class="form-control" required value="3">
                            </div>
                            <div class="col-md-3">
                                <label class="form-label">Bathrooms</label>
                                <input type="number" step="0.5" name="number of bathrooms" class="form-control" required value="2">
                            </div>
                            <div class="col-md-3">
                                <label class="form-label">Living Area (sq ft)</label>
                                <input type="number" step="any" name="living area" class="form-control" required value="2000">
                            </div>
                            <div class="col-md-3">
                                <label class="form-label">Lot Area (sq ft)</label>
                                <input type="number" step="any" name="lot area" class="form-control" required value="5000">
                            </div>

                            <div class="col-md-3">
                                <label class="form-label">Floors</label>
                                <input type="number" step="0.5" name="number of floors" class="form-control" required value="1">
                            </div>
                            <div class="col-md-3">
                                <label class="form-label">Waterfront Present</label>
                                <select name="waterfront present" class="form-select" required>
                                    <option value="0">No (0)</option>
                                    <option value="1">Yes (1)</option>
                                </select>
                            </div>
                            <div class="col-md-3">
                                <label class="form-label">Number of Views</label>
                                <input type="number" min="0" max="4" name="number of views" class="form-control" required value="0">
                            </div>
                            <div class="col-md-3">
                                <label class="form-label">Condition (1-5)</label>
                                <input type="number" min="1" max="5" name="condition of the house" class="form-control" required value="3">
                            </div>

                            <div class="col-md-3">
                                <label class="form-label">Grade (1-13)</label>
                                <input type="number" min="1" max="13" name="grade of the house" class="form-control" required value="7">
                            </div>
                            <div class="col-md-3">
                                <label class="form-label">Area (Excl. Basement)</label>
                                <input type="number" step="any" name="Area of the house(excluding basement)" class="form-control" required value="1500">
                            </div>
                            <div class="col-md-3">
                                <label class="form-label">Basement Area</label>
                                <input type="number" step="any" name="Area of the basement" class="form-control" required value="500">
                            </div>
                            <div class="col-md-3">
                                <label class="form-label">Built Year</label>
                                <input type="number" name="Built Year" class="form-control" required value="1995">
                            </div>

                            <div class="col-md-3">
                                <label class="form-label">Renovation Year</label>
                                <input type="number" name="Renovation Year" class="form-control" required value="0">
                            </div>
                            <div class="col-md-3">
                                <label class="form-label">Lot Area Renovated</label>
                                <input type="number" step="any" name="lot_area_renov" class="form-control" required value="5000">
                            </div>
                            <div class="col-md-3">
                                <label class="form-label">Schools Nearby</label>
                                <input type="number" name="Number of schools nearby" class="form-control" required value="2">
                            </div>
                            <div class="col-md-3">
                                <label class="form-label">Distance from Airport</label>
                                <input type="number" step="any" name="Distance from the airport" class="form-control" required value="15">
                            </div>
                        </div>

                        <button type="submit" class="btn btn-primary w-100 mt-4">Predict Price</button>
                    </form>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    if model is None:
        return render_template_string(HTML_TEMPLATE, prediction=None, error="Model file missing on server.")

    prediction = None
    error = None

    if request.method == "POST":
        try:
            features = [float(request.form[feat]) for feat in FEATURE_NAMES]
            input_data = np.array([features])
            prediction = float(model.predict(input_data)[0])
        except Exception as e:
            error = f"Error during prediction: {str(e)}"

    return render_template_string(HTML_TEMPLATE, prediction=prediction, error=error)


@app.route("/predict", methods=["POST"])
def api_predict():
    if model is None:
        return jsonify({"error": "Model file missing"}), 500

    try:
        data = request.get_json(force=True)
        features = [float(data[feat]) for feat in FEATURE_NAMES]
        input_data = np.array([features])
        prediction = float(model.predict(input_data)[0])
        return jsonify({"predicted_price": prediction})
    except KeyError as e:
        return jsonify({"error": f"Missing input parameter: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
