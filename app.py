import os
import cv2
import numpy as np
import joblib
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from skimage.feature import hog, local_binary_pattern

app = Flask(__name__)
CORS(app)  # Enable CORS to connect from React

# Load your trained model and scaler
models_dir = "models"
model_path = os.path.join(models_dir, "best_prawn_disease_model.pkl")
scaler_path = os.path.join(models_dir, "scaler.pkl")

best_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Disease names
categories = ['HEALTHY', 'BLACK GILL', 'WHITE SPOT', 'BLACK SPOT']

# Feature extraction
def extract_features(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hog_feat = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9, visualize=False)

        hist_features = []
        for i in range(3):
            hist = cv2.calcHist([image], [i], None, [32], [0, 256])
            hist_features.extend(hist.flatten())

        lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))

        return np.hstack([hog_feat, hist_features, lbp_hist])
    except:
        return None

# Affected area detection
def detect_spot(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        return max(contours, key=cv2.contourArea)
    return None

@app.route("/")
def home():
    return "Shrimp Disease Detection API is Live"

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400

    file = request.files['file']
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (128, 128))

    features = extract_features(img_resized)
    if features is None:
        return jsonify({"error": "Feature extraction failed"}), 500

    features = scaler.transform(features.reshape(1, -1))
    probs = best_model.predict_proba(features)[0]
    pred_index = np.argmax(probs)
    predicted = categories[pred_index]

    probs_dict = {categories[i]: float(probs[i]) for i in range(len(categories))}
    probs_sorted = dict(sorted(probs_dict.items(), key=lambda x: x[1], reverse=True))

    # Optional bounding box
    img_box = img.copy()
    if predicted != "HEALTHY":
        contour = detect_spot(img)
        if contour is not None:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img_box, (x, y), (x + w, y + h), (255, 0, 0), 3)

    _, buffer = cv2.imencode(".png", cv2.cvtColor(img_box, cv2.COLOR_RGB2BGR))
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        "predicted_disease": predicted,
        "probabilities": probs_sorted,
        "labeled_image_base64": img_base64
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
