from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import librosa
from joblib import load
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load models
age_model = load_model(os.path.join(BASE_DIR, "../models/age_from_voice 89.keras"))
gender_model = load_model(os.path.join(BASE_DIR, "../models/gender_from_voice 96.keras"))

# Encoders 
encoder_age = LabelEncoder()
encoder_gender = LabelEncoder()
encoder_age.classes_ = np.array(['twenties', 'seventies', 'thirties', 'sixties', 'fifties',
       'fourties', 'teens', 'eighties'])
encoder_gender.classes_ = np.array(['male', 'female','other'])

# -----------------------------
# Audio Feature extraction
# -----------------------------
def audio_feature_extraction(filepath, sampling_rate=48000):
    features = []
    audio, _ = librosa.load(filepath, sr=sampling_rate)

    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sampling_rate))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sampling_rate))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sampling_rate))
    features += [spectral_centroid, spectral_bandwidth, spectral_rolloff]

    mfcc = librosa.feature.mfcc(y=audio, sr=sampling_rate)
    features += [np.mean(m) for m in mfcc]

    return np.array(features).reshape(1, -1)

# -----------------------------
# Scaling (single input version)
# -----------------------------
def scale_features(features, scaler):
    return scaler.transform(features)

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:    
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(filepath)

        # Extract features
        features = audio_feature_extraction(filepath)

        # Load pre-fitted scaler 
        scaler = load(os.path.join(BASE_DIR, "../scaler.pkl"))
        scaled_features = scaler.transform(features)  

        # Predict
        pred_age = age_model.predict(scaled_features, verbose=0)
        pred_gender = gender_model.predict(scaled_features, verbose=0)

        age_label = encoder_age.inverse_transform([np.argmax(pred_age)])
        gender_label = encoder_gender.inverse_transform([np.argmax(pred_gender)])
    
        os.remove(filepath)

        return jsonify({
            "age_prediction": age_label[0],
            "gender_prediction": gender_label[0],
            "age_confidence": float(np.max(pred_age)),
            "gender_confidence": float(np.max(pred_gender))
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
