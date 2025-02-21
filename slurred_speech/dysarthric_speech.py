import os
import numpy as np
import pandas as pd
import librosa
import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

def extract_features(y, sr):
    """
    Given an audio signal y with sampling rate sr,
    extract a comprehensive set of acoustic features and return them as a pandas DataFrame.
    """
    features = {}
    
    # Ensure a minimum duration (100 ms)
    min_duration = 0.1  # seconds
    if len(y) / sr < min_duration:
        y = librosa.util.fix_length(y, size=int(min_duration * sr))
    
    # 1. MFCC Features (13 coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=512)
    for i, coeff in enumerate(mfccs):
        features[f'mfcc_{i}'] = np.mean(coeff)
    
    # 2. Delta and Delta-Delta MFCCs
    if mfccs.shape[1] >= 3:
        delta_mfccs = librosa.feature.delta(mfccs, mode='nearest')
        delta2_mfccs = librosa.feature.delta(mfccs, order=2, mode='nearest')
        for i, coeff in enumerate(delta_mfccs):
            features[f'delta_mfcc_{i}'] = np.mean(coeff)
        for i, coeff in enumerate(delta2_mfccs):
            features[f'delta2_mfcc_{i}'] = np.mean(coeff)
    else:
        for i in range(13):
            features[f'delta_mfcc_{i}'] = 0
            features[f'delta2_mfcc_{i}'] = 0

    # 3. Pitch Features using librosa.piptrack
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitches = pitches[pitches > 0]
    if len(pitches) > 0:
        features['pitch_mean'] = np.mean(pitches)
        features['pitch_std'] = np.std(pitches)
        # Simplified pitch period feature (for example, its mean)
        pitch_periods = 1 / pitches
        features['ppe'] = np.mean(pitch_periods)
    else:
        features['pitch_mean'] = 0
        features['pitch_std'] = 0
        features['ppe'] = 0

    # 4. Voice Quality Features using parselmouth (if installed)
    try:
        import parselmouth
        sound = parselmouth.Sound(y, sr)
        point_process = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", 75, 600)
        jitter = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer = parselmouth.praat.call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        hnr = parselmouth.praat.call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        features['jitter'] = jitter
        features['shimmer'] = shimmer
        features['hnr_mean'] = np.mean(hnr.values)
    except Exception as e:
        features['jitter'] = 0
        features['shimmer'] = 0
        features['hnr_mean'] = 0

    # 5. Temporal Features: Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features['speech_rate'] = np.mean(zcr) * sr

    # 6. Pause Characteristics
    non_silent_intervals = librosa.effects.split(y, top_db=20)
    features['pause_count'] = max(0, len(non_silent_intervals) - 1)
    if features['pause_count'] > 0:
        pauses = [non_silent_intervals[i][0] - non_silent_intervals[i-1][1] 
                  for i in range(1, len(non_silent_intervals))]
        features['pause_duration_mean'] = np.mean(pauses) / sr
    else:
        features['pause_duration_mean'] = 0

    # 7. Energy-based Features: RMS Energy
    rms = librosa.feature.rms(y=y)
    features['rms_mean'] = np.mean(rms)
    features['rms_std'] = np.std(rms)

    return pd.DataFrame([features])

def load_model():
    """
    Loads the pre-trained model from 'LGBM_CPU_final.pkl'.
    The file is expected to be a pickle file containing a dictionary with keys 'scaler' and 'model'.
    """
    with open('LGBM_CPU_final.pkl', 'rb') as f:
        model_data = pickle.load(f)
    return model_data

def predict_features(features, model_data):
    """
    Uses the scaler and the pre-trained model to predict the label and probabilities.
    """
    scaled_features = model_data['scaler'].transform(features)
    prediction = model_data['model'].predict(scaled_features)
    probability = model_data['model'].predict_proba(scaled_features)
    return prediction, probability

@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects a POST request with form-data containing an audio file with key 'audio'.
    Processes the audio file, extracts features, and returns a JSON prediction.
    """
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided.'}), 400
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'Empty filename provided.'}), 400

    # Save uploaded file temporarily
    temp_filename = "temp_audio.wav"
    file.save(temp_filename)

    try:
        # Load audio and extract features
        y, sr = librosa.load(temp_filename, sr=None)
        features = extract_features(y, sr)
        model_data = load_model()
        prediction, probability = predict_features(features, model_data)
    except Exception as e:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        return jsonify({'error': str(e)}), 500

    os.remove(temp_filename)

    # Return the result as JSON. Here, label 1 indicates dysarthric/slurred speech.
    result = {
        'dysarthric': bool(prediction[0]),
        'probability': probability[0].tolist()  # returns probability for both classes
    }
    return jsonify(result)

if __name__ == '__main__':
    # Run the Flask API on port 5000, accessible on your network.
    app.run(host='0.0.0.0', port=5002, debug=True)
