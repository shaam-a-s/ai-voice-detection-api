import os
import librosa
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def extract_robust_features(audio, sr):
    # 1. MFCCs (13)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    
    # 2. Delta MFCCs (13) - represents rate of change
    delta_mfcc = librosa.feature.delta(mfcc)
    delta_mean = np.mean(delta_mfcc, axis=1)
    
    # 3. Delta-Delta MFCCs (13) - represents acceleration
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    delta2_mean = np.mean(delta2_mfcc, axis=1)
    
    # 4. Spectral Features (2)
    # Centroid: "brightness" of sound
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    centroid_mean = np.mean(spectral_centroid)
    
    # Flatness: how noise-like vs tonal the sound is
    spectral_flatness = librosa.feature.spectral_flatness(y=audio)
    flatness_mean = np.mean(spectral_flatness)
    
    # Concatenate all features (Total dimensions: 13+13+13+1+1 = 41)
    features = np.concatenate([
        mfcc_mean, 
        delta_mean, 
        delta2_mean, 
        [centroid_mean], 
        [flatness_mean]
    ])
    
    return features

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        if len(audio) == 0:
            return np.zeros(41)
        return extract_robust_features(audio, sr)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return np.zeros(41)

X = []
y = []

print("Loading Human samples...")
# load human samples
human_dir = "data/human"
if os.path.exists(human_dir):
    for file in os.listdir(human_dir):
        if file.endswith(".mp3"):
            features = extract_features(os.path.join(human_dir, file))
            X.append(features)
            y.append(0)  # 0 = HUMAN
else:
    print("Warning: data/human directory missing!")

print("Loading AI samples...")
# load AI samples
ai_dir = "data/ai"
if os.path.exists(ai_dir):
    for file in os.listdir(ai_dir):
        if file.endswith(".mp3"):
            features = extract_features(os.path.join(ai_dir, file))
            X.append(features)
            y.append(1)  # 1 = AI_GENERATED
else:
    print("Warning: data/ai directory missing!")

X = np.array(X)
y = np.array(y)

print(f"Total samples: {len(X)}")
if len(X) < 2:
    print("Not enough samples to train. Please ensure data generation worked.")
    exit(1)

# train-test split
# Handle small dataset size for split
test_size = 0.2
if len(X) < 5:
    test_size = 0.5 # Force split if tiny

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y if len(np.unique(y)) > 1 and len(y) > 4 else None
)

print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples (Feature dim: {X.shape[1]})...")

# --- Ensemble Training ---

# Model 1: Logistic Regression (Good for linear separation)
print("Training Logistic Regression...")
model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train, y_train)

# Model 2: Random Forest (Good for non-linear patterns)
print("Training Random Forest...")
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

# Evaluation (Soft Voting Manual Implementation for transparency)
def get_ensemble_prediction(data_X):
    # Average probabilities
    prob_lr = model_lr.predict_proba(data_X)[:, 1] # Probability of AI (class 1)
    prob_rf = model_rf.predict_proba(data_X)[:, 1]
    
    avg_prob = (prob_lr + prob_rf) / 2.0
    return avg_prob

if len(X_test) > 0:
    ensemble_probs = get_ensemble_prediction(X_test)
    y_pred_ensemble = [1 if p > 0.5 else 0 for p in ensemble_probs]
    accuracy = accuracy_score(y_test, y_pred_ensemble)
    print("Ensemble Model accuracy:", accuracy)
else:
    print("Skipping accuracy check (no test samples)")

# Save both models in a dictionary
ensemble_data = {
    'lr': model_lr,
    'rf': model_rf,
    'description': 'Ensemble (Voting) Classifier with Robust Features'
}

joblib.dump(ensemble_data, "voice_classifier_ensemble.pkl")
print("Models saved as voice_classifier_ensemble.pkl")
