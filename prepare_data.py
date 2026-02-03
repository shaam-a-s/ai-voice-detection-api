import os
import shutil
import numpy as np
import librosa
import soundfile as sf
from gtts import gTTS

# Ensure directories exist
os.makedirs("data/human", exist_ok=True)
os.makedirs("data/ai", exist_ok=True)

source_mp3 = "sample.mp3"
if not os.path.exists(source_mp3):
    print(f"Error: {source_mp3} not found. Please ensure Step 1 is done.")
    exit(1)

print("Generating Human Samples...")
# Load source audio once
y, sr = librosa.load(source_mp3, sr=None)

# Save original as h1
shutil.copy(source_mp3, "data/human/h1.mp3")

# Generate variations for Human (h2-h5) by adding slight noise
# This simulates different recordings/environments
for i in range(2, 6):
    noise = np.random.normal(0, 0.005, y.shape)
    y_noise = y + noise
    sf.write(f"data/human/h{i}.mp3", y_noise, sr)
    print(f"Created data/human/h{i}.mp3")

print("Generating AI Samples...")
# Generate AI samples using gTTS (a1-a5)
texts = [
    "This is an artificial intelligence voice.",
    "I am a computer program generated audio.",
    "Voice synthesis is becoming very advanced.",
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models can classify audio data."
]

for i, text in enumerate(texts):
    try:
        tts = gTTS(text=text, lang='en')
        save_path = f"data/ai/a{i+1}.mp3"
        tts.save(save_path)
        print(f"Created {save_path}")
    except Exception as e:
        print(f"Failed to create AI sample {i+1} with gTTS: {e}")
        # Fallback: Tone generator if gTTS fails (network issues etc)
        duration = 3.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        tone = 0.5 * np.sin(2 * np.pi * 440 * t) # 440Hz sine wave
        sf.write(f"data/ai/a{i+1}.mp3", tone, sr)
        print(f"Created {save_path} (Fallback Tone)")

print("Data preparation complete.")
