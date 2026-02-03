import librosa
import numpy as np

# load mp3 audio
audio, sample_rate = librosa.load("decoded.mp3", sr=None)

print("Audio loaded successfully")
print("Sample rate:", sample_rate)
print("Audio type:", type(audio))
print("Audio shape:", audio.shape)
print("First 10 audio values:", audio[:10])
