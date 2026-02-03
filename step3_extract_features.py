import librosa
import numpy as np

# load audio
audio, sample_rate = librosa.load("decoded.mp3", sr=None)

# extract MFCC features
mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sample_rate,
    n_mfcc=13
)

print("MFCC shape:", mfcc.shape)

# take mean across time axis to get fixed size
mfcc_mean = np.mean(mfcc, axis=1)

print("Final MFCC feature vector shape:", mfcc_mean.shape)
print("MFCC feature vector:")
print(mfcc_mean)
