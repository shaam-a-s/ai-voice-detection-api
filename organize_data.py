import os
import shutil
import glob
from pydub import AudioSegment
import soundfile as sf
import librosa
import random
from tqdm import tqdm
try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
except ImportError:
    print("static-ffmpeg not found, relying on system ffmpeg.")

# --- CONFIGURATION ---
ASVSPOOF_ROOT = r"C:\Users\shaam\Downloads\LA\LA"
ASVSPOOF_PROTOCOLS = os.path.join(ASVSPOOF_ROOT, "ASVspoof2019_LA_cm_protocols")
ASVSPOOF_TRAIN_FLAC = os.path.join(ASVSPOOF_ROOT, "ASVspoof2019_LA_train", "flac")
ASVSPOOF_DEV_FLAC = os.path.join(ASVSPOOF_ROOT, "ASVspoof2019_LA_dev", "flac")

CV_ROOT = r"C:\Users\shaam\Downloads\archive (1)"

DEST_HUMAN = r"D:\new\voice_project\data\human"
DEST_AI = r"D:\new\voice_project\data\ai"

MAX_FILES_PER_CLASS = 1500  # Cap to avoid taking hours to train
# ---------------------

def ensure_mp3_and_save(src_path, dest_path):
    """
    Reads audio from src_path.
    Saves as MP3 to dest_path.
    If ffmpeg missing, saves as WAV but names it .mp3 (hack).
    """
    try:
        # Try pydub (requires ffmpeg)
        audio = AudioSegment.from_file(src_path)
        audio.export(dest_path, format="mp3")
        return True
    except Exception as e:
        # Fallback REMOVED as per strict MP3 requirement.
        # If conversion fails, we skip the file.
        print(f"Failed to convert {src_path} (ffmpeg missing?): {e}")
        return False

def process_asvspoof():
    print("Processing ASVSpoof...")
    
    # Files to process
    # Format: [ (filename, label, source_dir), ... ]
    queue = []

    # 1. Train Protocol
    proto_train = os.path.join(ASVSPOOF_PROTOCOLS, "ASVspoof2019.LA.cm.train.trn.txt")
    if os.path.exists(proto_train):
        with open(proto_train, 'r') as f:
            for line in f:
                parts = line.strip().split(' ')
                # format: SPEAKER FILE - - LABEL
                filename = parts[1]
                label = parts[4] # bonafide or spoof
                queue.append( (filename, label, ASVSPOOF_TRAIN_FLAC) )
    
    # 2. Dev Protocol
    proto_dev = os.path.join(ASVSPOOF_PROTOCOLS, "ASVspoof2019.LA.cm.dev.trl.txt")
    if os.path.exists(proto_dev):
        with open(proto_dev, 'r') as f:
            for line in f:
                parts = line.strip().split(' ')
                filename = parts[1]
                label = parts[4]
                queue.append( (filename, label, ASVSPOOF_DEV_FLAC) )

    print(f"Found {len(queue)} ASVSpoof entries.")
    
    random.shuffle(queue)
    
    count_human = 0
    count_ai = 0
    
    for filename, label, folder in tqdm(queue, desc="ASVSpoof"):
        if label == "bonafide":
            if count_human >= MAX_FILES_PER_CLASS // 2: continue # Reserve space for CommonVoice
            target_dir = DEST_HUMAN
            prefix = "asv_human_"
            count_human += 1
        elif label == "spoof":
            if count_ai >= MAX_FILES_PER_CLASS: continue
            target_dir = DEST_AI
            prefix = "asv_ai_"
            count_ai += 1
        else:
            continue
            
        src_file = os.path.join(folder, filename + ".flac")
        if not os.path.exists(src_file):
            continue
            
        dest_file = os.path.join(target_dir, f"{prefix}{filename}.mp3")
        ensure_mp3_and_save(src_file, dest_file)

    print(f"Processed {count_human} Human and {count_ai} AI files from ASVSpoof.")

def process_common_voice():
    print("Processing Common Voice...")
    
    mp3_files = []
    # Recursively find mp3s
    for root, dirs, files in os.walk(CV_ROOT):
        # Skip weird folders if any
        if "cv-valid-train" in root or "cv-valid-dev" in root or "cv-valid-test" in root:
            for f in files:
                if f.endswith(".mp3"):
                    mp3_files.append(os.path.join(root, f))
    
    print(f"Found {len(mp3_files)} Common Voice files.")
    random.shuffle(mp3_files)
    
    count = 0
    # Check how many humans we already have
    existing_humans = len(os.listdir(DEST_HUMAN))
    needed = MAX_FILES_PER_CLASS - existing_humans
    
    if needed <= 0:
        print("Human quota already filled.")
        return

    for src in tqdm(mp3_files[:needed], desc="CommonVoice"):
        fname = os.path.basename(src)
        dest = os.path.join(DEST_HUMAN, f"cv_{fname}")
        shutil.copy(src, dest)
        count += 1

    print(f"Added {count} Common Voice files.")

def main():
    # Clear existing data? The prompt says "Integrate... and retrain". 
    # Usually implies adding to, or replacing?
    # "Integrate ... into existing data folders".
    # I will KEEP the few samples generated in step 4 (h1..h5) as they are minimal.
    
    process_asvspoof()
    process_common_voice()
    
    print("Data organization complete.")

if __name__ == "__main__":
    main()
