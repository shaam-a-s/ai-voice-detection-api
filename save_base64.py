import base64

INPUT_MP3 = "test1.mp3"       # make sure this file exists here
OUTPUT_TXT = "audio_base641.txt"

with open(INPUT_MP3, "rb") as f:
    encoded = base64.b64encode(f.read()).decode("utf-8")

with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
    f.write(encoded)

print("Base64 saved to audio_base64.txt")
