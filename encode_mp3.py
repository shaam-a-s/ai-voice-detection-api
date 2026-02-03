import base64

# open the mp3 file in binary mode
with open("sample.mp3", "rb") as f:
    audio_bytes = f.read()

# convert binary to base64
audio_base64 = base64.b64encode(audio_bytes)

# convert bytes to string
audio_base64_string = audio_base64.decode("utf-8")

# write to file instead of print to avoid shell encoding issues
with open("b64.txt", "w", encoding="utf-8") as f:
    f.write(audio_base64_string)

print("b64.txt created")
