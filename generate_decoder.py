
template = """import base64

audio_base64_string = \"\"\"
{}
\"\"\"

# remove newlines/spaces if any
audio_base64_string = audio_base64_string.strip()

# convert base64 string back to bytes
audio_bytes = base64.b64decode(audio_base64_string)

# write mp3 file
with open("decoded.mp3", "wb") as f:
    f.write(audio_bytes)

print("decoded.mp3 created")
"""

with open("b64.txt", "r", encoding="utf-8") as f:
    b64_content = f.read().strip()

with open("decode_base64.py", "w", encoding="utf-8") as f:
    f.write(template.format(b64_content))

print("decode_base64.py has been generated with the base64 string embedded.")
