import base64

with open("D:\new\voice_project\sample.mp3", "rb") as f:
    print(base64.b64encode(f.read()).decode())
