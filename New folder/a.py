import whisper
import speech_recognition as sr
import numpy as np
import threading
import queue
import time
import os

# ---------------------------
# CONFIGURATION
# ---------------------------
MODEL_NAME = "base"  # faster model
LANGUAGE = "auto"
PAUSE_THRESHOLD = 0.6
ENERGY_THRESHOLD = 300
PHRASE_TIMEOUT = 8
HTML_FILE = "captions.html"
MAX_LINE_LENGTH = 70  # limit caption width

# ---------------------------
# INITIALIZE
# ---------------------------
print("Loading Whisper model (base, CPU)...")
model = whisper.load_model(MODEL_NAME)
print("✅ Model loaded successfully\n")

recognizer = sr.Recognizer()
recognizer.energy_threshold = ENERGY_THRESHOLD
recognizer.pause_threshold = PAUSE_THRESHOLD
recognizer.dynamic_energy_threshold = True

audio_queue = queue.Queue()
stop_event = threading.Event()

# ---------------------------
# HTML TEMPLATE
# ---------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Live Captions</title>
<style>
  body {{
    margin: 0;
    background-color: rgba(0, 255, 0, 1); /* solid green background */
    color: white;
    font-family: Arial, Helvetica, sans-serif;
    font-size: 2em;
    text-align: center;
    display: flex;
    justify-content: center;
    align-items: flex-end;
    height: 100vh;
    padding: 20px;
  }}
  #caption {{
    width: 90%;
    line-height: 1.4;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.6);
  }}
</style>
</head>
<body>
<div id="caption">🎙️ Listening...</div>

<script>
const eventSource = new EventSource('/stream');
eventSource.onmessage = function(event) {{
  document.getElementById('caption').innerText = event.data;
}};
</script>
</body>
</html>
"""

# ---------------------------
# CREATE INITIAL HTML
# ---------------------------
with open(HTML_FILE, "w", encoding="utf-8") as f:
    f.write(HTML_TEMPLATE)
print(f"🌐 Caption page created: {HTML_FILE}\n(Open it in your browser)")

# ---------------------------
# SIMPLE SERVER FOR LIVE UPDATES
# ---------------------------
from flask import Flask, Response

app = Flask(__name__)
caption_queue = queue.Queue()

@app.route('/stream')
def stream():
    def event_stream():
        while True:
            text = caption_queue.get()
            yield f"data: {text}\n\n"
    return Response(event_stream(), mimetype="text/event-stream")

def run_server():
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)

# ---------------------------
# UPDATE CAPTION FUNCTION
# ---------------------------
def update_caption(text):
    caption_queue.put(text)

def split_text(text, max_len=70):
    lines = []
    while len(text) > max_len:
        split_idx = text.rfind(' ', 0, max_len)
        if split_idx == -1:
            split_idx = max_len
        lines.append(text[:split_idx].strip())
        text = text[split_idx:].strip()
    if text:
        lines.append(text)
    return lines

# ---------------------------
# WORKER — Transcribe
# ---------------------------
def transcribe_worker():
    while not stop_event.is_set():
        try:
            audio_data = audio_queue.get(timeout=1)
        except queue.Empty:
            continue
        try:
            wav_bytes = audio_data.get_wav_data(convert_rate=16000)
            audio_np = np.frombuffer(wav_bytes, np.int16).astype(np.float32) / 32768.0
            result = model.transcribe(audio_np, task="translate", fp16=False)
            text = result["text"].strip()
            if text:
                for line in split_text(text, MAX_LINE_LENGTH):
                    print(f"🗣️ English Translation: {line}")
                    update_caption(line)
                    time.sleep(0.6)
        except Exception as e:
            print(f"[Error while transcribing] {e}")

# ---------------------------
# LISTENER — Capture mic
# ---------------------------
def listen_continuously():
    with sr.Microphone(sample_rate=16000) as source:
        print("🎙️ Listening continuously... (Ctrl+C to stop)\n")
        while not stop_event.is_set():
            try:
                audio_data = recognizer.listen(source, timeout=None, phrase_time_limit=PHRASE_TIMEOUT)
                audio_queue.put(audio_data)
            except Exception as e:
                print(f"[Listening error] {e}")

# ---------------------------
# THREADS
# ---------------------------
server_thread = threading.Thread(target=run_server, daemon=True)
listener_thread = threading.Thread(target=listen_continuously, daemon=True)
worker_thread = threading.Thread(target=transcribe_worker, daemon=True)

server_thread.start()
listener_thread.start()
worker_thread.start()

# ---------------------------
# MAIN LOOP
# ---------------------------
try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\n🛑 Stopping...")
    stop_event.set()
    print("✅ Exited cleanly.")
