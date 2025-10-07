import whisper
import speech_recognition as sr
import numpy as np
import threading
import queue
import time
from flask import Flask, Response

# ---------------------------
# CONFIGURATION
# ---------------------------
MODEL_NAME = "base"        # faster, CPU-friendly
LANGUAGE = "auto"          # auto detect or 'fr', 'de', etc.
PAUSE_THRESHOLD = 0.6      # seconds of silence before chunk ends
ENERGY_THRESHOLD = 300     # microphone sensitivity
PHRASE_TIMEOUT = 8         # max seconds per phrase
MAX_LINE_LENGTH = 70       # max characters per caption line
HOST = "127.0.0.1"
PORT = 5000

# ---------------------------
# INITIALIZE Whisper
# ---------------------------
print("Loading Whisper model (base, CPU)...")
model = whisper.load_model(MODEL_NAME)
print("✅ Model loaded successfully\n")

# ---------------------------
# Initialize SpeechRecognition
# ---------------------------
recognizer = sr.Recognizer()
recognizer.energy_threshold = ENERGY_THRESHOLD
recognizer.pause_threshold = PAUSE_THRESHOLD
recognizer.dynamic_energy_threshold = True

audio_queue = queue.Queue()
caption_queue = queue.Queue()
stop_event = threading.Event()

# ---------------------------
# Flask App for Browser Overlay
# ---------------------------
app = Flask(__name__)

# HTML page served directly by Flask
HTML_TEMPLATE = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Live Captions</title>
<style>
  body {{
    margin: 0;
    background-color: rgba(0,255,0,1); /* solid green */
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

@app.route('/')
def index():
    return HTML_TEMPLATE

@app.route('/stream')
def stream():
    def event_stream():
        while True:
            text = caption_queue.get()
            yield f"data: {text}\n\n"
    return Response(event_stream(), mimetype='text/event-stream')

def run_server():
    app.run(host=HOST, port=PORT, debug=False, use_reloader=False)

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
def split_text(text, max_len=MAX_LINE_LENGTH):
    """Split long sentences into smaller chunks for display."""
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

def update_caption(text):
    caption_queue.put(text)

# ---------------------------
# WORKER THREAD — Transcribe audio
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
            result = model.transcribe(
                audio_np,
                task="translate",
                language=None if LANGUAGE == "auto" else LANGUAGE,
                fp16=False
            )
            text = result["text"].strip()
            if text:
                for line in split_text(text, MAX_LINE_LENGTH):
                    print(f"🗣️ English Translation: {line}")
                    update_caption(line)
                    time.sleep(0.6)
        except Exception as e:
            print(f"[Error while transcribing] {e}")

# ---------------------------
# LISTENER THREAD — Capture mic continuously
# ---------------------------
def listen_continuously():
    with sr.Microphone(sample_rate=16000) as source:
        print("🎙️ Listening continuously... (Ctrl+C to stop)\n")
        while not stop_event.is_set():
            try:
                audio_data = recognizer.listen(
                    source,
                    timeout=None,
                    phrase_time_limit=PHRASE_TIMEOUT
                )
                audio_queue.put(audio_data)
            except Exception as e:
                print(f"[Listening error] {e}")

# ---------------------------
# START THREADS
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
    print(f"🌐 Open browser at http://{HOST}:{PORT} to see live captions")
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\n🛑 Stopping...")
    stop_event.set()
    print("✅ Exited cleanly.")
