import threading
import queue
import time
import numpy as np
import whisper
import speech_recognition as sr

# ---------------------------
# Configuration
# ---------------------------
MODEL_NAME = "small"         # Whisper model
LANGUAGE = "auto"            # 'auto', 'fr', 'de', etc.
PAUSE_THRESHOLD = 0.8        # silence detection (seconds)
ENERGY_THRESHOLD = 300       # microphone sensitivity
MAX_PHRASE_TIME = 8          # seconds max per phrase

# ---------------------------
# Initialization
# ---------------------------
print("Loading Whisper model (small, CPU)...")
model = whisper.load_model(MODEL_NAME)
print("✅ Model loaded successfully")

recognizer = sr.Recognizer()
recognizer.energy_threshold = ENERGY_THRESHOLD
recognizer.pause_threshold = PAUSE_THRESHOLD
recognizer.dynamic_energy_threshold = True

audio_queue = queue.Queue()
stop_event = threading.Event()

# ---------------------------
# Worker thread for processing
# ---------------------------
def transcriber_worker():
    while not stop_event.is_set():
        try:
            audio_data = audio_queue.get(timeout=1)
        except queue.Empty:
            continue

        try:
            wav_bytes = audio_data.get_wav_data()
            audio_array = np.frombuffer(wav_bytes, np.int16).astype(np.float32) / 32768.0

            # Translate using Whisper
            result = model.transcribe(
                audio_array,
                task="translate",
                language=None if LANGUAGE == "auto" else LANGUAGE,
                fp16=False
            )
            text = result["text"].strip()
            if text:
                print(f"\n🗣️  English Translation: {text}\n")
        except Exception as e:
            print(f"[Error while processing chunk] {e}")

# ---------------------------
# Continuous listening function
# ---------------------------
def listen_continuously():
    with sr.Microphone(sample_rate=16000) as source:
        print("\n🎙️  Listening continuously... (Ctrl+C to stop)\n")
        while not stop_event.is_set():
            try:
                audio_data = recognizer.listen(source, phrase_time_limit=MAX_PHRASE_TIME)
                audio_queue.put(audio_data)
            except Exception as e:
                print(f"[Listening Error] {e}")

# ---------------------------
# Start threads
# ---------------------------
listener_thread = threading.Thread(target=listen_continuously, daemon=True)
worker_thread = threading.Thread(target=transcriber_worker, daemon=True)

listener_thread.start()
worker_thread.start()

# ---------------------------
# Keep main thread alive
# ---------------------------
try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\n🛑 Stopping...")
    stop_event.set()
    listener_thread.join()
    worker_thread.join()
    print("✅ Exited cleanly.")
