import sounddevice as sd
import numpy as np
import whisper
import queue
import threading
import time

# ---------------------------
# Configuration
# ---------------------------
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.5  # seconds
SILENCE_THRESHOLD = 0.01  # lower = more sensitive
MIN_SPEECH_DURATION = 1.0  # minimum duration before sending to whisper
LANGUAGE = "auto"  # or 'fr', 'de'
MODEL_NAME = "small"

# ---------------------------
# Initialize
# ---------------------------
print("Loading Whisper model (small, CPU)...")
model = whisper.load_model(MODEL_NAME)
print("✅ Model loaded successfully")

audio_queue = queue.Queue()
stop_event = threading.Event()

# ---------------------------
# Helper functions
# ---------------------------
def rms(audio):
    """Root Mean Square energy."""
    return np.sqrt(np.mean(np.square(audio)))

def record_audio():
    """Continuously capture audio from mic and push speech segments."""
    print("\n🎙️ Listening... (Ctrl+C to stop)\n")

    buffer = np.zeros(0, dtype=np.float32)
    silence_counter = 0
    speaking = False

    def callback(indata, frames, time_info, status):
        nonlocal buffer, silence_counter, speaking
        if status:
            print(status)
        audio_chunk = indata[:, 0]
        energy = rms(audio_chunk)

        if energy > SILENCE_THRESHOLD:
            speaking = True
            buffer = np.concatenate((buffer, audio_chunk))
            silence_counter = 0
        elif speaking:
            silence_counter += CHUNK_DURATION
            if silence_counter > 0.8:
                # End of speech
                if len(buffer) / SAMPLE_RATE > MIN_SPEECH_DURATION:
                    audio_queue.put(buffer.copy())
                buffer = np.zeros(0, dtype=np.float32)
                silence_counter = 0
                speaking = False

    with sd.InputStream(
        channels=1,
        samplerate=SAMPLE_RATE,
        callback=callback,
        blocksize=int(SAMPLE_RATE * CHUNK_DURATION),
    ):
        while not stop_event.is_set():
            time.sleep(0.1)

def transcriber():
    """Process audio chunks with Whisper sequentially."""
    while not stop_event.is_set():
        try:
            chunk = audio_queue.get(timeout=1)
        except queue.Empty:
            continue
        try:
            result = model.transcribe(
                chunk,
                task="translate",
                language=None if LANGUAGE == "auto" else LANGUAGE,
                fp16=False
            )
            text = result["text"].strip()
            if text:
                print(f"🗣️ English Translation: {text}\n")
        except Exception as e:
            print(f"[Error processing chunk] {e}")

# ---------------------------
# Start threads
# ---------------------------
record_thread = threading.Thread(target=record_audio, daemon=True)
process_thread = threading.Thread(target=transcriber, daemon=True)

record_thread.start()
process_thread.start()

try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\n🛑 Stopping...")
    stop_event.set()
    record_thread.join()
    process_thread.join()
    print("✅ Exited cleanly.")
