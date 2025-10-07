import whisper
import speech_recognition as sr
import numpy as np
import threading
import queue
import time
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

# ---------------------------
# CONFIGURATION
# ---------------------------
MODEL_NAME = "base"  # CPU-friendly
PAUSE_THRESHOLD = 0.6
ENERGY_THRESHOLD = 300
PHRASE_TIMEOUT = 8
MAX_LINE_LENGTH = 70

# ---------------------------
# LOAD WHISPER
# ---------------------------
print("Loading Whisper model (base, CPU)...")
model = whisper.load_model(MODEL_NAME)
print("✅ Model loaded successfully")

# ---------------------------
# SPEECH RECOGNITION
# ---------------------------
recognizer = sr.Recognizer()
recognizer.energy_threshold = ENERGY_THRESHOLD
recognizer.pause_threshold = PAUSE_THRESHOLD
recognizer.dynamic_energy_threshold = True

audio_queue = queue.Queue()
stop_event = threading.Event()
transcribed_text = []

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
def split_text(text, max_len=MAX_LINE_LENGTH):
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
# WORKER THREADS
# ---------------------------
def transcribe_worker(language_option):
    global transcribed_text
    while not stop_event.is_set():
        try:
            audio_data = audio_queue.get(timeout=1)
        except queue.Empty:
            continue
        try:
            wav_bytes = audio_data.get_wav_data(convert_rate=16000)
            audio_np = np.frombuffer(wav_bytes, np.int16).astype(np.float32)/32768.0
            result = model.transcribe(
                audio_np,
                task="translate",
                language=None if language_option=="Auto-Detect" else language_option.lower(),
                fp16=False
            )
            text = result["text"].strip()
            if text:
                for line in split_text(text):
                    transcribed_text.append(line)
        except Exception as e:
            print(f"[Error while transcribing] {e}")

def listen_continuously():
    with sr.Microphone(sample_rate=16000) as source:
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
# TKINTER GUI
# ---------------------------
class WhisperApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎙️ Real-Time Whisper Translator")
        self.root.geometry("700x500")
        
        # Language dropdown
        self.lang_var = tk.StringVar(value="Auto-Detect")
        self.lang_menu = ttk.Combobox(root, textvariable=self.lang_var, state="readonly")
        self.lang_menu['values'] = ("Auto-Detect", "French", "German")
        self.lang_menu.pack(pady=10)
        
        # Buttons
        self.btn_frame = tk.Frame(root)
        self.btn_frame.pack()
        
        self.start_btn = tk.Button(self.btn_frame, text="Start Listening", command=self.start_listening)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = tk.Button(self.btn_frame, text="Stop Listening", command=self.stop_listening)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        self.clear_btn = tk.Button(self.btn_frame, text="Clear Text", command=self.clear_text)
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Scrollable text area
        self.text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Arial", 14))
        self.text_area.pack(expand=True, fill=tk.BOTH, pady=10, padx=10)
        self.text_area.insert(tk.END, "🎙️ Listening...\n")
        
        # Start GUI update loop
        self.update_text_area()
    
    def start_listening(self):
        global stop_event, listener_thread, worker_thread, transcribed_text
        stop_event.clear()
        transcribed_text = []
        self.text_area.delete('1.0', tk.END)
        self.text_area.insert(tk.END, "🎙️ Listening...\n")
        
        language_option = self.lang_var.get()
        listener_thread = threading.Thread(target=listen_continuously, daemon=True)
        worker_thread = threading.Thread(target=transcribe_worker, args=(language_option,), daemon=True)
        listener_thread.start()
        worker_thread.start()
    
    def stop_listening(self):
        stop_event.set()
    
    def clear_text(self):
        global transcribed_text
        transcribed_text = []
        self.text_area.delete('1.0', tk.END)
    
    def update_text_area(self):
        self.text_area.delete('1.0', tk.END)
        self.text_area.insert(tk.END, "\n".join(transcribed_text))
        self.text_area.see(tk.END)  # auto-scroll
        self.root.after(500, self.update_text_area)  # refresh every 0.5 sec

# ---------------------------
# RUN APP
# ---------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = WhisperApp(root)
    root.mainloop()
