"""
Real-time VAD-based translator (Whisper base) -> English
- CPU-only
- Uses sounddevice + webrtcvad for proper speech-chunking (pause detection)
- Prevents cutting words at boundaries by using a pre-roll buffer
- Attempts to remove duplicate overlap between consecutive chunk translations
"""

import os
import tempfile
import wave
import warnings
from collections import deque
from difflib import SequenceMatcher

import gradio as gr
import numpy as np
import sounddevice as sd
import webrtcvad
import whisper

# ------------------------
# Suppress the FP16-on-CPU warning from whisper
# ------------------------
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# ------------------------
# Config / Tuning (change these if you want different sensitivity)
# ------------------------
SAMPLE_RATE = 16000               # Whisper prefers 16k
FRAME_MS = 30                     # frame size for VAD (10/20/30ms allowed)
VAD_AGGRESSIVENESS = 2            # 0-3 (higher more aggressive at filtering non-speech)
PRE_ROLL_MS = 300                 # keep 300ms before speech start to avoid cutting words
SILENCE_LIMIT_MS = 700           # consider speech ended after 700ms of silence
MAX_SEGMENT_SEC = 30              # force chunk if speech > 30s to avoid huge segments
OVERLAP_CLEAN_WORDS = 15         # when merging, try to remove up to this many overlapping words

# ------------------------
# Globals
# ------------------------
RUNNING = False
MODEL = whisper.load_model("base", device="cpu")   # CPU only as requested
vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

# Languages mapping for UI
LANG_MAP = {"Auto": None, "French": "fr", "German": "de"}

# ------------------------
# Helpers
# ------------------------
def words_overlap_merge(prev_text: str, new_text: str) -> str:
    """
    Merge prev_text and new_text, removing overlap if the end of prev_text matches the
    beginning of new_text. Uses word-level SequenceMatcher and only accepts overlaps
    that are at the end of prev_text and start of new_text.
    """
    if not prev_text:
        return new_text
    a_words = prev_text.split()
    b_words = new_text.split()
    if not a_words or not b_words:
        return (prev_text + " " + new_text).strip()

    # Run SequenceMatcher on words
    matcher = SequenceMatcher(None, a_words, b_words)
    # Find a matching block that ends at end of a_words and starts at 0 of b_words
    best_overlap = 0
    for block in matcher.get_matching_blocks():
        # block: (a_index, b_index, size)
        if (block.a + block.size == len(a_words)) and (block.b == 0) and block.size > 0:
            # restrict to reasonable overlap length
            if block.size <= OVERLAP_CLEAN_WORDS and block.size > best_overlap:
                best_overlap = block.size

    if best_overlap > 0:
        merged = " ".join(a_words + b_words[best_overlap:])
        return merged
    else:
        # no clear overlap: just append
        return (prev_text + " " + new_text).strip()

def write_wav_from_frames(frames_bytes_list, sample_rate=SAMPLE_RATE, path=None):
    """
    frames_bytes_list: list of bytes (each frame contains int16 PCM bytes)
    Writes a mono WAV file at sample_rate and returns file path.
    """
    if path is None:
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(frames_bytes_list))
    return path

# ------------------------
# Main generator function (used by Gradio start button)
# ------------------------
def start_listening(selected_lang: str):
    """
    Generator that reads mic audio, segments using VAD, passes finished segments
    to Whisper (task='translate'), and yields cumulative English translation.
    """
    global RUNNING
    if RUNNING:
        yield "Already listening..."
        return
    RUNNING = True

    # Derived params
    frame_size = int(SAMPLE_RATE * (FRAME_MS / 1000.0))   # samples per frame
    pre_roll_frames = max(1, int(PRE_ROLL_MS / FRAME_MS))
    silence_frames_limit = max(1, int(SILENCE_LIMIT_MS / FRAME_MS))
    max_segment_frames = int((MAX_SEGMENT_SEC * 1000) / FRAME_MS)

    yield "Listening... speak now (pause to finalize each chunk)."

    # ring buffer for pre-roll
    pre_buffer = deque(maxlen=pre_roll_frames)

    # speech collection buffer
    speech_frames = []
    speech_active = False
    silence_counter = 0
    cumulative_translation = ""

    # get language code for whisper; None for autodetect
    lang_code = LANG_MAP.get(selected_lang, None)

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE,
                            channels=1,
                            dtype='int16',
                            blocksize=frame_size):
            # Using blocking reads from stream inside generator
            while RUNNING:
                try:
                    frame, overflow = sd.RawInputStream.read(sd.RawInputStream, frame_size)
                    # The above is a hacky usage; instead use the following:
                except Exception:
                    # fallback: use sd.rec + wait (less ideal). We'll instead open a proper InputStream context:
                    break
    except Exception:
        # We'll open the InputStream properly below (safer)
        pass

    # Proper InputStream (works on Windows & Linux)
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16', blocksize=frame_size) as stream:
            while RUNNING:
                try:
                    data, overflowed = stream.read(frame_size)
                except Exception as e:
                    yield f"[audio read error: {e}]"
                    break

                # convert to bytes (int16)
                frame_bytes = data.tobytes()

                is_speech = vad.is_speech(frame_bytes, SAMPLE_RATE)

                # always keep rolling pre-buffer
                pre_buffer.append(frame_bytes)

                if not speech_active:
                    if is_speech:
                        # start speech segment; pre-roll included
                        speech_active = True
                        # include pre-roll frames into speech_frames
                        speech_frames = list(pre_buffer)
                        silence_counter = 0
                    else:
                        # waiting for start; continue
                        continue
                else:
                    # we are in a speech segment -> append current frame
                    speech_frames.append(frame_bytes)

                    if not is_speech:
                        silence_counter += 1
                    else:
                        silence_counter = 0

                    # if silence lasted long enough OR segment too long -> finalize chunk
                    if silence_counter >= silence_frames_limit or len(speech_frames) >= max_segment_frames:
                        # create wav
                        tmp_wav = write_wav_from_frames(speech_frames)
                        try:
                            # transcribe + translate to English
                            if lang_code is None:
                                res = MODEL.transcribe(tmp_wav, task="translate")
                            else:
                                res = MODEL.transcribe(tmp_wav, task="translate", language=lang_code)

                            chunk_text = (res.get("text") or "").strip()
                        except Exception as e:
                            chunk_text = f"[whisper error: {e}]"
                        finally:
                            # cleanup
                            try:
                                os.remove(tmp_wav)
                            except Exception:
                                pass

                        if chunk_text:
                            # merge with cumulative removing overlaps if possible
                            new_cum = words_overlap_merge(cumulative_translation, chunk_text)
                            cumulative_translation = new_cum
                            yield cumulative_translation
                        else:
                            # yield current cumulative even if empty or no text
                            yield cumulative_translation or "[no speech recognized in chunk]"

                        # prepare for next segment:
                        # keep last few frames as pre-roll to avoid missing trailing words between chunks
                        last_frames_for_preroll = speech_frames[-pre_roll_frames:] if len(speech_frames) >= pre_roll_frames else speech_frames
                        pre_buffer = deque(last_frames_for_preroll, maxlen=pre_roll_frames)
                        speech_frames = []
                        speech_active = False
                        silence_counter = 0

            # End while RUNNING
            yield cumulative_translation or "Stopped (no speech captured)."
    except Exception as e:
        yield f"[microphone open error: {e}]"
    finally:
        RUNNING = False

# ------------------------
# Stop button function
# ------------------------
def stop_listening():
    global RUNNING
    if not RUNNING:
        return "Not running."
    RUNNING = False
    return "Stopping..."

# ------------------------
# Gradio UI
# ------------------------
with gr.Blocks() as demo:
    gr.Markdown("## Real-time VAD → Whisper (base) → English\nPress **Start** to begin listening. The app uses VAD to split speech on pauses and translates each chunk to English.")
    with gr.Row():
        lang = gr.Dropdown(list(LANG_MAP.keys()), value="Auto", label="Input language (Auto/French/German)")
        start_btn = gr.Button("🔴 Start Listening")
        stop_btn = gr.Button("⏹ Stop")
    output = gr.Textbox(label="English translation (live)", lines=10)
    start_btn.click(fn=start_listening, inputs=[lang], outputs=[output])
    stop_btn.click(fn=stop_listening, outputs=[output])

if __name__ == "__main__":
    demo.launch()
