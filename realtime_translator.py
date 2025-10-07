"""
realtime_translator.py
----------------------

This module implements a simple demonstration of real‑time speech
translation using OpenAI’s Whisper model via the `faster‑whisper`
library.  The program opens a microphone stream, chunks the audio
into fixed‑length segments and translates those segments into
English on the fly.  The translated text is streamed back to the
caller.  A Gradio user interface is provided to make the program
easy to test from a browser, although the core logic is written
without any dependency on Gradio.

The pipeline works as follows:

1. A `sounddevice.InputStream` captures audio from the default
   microphone at 16 kHz mono.  Each callback from the stream
   enqueues a chunk of raw audio into a thread‑safe queue.

2. A worker thread reads audio samples from the queue and
   concatenates them into a buffer.  When the buffer has at
   least `chunk_duration` seconds of audio, the oldest chunk is
   popped from the buffer, converted to a numpy array of floats
   and passed to the Whisper model.  The model translates the
   speech to English text using `task='translate'`.  If a
   specific language is selected by the user the code sets the
   `language` option on the model; otherwise Whisper will
   automatically detect the language.

3. The output segments from Whisper are concatenated and
   appended to the global transcript, which is yielded back to
   the Gradio interface.  The transcript grows as the user
   continues to speak.

This example is intentionally minimal: it avoids more complex
strategies such as voice activity detection or local agreement
policies (as used in the `whisper_streaming` project).  Those
techniques can reduce latency and avoid duplicate text but
require more sophisticated state management.  For many small
projects, simply chunking the audio into regular intervals as
demonstrated here provides a reasonable balance between
latency and simplicity.

To run this script interactively from the command line without
Gradio, install `faster‑whisper` and `sounddevice`:

    pip install faster‑whisper sounddevice gradio

Then run:

    python realtime_translator.py

When the web page opens, choose a language (Auto, French or
German), click **Start Listening**, and begin speaking into
your microphone.  The translated English text will appear in
real time.  Click **Stop** to end the session.
"""

import argparse
import queue
import threading
import time
from typing import Iterator, Optional

import numpy as np
import sounddevice as sd

try:
    # Try to import faster‑whisper.  If it is not available
    # users will see an ImportError with a helpful message.
    from faster_whisper import WhisperModel
except ImportError as exc:
    raise ImportError(
        "This script requires the 'faster-whisper' package.\n"
        "Install it with: pip install faster-whisper\n"
    ) from exc

try:
    import gradio as gr
except ImportError:
    gr = None  # type: ignore


class RealTimeTranslator:
    """Class encapsulating the real‑time translation pipeline."""

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        sample_rate: int = 16_000,
        chunk_duration: float = 3.0,
    ) -> None:
        """
        Initialise the translator.

        Parameters
        ----------
        model_size : str
            Size of the Whisper model to load (e.g. 'base', 'small').
        device : str
            Device on which to run the model ('cpu' or 'cuda').
        compute_type : str
            Precision type used by faster‑whisper (e.g. 'int8').  This
            parameter only has an effect on GPU inference.
        sample_rate : int
            Desired sampling rate for audio capture.
        chunk_duration : float
            Number of seconds of audio to accumulate before running
            inference.  Smaller values reduce latency at the cost of
            slightly lower accuracy and more frequent calls to the
            model.
        """
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        # Thread‑safe queue for incoming audio samples
        self.audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()
        # Buffer for accumulating audio before transcription
        self._buffer = np.zeros((0,), dtype=np.float32)
        # Current transcript
        self.transcript = ""
        # Event to signal stop listening
        self._stop_event = threading.Event()
        # Lock to protect transcript
        self._lock = threading.Lock()
        # Load the Whisper model once during initialisation.  Using
        # faster‑whisper provides a significant speedup over the
        # reference implementation and supports running on both CPU
        # and GPU.  According to the Modal blog, pairing faster‑whisper
        # with streaming techniques provides lower latency for
        # interactive applications【180521563297583†L100-L112】.
        self.model = WhisperModel(
            model_size, device=device, compute_type=compute_type
        )

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """Callback called by sounddevice for each captured audio block."""
        # Flatten the audio to mono and convert to float32
        data = indata.copy().flatten().astype(np.float32)
        self.audio_queue.put(data)

    def _process_audio(self, language: Optional[str], task: str = "translate") -> Iterator[str]:
        """
        Generator that reads from the audio queue, runs Whisper on
        complete chunks and yields partial transcripts.

        Parameters
        ----------
        language : Optional[str]
            Language code to force Whisper to interpret the input as.
            Use ``None`` for automatic language detection.
        task : str
            Whisper task.  Use 'transcribe' to keep the original
            language or 'translate' to translate to English.  The
            default 'translate' will always output English.

        Yields
        ------
        str
            The updated transcript after each processed chunk.
        """
        while not self._stop_event.is_set():
            try:
                # Wait for up to 0.1 seconds for new audio
                chunk = self.audio_queue.get(timeout=0.1)
                # Append the chunk to our buffer
                self._buffer = np.concatenate((self._buffer, chunk))
                # If we have enough audio for a chunk, run transcription
                target_samples = int(self.chunk_duration * self.sample_rate)
                while len(self._buffer) >= target_samples:
                    # Extract the oldest chunk
                    audio_chunk = self._buffer[:target_samples]
                    # Remove processed chunk from buffer
                    self._buffer = self._buffer[target_samples:]
                    # Run inference.  Setting `language` to None
                    # triggers Whisper’s language detection, whereas
                    # specifying 'fr' or 'de' forces decoding as a
                    # particular language.  Whisper’s architecture
                    # splits incoming audio into 30‑second chunks and
                    # uses an encoder–decoder transformer to produce
                    # text with special tokens directing tasks such as
                    # translation【510332163869086†L124-L141】.  We
                    # override this behaviour to run on smaller
                    # segments for real‑time responsiveness.
                    segments, _ = self.model.transcribe(
                        audio_chunk,
                        task=task,
                        language=language,
                        beam_size=5,
                        vad_filter=False,
                    )
                    # Append the text from each segment to the
                    # transcript.  Acquire a lock to ensure thread
                    # safety if other threads access the transcript.
                    with self._lock:
                        for seg in segments:
                            self.transcript += seg.text.strip() + " "
                        # Yield the updated transcript to the caller
                        yield self.transcript.strip()
            except queue.Empty:
                continue
        # Signal end of stream
        with self._lock:
            yield self.transcript.strip()

    def start_stream(self, language_code: Optional[str]) -> Iterator[str]:
        """
        Start capturing audio from the microphone and return a
        generator of translated text.

        Parameters
        ----------
        language_code : Optional[str]
            None for auto‑detect, otherwise language code such as
            'fr' for French or 'de' for German.

        Returns
        -------
        Iterator[str]
            A generator that yields the running transcript.
        """
        # Reset state
        self.transcript = ""
        self._buffer = np.zeros((0,), dtype=np.float32)
        self._stop_event.clear()
        # Create the microphone stream.  We disable the blocksize so
        # sounddevice chooses a suitable buffer size.  The stream is
        # started as soon as we enter the context manager.
        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self._audio_callback,
        )
        stream.start()
        try:
            # Return the generator for processing audio
            return self._process_audio(language_code, task="translate")
        except Exception:
            stream.stop()
            stream.close()
            raise

    def stop(self) -> None:
        """Signal the translator to stop listening."""
        self._stop_event.set()


def launch_gradio() -> None:
    """Launch a Gradio demo for the real‑time translator."""
    if gr is None:
        raise ImportError(
            "gradio is not installed.  Install it with: pip install gradio"
        )
    # Instantiate the translator.  Use base model and run on CPU.
    translator = RealTimeTranslator(model_size="base", device="cpu")

    def start_translation(language: str) -> Iterator[str]:
        """
        Callback used by Gradio when the user presses Start Listening.

        Parameters
        ----------
        language : str
            Selected language from the dropdown.  'Auto' triggers
            Whisper’s language detection.  Otherwise use the ISO
            639‑1 code.

        Returns
        -------
        Iterator[str]
            Streaming generator of translated text.
        """
        # Map UI selection to language code
        lang_code: Optional[str]
        if language == "Auto":
            lang_code = None
        elif language == "French (fr)":
            lang_code = "fr"
        elif language == "German (de)":
            lang_code = "de"
        else:
            lang_code = None
        return translator.start_stream(lang_code)

    def stop_translation() -> str:
        """Stop the translation and return the final transcript."""
        translator.stop()
        # Return whatever transcript has been captured
        return translator.transcript.strip()

    with gr.Blocks(title="Real‑Time Speech Translation to English") as demo:
        gr.Markdown(
            """
            # Real‑Time Speech Translation

            This demo translates live microphone audio into English using
            OpenAI’s Whisper model.  Choose the input language, click **Start
            Listening**, and speak into your microphone.  The translated
            English text will stream live.

            **Note:** This application runs locally in your browser.  No
            audio leaves your machine.
            """
        )
        with gr.Row():
            language_dropdown = gr.Dropdown(
                ["Auto", "French (fr)", "German (de)"],
                value="Auto",
                label="Input language",
            )
        with gr.Row():
            start_button = gr.Button("Start Listening", variant="primary")
            stop_button = gr.Button("Stop", variant="stop")
        output_text = gr.Textbox(
            label="Translated English",
            lines=10,
            show_copy_button=True,
        )
        # Define streaming behaviour.  When the user clicks start,
        # call `start_translation` and stream its output into the
        # textbox.  When stop is clicked, call `stop_translation`.
        start_button.click(
            start_translation,
            inputs=language_dropdown,
            outputs=output_text,
            api_name="start_listening",
            stream=True,
        )
        stop_button.click(
            stop_translation,
            outputs=output_text,
            api_name="stop_listening",
        )
    demo.launch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a real‑time Whisper translator with optional Gradio UI."
    )
    parser.add_argument(
        "--no-ui",
        action="store_true",
        help="If set, do not launch the Gradio UI and instead run a simple\n"
             "command‑line demonstration that prints translated text to\n"
             "stdout.",
    )
    parser.add_argument(
        "--model-size",
        default="base",
        help="Whisper model size to load (e.g. tiny, base, small).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to run the model on (cpu or cuda).",
    )
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=3.0,
        help="Number of seconds of audio to accumulate before translation.",
    )
    args = parser.parse_args()

    translator = RealTimeTranslator(
        model_size=args.model_size,
        device=args.device,
        chunk_duration=args.chunk_duration,
    )
    if args.no_ui or gr is None:
        # Fallback: simple command‑line demonstration.  This mode
        # captures audio and prints the growing transcript to the
        # terminal.  Stop the capture with Ctrl+C.
        print(
            "Listening... Speak into your microphone. Press Ctrl+C to stop."
        )
        try:
            generator = translator.start_stream(language_code=None)
            for text in generator:
                # Clear the previous line and print the new transcript
                print("\r" + text, end="", flush=True)
        except KeyboardInterrupt:
            translator.stop()
            print("\nStopped. Final transcript:\n", translator.transcript)
    else:
        launch_gradio()