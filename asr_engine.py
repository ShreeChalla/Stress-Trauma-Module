"""
Real-Time Speech Recognition Engine using faster-whisper on GPU.

Features:
- faster-whisper 'small' model running on CUDA (float16) with CPU fallback.
- Silero VAD filtering (min_silence_duration_ms=500) to reject static/line noise.
- Domain initial prompt injection for NHAA 14566 helpline context.
- 3-second sliding audio window with 0.5-second overlapping context buffer.
"""
import logging
import numpy as np

logger = logging.getLogger(__name__)

INITIAL_PROMPT = (
    "Context: Emergency crisis helpline call involving trauma, distress, harassment, "
    "threats, FIR, legal aid, and immediate triage."
)


class RealtimeASREngine:
    def __init__(self, model_size: str = "small", device: str = "cuda", compute_type: str = "float16"):
        self.model_size = model_size
        self.preferred_device = device
        self.compute_type = compute_type
        self.model = None
        self._init_model()

    def _init_model(self):
        try:
            from faster_whisper import WhisperModel
            try:
                # Try CUDA first if requested
                if self.preferred_device == "cuda":
                    import torch
                    if torch.cuda.is_available():
                        self.model = WhisperModel(
                            self.model_size,
                            device="cuda",
                            compute_type=self.compute_type,
                        )
                        logger.info("Initialized faster-whisper on CUDA.")
                        return
                    else:
                        logger.warning("CUDA requested but not available. Falling back to CPU.")
                
                # CPU fallback
                self.model = WhisperModel(
                    self.model_size,
                    device="cpu",
                    compute_type="int8",
                )
                logger.info("Initialized faster-whisper on CPU.")
            except Exception as e:
                logger.warning(f"Could not load faster-whisper on {self.preferred_device}: {e}. Retrying CPU.")
                self.model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
        except Exception as e:
            logger.error(f"faster-whisper not available or failed to load: {e}")
            self.model = None

    def transcribe_window(
        self,
        audio_window: np.ndarray,
        sr: int = 16000,
        language: str = "en",
    ) -> str:
        """
        Transcribes a 3-second (or current window) audio array with Silero VAD.
        """
        if self.model is None:
            return ""

        if audio_window.ndim > 1:
            audio_window = np.mean(audio_window, axis=1)

        # Ensure float32 normalized in [-1.0, 1.0]
        if audio_window.dtype != np.float32:
            audio_window = audio_window.astype(np.float32)
            if np.max(np.abs(audio_window)) > 1.0:
                audio_window /= 32768.0

        if len(audio_window) < sr * 0.2:
            return ""

        try:
            segments, info = self.model.transcribe(
                audio_window,
                language=language,
                initial_prompt=INITIAL_PROMPT,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
                beam_size=5,
            )
            text_parts = [segment.text.strip() for segment in segments]
            return " ".join(part for part in text_parts if part)
        except Exception as e:
            logger.error(f"ASR transcription error: {e}")
            return ""


class AudioBufferManager:
    """
    Manages 3-second window buffers with 0.5-second overlapping context.
    16 kHz audio:
    - Window length: 3.0 seconds = 48,000 samples
    - Context overlap: 0.5 seconds = 8,000 samples
    """
    def __init__(self, sr: int = 16000, window_sec: float = 3.0, overlap_sec: float = 0.5):
        self.sr = sr
        self.window_samples = int(sr * window_sec)
        self.overlap_samples = int(sr * overlap_sec)
        self.step_samples = self.window_samples - self.overlap_samples
        self.buffer = np.array([], dtype=np.float32)

    def append(self, chunk: np.ndarray):
        if chunk.ndim > 1:
            chunk = np.mean(chunk, axis=1)
        chunk = chunk.astype(np.float32)
        if np.max(np.abs(chunk)) > 1.0:
            chunk /= 32768.0
        self.buffer = np.concatenate((self.buffer, chunk))

    def has_window(self) -> bool:
        return len(self.buffer) >= self.window_samples

    def pop_window(self) -> np.ndarray:
        """
        Extracts a full 3-second window, sliding forward by (window - overlap) samples.
        """
        if len(self.buffer) < self.window_samples:
            return np.array([], dtype=np.float32)

        window = self.buffer[:self.window_samples]
        # Slide forward by step_samples, keeping overlap
        self.buffer = self.buffer[self.step_samples:]
        return window

    def reset(self):
        self.buffer = np.array([], dtype=np.float32)
