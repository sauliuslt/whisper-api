"""Whisper model management with GPU acceleration using faster-whisper."""

import os
import logging
import torch
from faster_whisper import WhisperModel as FasterWhisperModel
from typing import Optional, Dict, Any
from app.config import settings

logger = logging.getLogger(__name__)


class WhisperModel:
    """Singleton class for managing Whisper model with GPU acceleration."""

    _instance = None
    _model = None
    _device = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the Whisper model manager."""
        if self._model is None:
            self.load_model()

    def load_model(self):
        """Load Whisper model with GPU acceleration."""
        try:
            # Debug GPU detection
            logger.info("=" * 50)
            logger.info("GPU DETECTION DEBUG INFO")
            logger.info(f"settings.whisper_device: {settings.whisper_device}")
            logger.info(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
            logger.info(f"torch.cuda.device_count(): {torch.cuda.device_count() if torch.cuda.is_available() else 'N/A'}")
            logger.info(f"CUDA_VISIBLE_DEVICES env: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

            if torch.cuda.is_available():
                logger.info(f"CUDA version: {torch.version.cuda}")
                logger.info(f"PyTorch version: {torch.__version__}")
                for i in range(torch.cuda.device_count()):
                    logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                    props = torch.cuda.get_device_properties(i)
                    logger.info(f"  - Memory: {props.total_memory / 1e9:.2f} GB")
                    logger.info(f"  - Compute Capability: {props.major}.{props.minor}")
            else:
                logger.warning("CUDA is NOT available!")
                logger.info(f"PyTorch version: {torch.__version__}")
                logger.info("Possible reasons:")
                logger.info("  1. No NVIDIA GPU present")
                logger.info("  2. NVIDIA drivers not installed")
                logger.info("  3. PyTorch installed without CUDA support")
                logger.info("  4. CUDA toolkit version mismatch")
            logger.info("=" * 50)

            # Set CUDA device
            if settings.whisper_device == "cuda" and torch.cuda.is_available():
                os.environ["CUDA_VISIBLE_DEVICES"] = settings.cuda_visible_devices
                self._device = "cuda"
                logger.info(f"✓ Using GPU: {torch.cuda.get_device_name()}")
                logger.info(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            else:
                self._device = "cpu"
                if settings.whisper_device == "cuda":
                    logger.error("GPU was requested but not available, falling back to CPU")
                else:
                    logger.info("Using CPU as configured")

            # Load model using faster-whisper
            logger.info(f"Loading Whisper model: {settings.whisper_model}")

            # Set download root for model caching
            if settings.enable_model_cache:
                os.makedirs(settings.model_cache_dir, exist_ok=True)
                download_root = settings.model_cache_dir
            else:
                download_root = None

            # Map compute type
            compute_type = settings.whisper_compute_type if self._device == "cuda" else "int8"

            self._model = FasterWhisperModel(
                model_size_or_path=settings.whisper_model,
                device=self._device,
                compute_type=compute_type,
                download_root=download_root,
                device_index=0 if self._device == "cuda" else None
            )

            logger.info("Model loaded successfully")

            # Warm up the model
            self._warmup_model()

        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise

    def _warmup_model(self):
        """Warm up the model with a dummy inference."""
        try:
            logger.info("Warming up model...")
            # Create a small dummy audio array (1 second of silence)
            import numpy as np
            dummy_audio = np.zeros(16000, dtype=np.float32)

            # Perform warmup transcription
            segments, _ = self._model.transcribe(
                dummy_audio,
                language='en',
                beam_size=5
            )
            # Consume the generator
            list(segments)
            logger.info("Model warmup completed successfully")
        except Exception as e:
            logger.warning(f"Model warmup failed (non-critical): {e}")

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        temperature: float = 0.0,
        word_timestamps: bool = False
    ) -> Dict[str, Any]:
        """
        Transcribe audio file using Whisper model.

        Args:
            audio_path: Path to the audio file
            language: Optional language code
            prompt: Optional prompt text
            temperature: Sampling temperature
            word_timestamps: Whether to include word-level timestamps

        Returns:
            Transcription results dictionary
        """
        try:
            # Prepare transcription options for faster-whisper
            options = {
                "language": language,
                "initial_prompt": prompt,
                "temperature": temperature,
                "word_timestamps": word_timestamps,
                "beam_size": 5,
                "condition_on_previous_text": False,  # Prevent hallucination loops
                "vad_filter": True,  # Enable voice activity detection
                "vad_parameters": {
                    "threshold": 0.5,
                    "min_speech_duration_ms": 250,
                    "min_silence_duration_ms": 100
                }
            }

            # Remove None values
            options = {k: v for k, v in options.items() if v is not None}

            # Perform transcription
            segments, info = self._model.transcribe(audio_path, **options)

            # Convert generator to list and format output to match OpenAI Whisper format
            segments_list = list(segments)

            # Build text
            text = " ".join([segment.text.strip() for segment in segments_list])

            # Build segments array
            formatted_segments = []
            for segment in segments_list:
                seg_dict = {
                    "id": segment.id,
                    "seek": segment.seek,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "tokens": segment.tokens,
                    "temperature": segment.temperature,
                    "avg_logprob": segment.avg_logprob,
                    "compression_ratio": segment.compression_ratio,
                    "no_speech_prob": segment.no_speech_prob
                }

                # Add words if word_timestamps is enabled
                if word_timestamps and segment.words:
                    seg_dict["words"] = [
                        {
                            "word": word.word,
                            "start": word.start,
                            "end": word.end,
                            "probability": word.probability
                        }
                        for word in segment.words
                    ]

                formatted_segments.append(seg_dict)

            # Build result dict matching OpenAI Whisper format
            result = {
                "text": text,
                "segments": formatted_segments,
                "language": info.language if hasattr(info, 'language') else language
            }

            return result

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise

    def get_device_info(self) -> Dict[str, Any]:
        """Get information about the current device."""
        info = {
            "device": self._device,
            "model_loaded": self._model is not None,
            "model_name": settings.whisper_model
        }

        if self._device == "cuda":
            info.update({
                "gpu_available": True,
                "gpu_name": torch.cuda.get_device_name(),
                "gpu_memory_used": torch.cuda.memory_allocated() / 1e9,
                "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory / 1e9,
                "compute_type": settings.whisper_compute_type
            })
        else:
            info["gpu_available"] = False

        return info

    def cleanup(self):
        """Clean up model and free GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None

            if self._device == "cuda":
                torch.cuda.empty_cache()

            logger.info("Model cleaned up")


# Global model instance
whisper_model = WhisperModel()
