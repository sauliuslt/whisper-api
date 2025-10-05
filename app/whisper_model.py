"""Whisper model management with GPU acceleration."""

import os
import logging
import torch
import whisper
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

                # Set memory fraction
                torch.cuda.set_per_process_memory_fraction(settings.gpu_memory_fraction)

                logger.info(f"✓ Using GPU: {torch.cuda.get_device_name()}")
                logger.info(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            else:
                self._device = "cpu"
                if settings.whisper_device == "cuda":
                    logger.error("GPU was requested but not available, falling back to CPU")
                else:
                    logger.info("Using CPU as configured")

            # Load model
            logger.info(f"Loading Whisper model: {settings.whisper_model}")

            # Set download root for model caching
            if settings.enable_model_cache:
                os.makedirs(settings.model_cache_dir, exist_ok=True)
                download_root = settings.model_cache_dir
            else:
                download_root = None

            self._model = whisper.load_model(
                name=settings.whisper_model,
                device=self._device,
                download_root=download_root
            )

            # Set compute type for faster inference
            if self._device == "cuda" and settings.whisper_compute_type == "float16":
                self._model = self._model.half()
            elif self._device == "cpu":
                # Ensure model is in float32 for CPU
                self._model = self._model.float()

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
            dummy_audio = torch.zeros(16000).to(self._device)
            _ = self._model.transcribe(
                dummy_audio.cpu().numpy(),
                fp16=False if self._device == "cpu" else (settings.whisper_compute_type == "float16")
            )
            logger.info("Model warmup completed")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")

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
            # Prepare transcription options
            options = {
                "language": language,
                "initial_prompt": prompt,
                "temperature": temperature,
                "word_timestamps": word_timestamps,
                "fp16": False if self._device == "cpu" else (settings.whisper_compute_type == "float16"),
                "verbose": False
            }

            # Remove None values
            options = {k: v for k, v in options.items() if v is not None}

            # Perform transcription
            result = self._model.transcribe(audio_path, **options)

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