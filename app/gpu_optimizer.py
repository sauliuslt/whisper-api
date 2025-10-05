"""GPU optimization utilities for Whisper inference."""

import torch
import logging
from typing import Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class GPUOptimizer:
    """Optimize GPU usage for Whisper inference."""

    @staticmethod
    def optimize_memory():
        """Optimize GPU memory usage."""
        if torch.cuda.is_available():
            # Clear cache
            torch.cuda.empty_cache()

            # Set memory allocator settings for better performance
            torch.cuda.set_per_process_memory_fraction(0.9)

            # Enable TF32 for Ampere GPUs (30xx series and newer)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            # Enable cudnn autotuner for better performance
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

            logger.info("GPU memory optimizations applied")

    @staticmethod
    def get_optimal_batch_size(model_size: str, gpu_memory_gb: float) -> int:
        """
        Get optimal batch size based on model and GPU memory.

        Args:
            model_size: Whisper model size
            gpu_memory_gb: Available GPU memory in GB

        Returns:
            Recommended batch size
        """
        # Approximate memory requirements per batch item (GB)
        memory_per_item = {
            "tiny": 0.5,
            "base": 0.7,
            "small": 1.0,
            "medium": 2.0,
            "large": 3.0,
            "large-v2": 3.0,
            "large-v3": 3.5
        }

        model_base = model_size.split("-")[0]  # Handle large-v2, large-v3
        mem_required = memory_per_item.get(model_base, 3.0)

        # Reserve 2GB for system and other operations
        available_memory = max(gpu_memory_gb - 2, 1)

        # Calculate batch size
        batch_size = max(int(available_memory / mem_required), 1)

        logger.info(f"Optimal batch size for {model_size}: {batch_size}")
        return batch_size

    @staticmethod
    @contextmanager
    def autocast_context(enabled: bool = True, dtype: torch.dtype = torch.float16):
        """
        Context manager for automatic mixed precision.

        Args:
            enabled: Whether to enable autocast
            dtype: Data type for autocast
        """
        if enabled and torch.cuda.is_available():
            with torch.cuda.amp.autocast(dtype=dtype):
                yield
        else:
            yield

    @staticmethod
    def profile_memory():
        """Profile current GPU memory usage."""
        if not torch.cuda.is_available():
            return {"gpu_available": False}

        return {
            "gpu_available": True,
            "allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
            "reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
            "free_mb": (torch.cuda.get_device_properties(0).total_memory -
                       torch.cuda.memory_reserved()) / 1024 / 1024
        }

    @staticmethod
    def enable_flash_attention():
        """Enable Flash Attention if available (requires specific PyTorch version)."""
        try:
            # Check if flash attention is available
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                # Enable flash attention
                torch.backends.cuda.enable_flash_sdp(True)
                logger.info("Flash Attention enabled")
                return True
        except Exception as e:
            logger.debug(f"Flash Attention not available: {e}")
        return False

    @staticmethod
    def set_gpu_device(device_id: int = 0):
        """
        Set specific GPU device.

        Args:
            device_id: CUDA device ID
        """
        if torch.cuda.is_available():
            torch.cuda.set_device(device_id)
            logger.info(f"Using GPU device: {device_id} - {torch.cuda.get_device_name(device_id)}")

    @staticmethod
    def optimize_model_for_inference(model):
        """
        Apply inference optimizations to model.

        Args:
            model: PyTorch model

        Returns:
            Optimized model
        """
        if not torch.cuda.is_available():
            return model

        # Set to evaluation mode
        model.eval()

        # Disable gradient computation
        for param in model.parameters():
            param.requires_grad = False

        # Try to compile model with torch.compile (PyTorch 2.0+)
        try:
            if hasattr(torch, 'compile'):
                model = torch.compile(model, mode='reduce-overhead')
                logger.info("Model compiled with torch.compile")
        except Exception as e:
            logger.debug(f"torch.compile not available: {e}")

        return model


# Initialize GPU optimizations on module import
if torch.cuda.is_available():
    GPUOptimizer.optimize_memory()