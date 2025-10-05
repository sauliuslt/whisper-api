"""Configuration management for Whisper API."""

from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Whisper Model Configuration
    whisper_model: str = Field(default="large-v3", description="Whisper model to use")
    whisper_device: str = Field(default="cuda", description="Device for inference (cuda/cpu)")
    whisper_compute_type: str = Field(default="float16", description="Compute precision")

    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    api_workers: int = Field(default=1, description="Number of API workers")
    max_audio_file_size_mb: int = Field(default=25, description="Max audio file size in MB")
    allowed_audio_formats: str = Field(
        default="mp3,mp4,mpeg,mpga,m4a,wav,webm,flac,ogg",
        description="Comma-separated list of allowed audio formats"
    )

    # GPU Configuration
    cuda_visible_devices: str = Field(default="0", description="CUDA device IDs")
    gpu_memory_fraction: float = Field(default=0.9, description="GPU memory fraction to use")

    # Performance
    enable_model_cache: bool = Field(default=True, description="Enable model caching")
    model_cache_dir: str = Field(default="/app/models", description="Model cache directory")
    batch_size: int = Field(default=1, description="Batch size for inference")
    num_workers: int = Field(default=4, description="Number of data loading workers")

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Logging format")

    class Config:
        env_file = ".env"
        case_sensitive = False

    @property
    def allowed_formats_list(self) -> List[str]:
        """Get allowed audio formats as a list."""
        return [fmt.strip() for fmt in self.allowed_audio_formats.split(",")]

    @property
    def max_file_size_bytes(self) -> int:
        """Get max file size in bytes."""
        return self.max_audio_file_size_mb * 1024 * 1024


# Global settings instance
settings = Settings()