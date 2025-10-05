"""Pydantic models for API requests and responses."""

from typing import Optional, Literal, List, Dict, Any
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class ResponseFormat(str, Enum):
    """Supported response formats matching OpenAI API."""
    JSON = "json"
    TEXT = "text"
    SRT = "srt"
    VERBOSE_JSON = "verbose_json"
    VTT = "vtt"


class TranscriptionRequest(BaseModel):
    """Request model for audio transcription."""
    model: str = Field(default="whisper-1", description="Model to use for transcription")
    language: Optional[str] = Field(default=None, description="Language of the audio (ISO-639-1)")
    prompt: Optional[str] = Field(default=None, description="Optional prompt to guide the model")
    response_format: ResponseFormat = Field(default=ResponseFormat.JSON, description="Response format")
    temperature: float = Field(default=0.0, ge=0.0, le=1.0, description="Sampling temperature")
    timestamp_granularities: Optional[List[Literal["word", "segment"]]] = Field(
        default=None, description="Timestamp granularities for verbose_json format"
    )

    @field_validator("language")
    def validate_language(cls, v):
        """Validate language code."""
        if v is not None and len(v) != 2:
            raise ValueError("Language must be a 2-letter ISO-639-1 code")
        return v


class TranscriptionResponse(BaseModel):
    """Basic transcription response."""
    text: str = Field(description="Transcribed text")


class Word(BaseModel):
    """Word-level timestamp information."""
    word: str
    start: float
    end: float


class Segment(BaseModel):
    """Segment-level information."""
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: List[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    words: Optional[List[Word]] = None


class VerboseTranscriptionResponse(BaseModel):
    """Verbose transcription response with detailed information."""
    task: str = Field(default="transcribe")
    language: str
    duration: float
    text: str
    segments: List[Segment]
    words: Optional[List[Word]] = None


class ErrorResponse(BaseModel):
    """Error response model."""
    error: Dict[str, Any] = Field(description="Error details")

    class Config:
        json_schema_extra = {
            "example": {
                "error": {
                    "message": "Invalid audio file format",
                    "type": "invalid_request_error",
                    "code": "invalid_audio_format"
                }
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(default="healthy")
    model_loaded: bool
    gpu_available: bool
    gpu_name: Optional[str] = None
    gpu_memory_used: Optional[float] = None
    gpu_memory_total: Optional[float] = None

    class Config:
        protected_namespaces = ()  # Disable protected namespace check