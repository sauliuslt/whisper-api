"""Main FastAPI application."""

import os
import logging
import json
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from app.config import settings
from app.models import (
    ResponseFormat,
    TranscriptionResponse,
    ErrorResponse,
    HealthResponse
)
from app.whisper_model import whisper_model
from app.audio_utils import AudioProcessor
from app.response_formatters import ResponseFormatter

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s' if settings.log_format == "plain"
    else '{"time": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Startup
    logger.info("Starting Whisper API server...")
    try:
        # Model is loaded automatically via singleton
        device_info = whisper_model.get_device_info()
        logger.info(f"Device info: {device_info}")
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down Whisper API server...")
    whisper_model.cleanup()


# Create FastAPI app
app = FastAPI(
    title="Whisper API",
    description="OpenAI-compatible transcription API using Whisper v3 with GPU acceleration",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint returning health status."""
    return await health_check()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    device_info = whisper_model.get_device_info()
    return HealthResponse(
        status="healthy",
        model_loaded=device_info["model_loaded"],
        gpu_available=device_info["gpu_available"],
        gpu_name=device_info.get("gpu_name"),
        gpu_memory_used=device_info.get("gpu_memory_used"),
        gpu_memory_total=device_info.get("gpu_memory_total")
    )


@app.post("/v1/audio/transcriptions")
@app.post("/api/v1/audio/transcriptions")  # Also support /api prefix
async def transcribe_audio(
    file: UploadFile = File(...),
    model: str = Form(default="whisper-1"),
    language: Optional[str] = Form(default=None),
    prompt: Optional[str] = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0),
    timestamp_granularities: Optional[str] = Form(default=None)
):
    """
    Transcribe audio file using Whisper model.
    Compatible with OpenAI's audio transcription API.
    """
    audio_path = None
    wav_path = None

    try:
        # Validate response format
        try:
            format_type = ResponseFormat(response_format)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=ResponseFormatter.format_error(
                    f"Invalid response_format: {response_format}. Must be one of: json, text, srt, verbose_json, vtt"
                )
            )

        # Validate temperature
        if not 0.0 <= temperature <= 1.0:
            raise HTTPException(
                status_code=400,
                detail=ResponseFormatter.format_error("Temperature must be between 0.0 and 1.0")
            )

        # Validate language code if provided
        if language and len(language) != 2:
            raise HTTPException(
                status_code=400,
                detail=ResponseFormatter.format_error("Language must be a 2-letter ISO-639-1 code")
            )

        # Read file content
        file_content = await file.read()

        # Validate file size
        if len(file_content) > settings.max_file_size_bytes:
            raise HTTPException(
                status_code=400,
                detail=ResponseFormatter.format_error(
                    f"File size exceeds maximum of {settings.max_audio_file_size_mb}MB"
                )
            )

        # Save uploaded file
        audio_path = AudioProcessor.save_uploaded_file(file_content, file.filename)

        # Validate audio file
        is_valid, error_msg = AudioProcessor.validate_audio_file(
            audio_path,
            settings.max_file_size_bytes
        )
        if not is_valid:
            raise HTTPException(
                status_code=400,
                detail=ResponseFormatter.format_error(error_msg)
            )

        # Convert to WAV for Whisper processing
        wav_path = AudioProcessor.convert_to_wav(audio_path)

        # Determine if word timestamps are needed
        need_word_timestamps = (
            format_type == ResponseFormat.VERBOSE_JSON or
            (timestamp_granularities and "word" in timestamp_granularities)
        )

        # Perform transcription
        logger.info(f"Transcribing file: {file.filename}")
        result = whisper_model.transcribe(
            audio_path=wav_path,
            language=language,
            prompt=prompt,
            temperature=temperature,
            word_timestamps=need_word_timestamps
        )

        # Get audio duration if not provided
        if "duration" not in result:
            result["duration"] = AudioProcessor.get_audio_duration(wav_path)

        # Format response based on requested format
        formatted_response = ResponseFormatter.format_response(result, format_type)

        # Return appropriate response type
        if format_type == ResponseFormat.TEXT:
            return PlainTextResponse(content=formatted_response)
        elif format_type == ResponseFormat.SRT:
            return PlainTextResponse(
                content=formatted_response,
                media_type="text/srt"
            )
        elif format_type == ResponseFormat.VTT:
            return PlainTextResponse(
                content=formatted_response,
                media_type="text/vtt"
            )
        else:
            return JSONResponse(content=formatted_response)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=ResponseFormatter.format_error(
                f"Internal server error: {str(e)}",
                "internal_error"
            )
        )
    finally:
        # Cleanup temporary files
        if audio_path:
            AudioProcessor.cleanup_temp_file(audio_path)
        if wav_path and wav_path != audio_path:
            AudioProcessor.cleanup_temp_file(wav_path)


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions with proper error format."""
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.detail if isinstance(exc.detail, dict) else {"error": {"message": exc.detail}}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ResponseFormatter.format_error(
            "Internal server error",
            "internal_error"
        )
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        reload=False
    )