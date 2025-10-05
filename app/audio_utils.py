"""Audio file processing utilities."""

import os
import tempfile
import logging
import subprocess
from typing import Optional, Tuple
from pathlib import Path
import ffmpeg

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Handle audio file processing and validation."""

    SUPPORTED_FORMATS = {
        "mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm", "flac", "ogg"
    }

    @staticmethod
    def validate_audio_file(file_path: str, max_size_bytes: int) -> Tuple[bool, Optional[str]]:
        """
        Validate audio file format and size.

        Args:
            file_path: Path to the audio file
            max_size_bytes: Maximum allowed file size in bytes

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check file exists
            if not os.path.exists(file_path):
                return False, "File does not exist"

            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size > max_size_bytes:
                return False, f"File size {file_size / 1e6:.2f}MB exceeds maximum {max_size_bytes / 1e6:.2f}MB"

            # Check format by extension
            extension = Path(file_path).suffix.lower().lstrip(".")
            if extension not in AudioProcessor.SUPPORTED_FORMATS:
                # Try to detect format using ffprobe
                if not AudioProcessor.probe_audio_format(file_path):
                    return False, f"Unsupported audio format: {extension}"

            return True, None

        except Exception as e:
            return False, f"Validation error: {str(e)}"

    @staticmethod
    def probe_audio_format(file_path: str) -> bool:
        """
        Use ffprobe to check if file is a valid audio file.

        Args:
            file_path: Path to the audio file

        Returns:
            True if valid audio file, False otherwise
        """
        try:
            probe = ffmpeg.probe(file_path, select_streams='a')
            return len(probe.get('streams', [])) > 0
        except ffmpeg.Error:
            return False
        except Exception as e:
            logger.error(f"Error probing audio file: {e}")
            return False

    @staticmethod
    def convert_to_wav(input_path: str, output_path: Optional[str] = None) -> str:
        """
        Convert audio file to WAV format for Whisper processing.

        Args:
            input_path: Path to input audio file
            output_path: Optional output path (temp file if not provided)

        Returns:
            Path to converted WAV file
        """
        try:
            if output_path is None:
                temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                output_path = temp_file.name
                temp_file.close()

            # Convert to 16kHz mono WAV (Whisper's expected format)
            stream = ffmpeg.input(input_path)
            stream = ffmpeg.output(
                stream,
                output_path,
                acodec='pcm_s16le',
                ac=1,  # Mono
                ar='16k'  # 16kHz sample rate
            )
            ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)

            logger.info(f"Converted {input_path} to WAV format")
            return output_path

        except ffmpeg.Error as e:
            logger.error(f"FFmpeg error: {e.stderr.decode()}")
            raise ValueError(f"Failed to convert audio file: {e.stderr.decode()}")
        except Exception as e:
            logger.error(f"Conversion error: {e}")
            raise

    @staticmethod
    def get_audio_duration(file_path: str) -> float:
        """
        Get duration of audio file in seconds.

        Args:
            file_path: Path to audio file

        Returns:
            Duration in seconds
        """
        try:
            probe = ffmpeg.probe(file_path)
            duration = float(probe['streams'][0]['duration'])
            return duration
        except Exception as e:
            logger.error(f"Error getting audio duration: {e}")
            return 0.0

    @staticmethod
    def save_uploaded_file(file_content: bytes, filename: str) -> str:
        """
        Save uploaded file to temporary location.

        Args:
            file_content: File content as bytes
            filename: Original filename

        Returns:
            Path to saved file
        """
        try:
            # Get file extension
            extension = Path(filename).suffix

            # Create temp file with same extension
            with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as tmp_file:
                tmp_file.write(file_content)
                return tmp_file.name
        except Exception as e:
            logger.error(f"Error saving uploaded file: {e}")
            raise

    @staticmethod
    def cleanup_temp_file(file_path: str):
        """
        Clean up temporary file.

        Args:
            file_path: Path to temporary file
        """
        try:
            if os.path.exists(file_path) and file_path.startswith(tempfile.gettempdir()):
                os.remove(file_path)
                logger.debug(f"Cleaned up temp file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp file {file_path}: {e}")