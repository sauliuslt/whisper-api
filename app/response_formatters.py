"""Response formatting utilities for different output formats."""

from typing import Dict, Any, List
from app.models import ResponseFormat


class ResponseFormatter:
    """Format transcription results for different output formats."""

    @staticmethod
    def format_response(result: Dict[str, Any], format_type: ResponseFormat) -> Any:
        """
        Format transcription result based on requested format.

        Args:
            result: Whisper transcription result
            format_type: Requested response format

        Returns:
            Formatted response
        """
        if format_type == ResponseFormat.TEXT:
            return result["text"]

        elif format_type == ResponseFormat.JSON:
            return {"text": result["text"]}

        elif format_type == ResponseFormat.VERBOSE_JSON:
            return ResponseFormatter._format_verbose_json(result)

        elif format_type == ResponseFormat.SRT:
            return ResponseFormatter._format_srt(result)

        elif format_type == ResponseFormat.VTT:
            return ResponseFormatter._format_vtt(result)

        else:
            return {"text": result["text"]}

    @staticmethod
    def _format_verbose_json(result: Dict[str, Any]) -> Dict[str, Any]:
        """Format as verbose JSON with segments and timestamps."""
        verbose = {
            "task": "transcribe",
            "language": result.get("language", "unknown"),
            "duration": result.get("duration", 0.0),
            "text": result["text"],
            "segments": []
        }

        # Add segments
        for segment in result.get("segments", []):
            seg_data = {
                "id": segment["id"],
                "seek": segment.get("seek", 0),
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"],
                "tokens": segment.get("tokens", []),
                "temperature": segment.get("temperature", 0.0),
                "avg_logprob": segment.get("avg_logprob", 0.0),
                "compression_ratio": segment.get("compression_ratio", 0.0),
                "no_speech_prob": segment.get("no_speech_prob", 0.0)
            }

            # Add word-level timestamps if available
            if "words" in segment:
                seg_data["words"] = [
                    {
                        "word": word["word"],
                        "start": word["start"],
                        "end": word["end"]
                    }
                    for word in segment["words"]
                ]

            verbose["segments"].append(seg_data)

        # Add top-level words if requested
        if "words" in result:
            verbose["words"] = [
                {
                    "word": word["word"],
                    "start": word["start"],
                    "end": word["end"]
                }
                for word in result["words"]
            ]

        return verbose

    @staticmethod
    def _format_srt(result: Dict[str, Any]) -> str:
        """Format as SRT subtitle file."""
        srt_lines = []
        segments = result.get("segments", [])

        for i, segment in enumerate(segments, 1):
            # Format: subtitle number
            srt_lines.append(str(i))

            # Format: timestamps (HH:MM:SS,mmm --> HH:MM:SS,mmm)
            start_time = ResponseFormatter._seconds_to_srt_time(segment["start"])
            end_time = ResponseFormatter._seconds_to_srt_time(segment["end"])
            srt_lines.append(f"{start_time} --> {end_time}")

            # Format: subtitle text
            srt_lines.append(segment["text"].strip())

            # Empty line between subtitles
            srt_lines.append("")

        return "\n".join(srt_lines)

    @staticmethod
    def _format_vtt(result: Dict[str, Any]) -> str:
        """Format as WebVTT subtitle file."""
        vtt_lines = ["WEBVTT", ""]
        segments = result.get("segments", [])

        for segment in segments:
            # Format: timestamps (HH:MM:SS.mmm --> HH:MM:SS.mmm)
            start_time = ResponseFormatter._seconds_to_vtt_time(segment["start"])
            end_time = ResponseFormatter._seconds_to_vtt_time(segment["end"])
            vtt_lines.append(f"{start_time} --> {end_time}")

            # Format: subtitle text
            vtt_lines.append(segment["text"].strip())

            # Empty line between subtitles
            vtt_lines.append("")

        return "\n".join(vtt_lines)

    @staticmethod
    def _seconds_to_srt_time(seconds: float) -> str:
        """Convert seconds to SRT time format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    @staticmethod
    def _seconds_to_vtt_time(seconds: float) -> str:
        """Convert seconds to WebVTT time format (HH:MM:SS.mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

    @staticmethod
    def format_error(error_message: str, error_type: str = "invalid_request_error") -> Dict[str, Any]:
        """Format error response."""
        return {
            "error": {
                "message": error_message,
                "type": error_type,
                "code": error_type
            }
        }