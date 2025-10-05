"""Test script for Whisper API."""

import requests
import json
import sys
from pathlib import Path


def test_transcription(audio_file: str, api_url: str = "http://localhost:8000"):
    """Test audio transcription with different response formats."""

    # Check if file exists
    if not Path(audio_file).exists():
        print(f"Error: Audio file '{audio_file}' not found")
        return

    print(f"\nTesting Whisper API with file: {audio_file}")
    print("=" * 50)

    # Test different response formats
    formats = ["json", "text", "verbose_json", "srt", "vtt"]

    for response_format in formats:
        print(f"\nTesting {response_format} format...")

        try:
            with open(audio_file, 'rb') as f:
                response = requests.post(
                    f"{api_url}/v1/audio/transcriptions",
                    files={'file': f},
                    data={
                        'model': 'whisper-1',
                        'response_format': response_format,
                        'temperature': 0.0
                    }
                )

            if response.status_code == 200:
                print(f"✓ Success! Status code: {response.status_code}")

                # Display response based on format
                if response_format in ["json", "verbose_json"]:
                    result = response.json()
                    print(f"Response: {json.dumps(result, indent=2)[:500]}...")
                else:
                    print(f"Response: {response.text[:500]}...")
            else:
                print(f"✗ Error! Status code: {response.status_code}")
                print(f"Response: {response.text}")

        except Exception as e:
            print(f"✗ Request failed: {e}")


def test_health_check(api_url: str = "http://localhost:8000"):
    """Test health check endpoint."""
    print("\nTesting Health Check...")
    print("=" * 50)

    try:
        response = requests.get(f"{api_url}/health")
        if response.status_code == 200:
            print(f"✓ API is healthy!")
            health_data = response.json()
            print(json.dumps(health_data, indent=2))
        else:
            print(f"✗ Health check failed! Status code: {response.status_code}")
    except Exception as e:
        print(f"✗ Cannot connect to API: {e}")
        print("Make sure the API server is running on port 8000")
        sys.exit(1)


def create_test_audio():
    """Create a test audio file using text-to-speech (requires pyttsx3)."""
    try:
        import pyttsx3

        engine = pyttsx3.init()
        test_text = "This is a test audio file for the Whisper API transcription service."

        # Save to file
        output_file = "test_audio.wav"
        engine.save_to_file(test_text, output_file)
        engine.runAndWait()

        print(f"Created test audio file: {output_file}")
        return output_file
    except ImportError:
        print("pyttsx3 not installed. Using any existing audio file.")
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Whisper API")
    parser.add_argument(
        "--audio",
        type=str,
        help="Path to audio file for testing"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="API URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--create-test-audio",
        action="store_true",
        help="Create a test audio file"
    )

    args = parser.parse_args()

    # Test health check first
    test_health_check(args.url)

    # Create test audio if requested
    if args.create_test_audio:
        audio_file = create_test_audio()
        if audio_file:
            test_transcription(audio_file, args.url)
    elif args.audio:
        test_transcription(args.audio, args.url)
    else:
        print("\nNo audio file provided. Use --audio <file> or --create-test-audio")
        print("Example: python test_api.py --audio sample.mp3")