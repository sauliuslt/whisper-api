# Whisper API - GPU-Accelerated OpenAI-Compatible Transcription Service

A high-performance REST API service that provides OpenAI-compatible audio transcription using Whisper v3 with GPU acceleration.

## Features

- **Full OpenAI API Compatibility**: Drop-in replacement for OpenAI's `/v1/audio/transcriptions` endpoint
- **GPU Acceleration**: Optimized for NVIDIA GPUs with CUDA support
- **Whisper v3 Support**: Uses the latest Whisper large-v3 model for best accuracy
- **Multiple Output Formats**: JSON, text, SRT, VTT, and verbose JSON with timestamps
- **Production Ready**: Docker support with health checks and proper error handling
- **Optimized Performance**: FP16 inference, model caching, and GPU memory management

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA 11.8+ support
- FFmpeg for audio processing
- 8GB+ GPU memory for large-v3 model

## Installation

### Local Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/whisper-api.git
cd whisper-api
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create `.env` file:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Run the server:
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Docker Installation

1. Build and run with Docker Compose:
```bash
docker-compose up -d
```

The API will be available at `http://localhost:8000`

## API Usage

### Transcribe Audio

**Endpoint**: `POST /v1/audio/transcriptions`

**Request**:
```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -H "Content-Type: multipart/form-data" \
  -F file="@/path/to/audio.mp3" \
  -F model="whisper-1" \
  -F response_format="json"
```

**Parameters**:
- `file` (required): Audio file (mp3, mp4, mpeg, mpga, m4a, wav, webm, flac, ogg)
- `model`: Model to use (default: "whisper-1")
- `language`: ISO-639-1 language code (optional, auto-detected if not provided)
- `prompt`: Optional text to guide the model
- `response_format`: Output format - `json`, `text`, `srt`, `vtt`, `verbose_json` (default: "json")
- `temperature`: Sampling temperature 0-1 (default: 0)

### Response Formats

#### JSON Response
```json
{
  "text": "This is the transcribed text."
}
```

#### Verbose JSON Response
```json
{
  "task": "transcribe",
  "language": "en",
  "duration": 10.5,
  "text": "This is the transcribed text.",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 2.5,
      "text": "This is the transcribed text.",
      "tokens": [1, 2, 3],
      "temperature": 0.0,
      "avg_logprob": -0.5,
      "compression_ratio": 1.2,
      "no_speech_prob": 0.01
    }
  ]
}
```

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "gpu_available": true,
  "gpu_name": "NVIDIA GeForce RTX 3090",
  "gpu_memory_used": 4.5,
  "gpu_memory_total": 24.0
}
```

## Python Client Example

```python
import requests

# Transcribe audio file
with open('audio.mp3', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/v1/audio/transcriptions',
        files={'file': f},
        data={
            'model': 'whisper-1',
            'response_format': 'json'
        }
    )

transcription = response.json()
print(transcription['text'])
```

## OpenAI SDK Compatibility

```python
from openai import OpenAI

# Point to your local Whisper API
client = OpenAI(
    api_key="dummy",  # API key not required
    base_url="http://localhost:8000/v1"
)

# Use exactly like OpenAI's API
with open("audio.mp3", "rb") as f:
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=f,
        response_format="text"
    )
    print(transcription)
```

## Configuration

Edit `.env` file or set environment variables:

```bash
# Whisper Model Configuration
WHISPER_MODEL=large-v3  # tiny, base, small, medium, large, large-v2, large-v3
WHISPER_DEVICE=cuda     # cuda or cpu
WHISPER_COMPUTE_TYPE=float16  # float16, float32

# GPU Configuration
CUDA_VISIBLE_DEVICES=0  # GPU device ID
GPU_MEMORY_FRACTION=0.9  # Fraction of GPU memory to use

# API Configuration
MAX_AUDIO_FILE_SIZE_MB=25
API_PORT=8000
```

## Model Sizes and Requirements

| Model | Parameters | GPU Memory | Relative Speed |
|-------|------------|------------|----------------|
| tiny | 39M | ~1 GB | ~32x |
| base | 74M | ~1 GB | ~16x |
| small | 244M | ~2 GB | ~6x |
| medium | 769M | ~5 GB | ~2x |
| large-v3 | 1550M | ~10 GB | 1x |

## Performance Tips

1. **Use GPU**: CPU inference is 10-50x slower than GPU
2. **Model Selection**: Use smaller models for faster inference if accuracy permits
3. **Batch Processing**: Increase batch size if you have enough GPU memory
4. **FP16 Inference**: Enabled by default for 2x speedup with minimal quality loss
5. **Model Caching**: Models are cached locally after first download

## Development

### Run Tests
```bash
pytest tests/
```

### Format Code
```bash
black app/
isort app/
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `GPU_MEMORY_FRACTION` in `.env`
- Use a smaller model
- Reduce batch size

### Model Download Issues
- Check internet connection
- Manually download model:
  ```python
  import whisper
  whisper.load_model("large-v3", download_root="./models")
  ```

### Audio Format Not Supported
- Ensure FFmpeg is installed
- Check supported formats in configuration

## License

MIT License

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for the amazing speech recognition model
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [PyTorch](https://pytorch.org/) for deep learning infrastructure