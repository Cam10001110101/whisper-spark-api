# Whisper Spark API — User Guide

Transcription service running **OpenAI Whisper large-v3** on the DGX Spark GPU.
Base URL: `http://192.168.86.101:9000`

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check + model info |
| `POST` | `/transcribe` | Batch transcription (file upload) |
| `WS` | `/ws` | Streaming transcription over WebSocket |

---

## GET /health

Returns server status and loaded model info.

```bash
curl http://192.168.86.101:9000/health
```

**Response**
```json
{
  "status": "ok",
  "model": "large-v3",
  "device": "cuda"
}
```

---

## POST /transcribe

Upload an audio file and receive a full transcription. Accepts any format ffmpeg understands (WAV, MP3, MP4, M4A, FLAC, OGG, WebM, etc.).

**Form fields**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | Yes | Audio file |
| `language` | string | No | ISO 639-1 code (e.g. `en`, `fr`, `de`). Omit for auto-detect. |
| `task` | string | No | `transcribe` (default) or `translate` (to English) |

**Examples**

```bash
# Basic transcription (auto-detect language)
curl -X POST http://192.168.86.101:9000/transcribe \
  -F "file=@recording.wav"

# With language hint (faster, slightly more accurate)
curl -X POST http://192.168.86.101:9000/transcribe \
  -F "file=@recording.wav" \
  -F "language=en"

# Translate foreign speech to English
curl -X POST http://192.168.86.101:9000/transcribe \
  -F "file=@german_audio.mp3" \
  -F "task=translate"

# Pretty print
curl -X POST http://192.168.86.101:9000/transcribe \
  -F "file=@recording.wav" | python3 -m json.tool
```

**Response**
```json
{
  "text": "Hello world, this is a test.",
  "language": "en",
  "language_probability": null,
  "duration": 3.5,
  "segments": [
    { "start": 0.0, "end": 1.8, "text": "Hello world," },
    { "start": 1.8, "end": 3.5, "text": " this is a test." }
  ]
}
```

**Python example**
```python
import requests

with open("recording.wav", "rb") as f:
    resp = requests.post(
        "http://192.168.86.101:9000/transcribe",
        files={"file": f},
        data={"language": "en"},
    )

result = resp.json()
print(result["text"])
for seg in result["segments"]:
    print(f"[{seg['start']:.1f}s → {seg['end']:.1f}s] {seg['text']}")
```

---

## WebSocket /ws

Streaming transcription. Useful for longer recordings or when you want per-segment results as they arrive.

**Connection**
```
ws://192.168.86.101:9000/ws
```

**Protocol**

| Direction | Frame type | Content | Description |
|-----------|-----------|---------|-------------|
| Client → Server | binary | raw audio bytes | Append audio to buffer (any ffmpeg format) |
| Client → Server | text | `{"action": "transcribe"}` | Transcribe buffered audio, reset buffer, keep connection open |
| Client → Server | text | `{"action": "end"}` | Transcribe buffered audio, then close |
| Server → Client | text | `{"type": "segment", "start": 0.0, "end": 2.1, "text": "..."}` | One segment result |
| Server → Client | text | `{"type": "done", "text": "full text", "language": "en"}` | Transcription complete |

**Single-shot example** (send audio once, close)

```python
import asyncio, json, websockets

async def transcribe_file(path: str) -> str:
    async with websockets.connect("ws://192.168.86.101:9000/ws") as ws:
        with open(path, "rb") as f:
            await ws.send(f.read())
        await ws.send(json.dumps({"action": "end"}))

        segments = []
        try:
            async for msg in ws:
                data = json.loads(msg)
                if data["type"] == "segment":
                    print(f"[{data['start']:.1f}s] {data['text']}")
                    segments.append(data)
                elif data["type"] == "done":
                    return data["text"]
        except websockets.exceptions.ConnectionClosed:
            pass

    return " ".join(s["text"] for s in segments)

text = asyncio.run(transcribe_file("recording.wav"))
print(text)
```

**Multi-shot example** (multiple transcriptions per connection)

```python
import asyncio, json, websockets

async def session():
    async with websockets.connect("ws://192.168.86.101:9000/ws") as ws:
        for audio_path in ["chunk1.wav", "chunk2.wav"]:
            with open(audio_path, "rb") as f:
                await ws.send(f.read())
            await ws.send(json.dumps({"action": "transcribe"}))

            # Collect until done
            try:
                async for msg in ws:
                    data = json.loads(msg)
                    if data["type"] == "done":
                        print(f"{audio_path}: {data['text']}")
                        break
            except websockets.exceptions.ConnectionClosed:
                break

        # Close session
        await ws.send(json.dumps({"action": "end"}))

asyncio.run(session())
```

---

## Configuration

Set via environment variables in `docker-compose.yml` or at `docker compose up` time.

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL` | `large-v3` | Model size: `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3` |
| `WHISPER_DEVICE` | `cuda` | `cuda` or `cpu` |

**Switch to a smaller/faster model**
```bash
WHISPER_MODEL=medium docker compose up -d
```

**Use CPU (no GPU)**
```bash
WHISPER_DEVICE=cpu docker compose up -d
```

---

## Deployment

**Start**
```bash
cd ~/GITHUB/whisper-spark-api
docker compose up -d
```

**Stop**
```bash
docker compose down
```

**View logs**
```bash
docker compose logs -f
```

**Rebuild after code changes**
```bash
docker compose up --build -d
```

> **Note:** The model is cached in the `whisper-models` Docker volume. Do **not** run `docker compose down -v` — this deletes the cache and forces a ~3GB re-download on next start.

**First startup** downloads and loads the model (~1–2 min). Subsequent restarts load from cache (~15s).

---

## GPU monitoring

SSH to the Spark and run:
```bash
watch -n1 nvidia-smi
```

GPU utilization will spike to ~80–100% during transcription and return to idle when done.

---

## Model size tradeoffs

| Model | VRAM | Speed | Accuracy |
|-------|------|-------|----------|
| `tiny` | ~1 GB | ~32x | Low |
| `base` | ~1 GB | ~16x | Low |
| `small` | ~2 GB | ~6x | Medium |
| `medium` | ~5 GB | ~2x | Good |
| `large-v2` | ~10 GB | 1x | High |
| `large-v3` | ~10 GB | 1x | Best |

The GB10 has unified memory (CPU+GPU shared pool), so VRAM is not a hard limit.

---

## Concurrency

Requests are serialized — simultaneous calls queue behind each other while the GPU processes one at a time. This is intentional: the GPU is faster to run one job at full utilization than to context-switch between jobs.
