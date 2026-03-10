import argparse
import asyncio
import json
import os
import tempfile
from io import BytesIO

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3")
DEVICE = os.environ.get("WHISPER_DEVICE", "cuda")
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "float16")

app = FastAPI()

model: WhisperModel = None
model_lock = asyncio.Lock()


@app.on_event("startup")
async def load_model():
    global model
    model = WhisperModel(WHISPER_MODEL, device=DEVICE, compute_type=COMPUTE_TYPE)
    print(f"Loaded model={WHISPER_MODEL} device={DEVICE} compute_type={COMPUTE_TYPE}")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": WHISPER_MODEL,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
    }


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form(None),
    task: str = Form("transcribe"),
):
    audio_bytes = await file.read()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename or "audio.wav")[1] or ".wav")
    try:
        tmp.write(audio_bytes)
        tmp.flush()
        tmp.close()

        async with model_lock:
            segments_iter, info = model.transcribe(
                tmp.name,
                language=language,
                task=task,
            )
            segments = [
                {"start": seg.start, "end": seg.end, "text": seg.text}
                for seg in segments_iter
            ]

        full_text = " ".join(s["text"].strip() for s in segments)
        return {
            "text": full_text,
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
            "segments": segments,
        }
    finally:
        os.unlink(tmp.name)


@app.websocket("/ws")
async def websocket_transcribe(websocket: WebSocket):
    await websocket.accept()
    audio_buffer = BytesIO()

    async def run_transcription():
        audio_buffer.seek(0)
        audio_bytes = audio_buffer.read()
        if not audio_bytes:
            return [], None

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        try:
            tmp.write(audio_bytes)
            tmp.flush()
            tmp.close()

            async with model_lock:
                segments_iter, info = model.transcribe(tmp.name)
                segments = [
                    {"start": seg.start, "end": seg.end, "text": seg.text}
                    for seg in segments_iter
                ]
            return segments, info
        finally:
            os.unlink(tmp.name)

    try:
        while True:
            message = await websocket.receive()

            if "bytes" in message and message["bytes"]:
                audio_buffer.write(message["bytes"])

            elif "text" in message and message["text"]:
                try:
                    data = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue

                action = data.get("action")

                if action in ("transcribe", "end"):
                    segments, info = await run_transcription()

                    for seg in segments:
                        await websocket.send_text(json.dumps({
                            "type": "segment",
                            "start": seg["start"],
                            "end": seg["end"],
                            "text": seg["text"],
                        }))

                    full_text = " ".join(s["text"].strip() for s in segments)
                    await websocket.send_text(json.dumps({
                        "type": "done",
                        "text": full_text,
                        "language": info.language if info else None,
                    }))

                    # Reset buffer for next cycle
                    audio_buffer = BytesIO()

                    if action == "end":
                        await websocket.close()
                        return

    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port)
