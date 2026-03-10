import argparse
import asyncio
import json
import os
import tempfile
from io import BytesIO

import uvicorn
import whisper
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3")
DEVICE = os.environ.get("WHISPER_DEVICE", "cuda")

app = FastAPI()

model: whisper.Whisper = None
model_lock = asyncio.Lock()


@app.on_event("startup")
async def load_model():
    global model
    model = whisper.load_model(WHISPER_MODEL, device=DEVICE)
    print(f"Loaded model={WHISPER_MODEL} device={DEVICE}")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": WHISPER_MODEL,
        "device": DEVICE,
    }


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form(None),
    task: str = Form("transcribe"),
):
    audio_bytes = await file.read()
    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(audio_bytes)
        tmp.flush()
        tmp.close()

        async with model_lock:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: model.transcribe(tmp.name, language=language, task=task),
            )

        segments = [
            {"start": s["start"], "end": s["end"], "text": s["text"]}
            for s in result.get("segments", [])
        ]
        duration = segments[-1]["end"] if segments else 0.0
        return {
            "text": result["text"].strip(),
            "language": result.get("language"),
            "language_probability": None,
            "duration": duration,
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
            return [], None, None

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        try:
            tmp.write(audio_bytes)
            tmp.flush()
            tmp.close()

            async with model_lock:
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: model.transcribe(tmp.name),
                )

            segments = [
                {"start": s["start"], "end": s["end"], "text": s["text"]}
                for s in result.get("segments", [])
            ]
            return segments, result.get("language"), result["text"].strip()
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
                    segments, language, full_text = await run_transcription()

                    for seg in segments:
                        await websocket.send_text(json.dumps({
                            "type": "segment",
                            "start": seg["start"],
                            "end": seg["end"],
                            "text": seg["text"],
                        }))

                    await websocket.send_text(json.dumps({
                        "type": "done",
                        "text": full_text or "",
                        "language": language,
                    }))

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
