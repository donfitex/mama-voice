from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from mamavoice.config import LANGS
from mamavoice.pipeline import Pipeline
from mamavoice.safety.red_flags import check

app = FastAPI(title="MamaVoice API", version="0.1.0")
_pipe = Pipeline()


class AskRequest(BaseModel):
    text: str = Field(min_length=2, max_length=1000)
    lang: str = "en"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/safety/check")
def safety_check(req: AskRequest):
    """Cheap, model-free endpoint: usable on very weak infrastructure."""
    r = check(req.text, req.text, req.lang)
    return {"escalate": r.escalate, "categories": r.categories}


@app.post("/ask")
def ask(req: AskRequest):
    if req.lang not in LANGS:
        raise HTTPException(400, f"lang must be one of {list(LANGS)}")
    return _pipe.ask_text(req.text, req.lang).__dict__


@app.post("/ask/voice")
async def ask_voice(lang: str = Form("en"), audio: UploadFile = File(...)):
    if lang not in LANGS:
        raise HTTPException(400, f"lang must be one of {list(LANGS)}")
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "in"
        with open(src, "wb") as f:
            shutil.copyfileobj(audio.file, f)
        out = Path(d) / "out.wav"
        res = _pipe.ask_audio(str(src), lang, str(out))
        keep = Path(tempfile.gettempdir()) / f"mv_{abs(hash(res.answer))}.wav"
        shutil.copy(out, keep)
    return FileResponse(keep, media_type="audio/wav", headers={"X-Escalated": str(res.escalated)})
