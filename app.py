"""Web UI: upload an invoice image and get the validated JSON back.

Local:   uvicorn app:app --reload
Workers: uv run pywrangler deploy
"""

import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response

from extractor import ExtractionError, extract_invoice, get_model_name
from models import InvoiceResult
from validator import build_result

load_dotenv()

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Invoice Extractor")


def _env_value(env: object, name: str) -> str | None:
    value = getattr(env, name, None)
    if value is None and hasattr(env, "get"):
        value = env.get(name)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


@app.middleware("http")
async def inject_worker_secrets(request: Request, call_next):
    """Map Cloudflare Worker bindings onto os.environ for the existing extractor."""
    env = request.scope.get("env")
    if env is not None:
        api_key = _env_value(env, "GROQ_API_KEY")
        if api_key:
            os.environ["GROQ_API_KEY"] = api_key
        model = _env_value(env, "GROQ_MODEL")
        if model:
            os.environ["GROQ_MODEL"] = model
    return await call_next(request)


async def _asset(request: Request, path: str) -> Response | None:
    env = request.scope.get("env")
    assets = getattr(env, "ASSETS", None) if env is not None else None
    if assets is None:
        return None
    resp = await assets.fetch(f"https://assets.local/{path.lstrip('/')}")
    body = await resp.bytes()
    return Response(content=body, status_code=resp.status, headers=dict(resp.headers))


@app.get("/")
async def index(request: Request) -> Response:
    asset = await _asset(request, "index.html")
    if asset is not None:
        return asset
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/extract", response_model=InvoiceResult)
async def extract(file: UploadFile = File(...)) -> InvoiceResult:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    suffix = Path(file.filename).suffix or ".bin"
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    model = get_model_name()
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        raw = extract_invoice(tmp_path, model=model)
    except ExtractionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

    return build_result(raw, source_file=file.filename, ai_model=model)


@app.get("/{path:path}")
async def frontend(path: str, request: Request) -> Response:
    """Workers Static Assets catch-all (equivalent of serving ./static)."""
    asset = await _asset(request, path or "index.html")
    if asset is not None:
        return asset
    raise HTTPException(status_code=404, detail="Not found.")


# Cloudflare Python Workers entrypoint (ignored when running under uvicorn).
try:
    from workers import asgi

    Default = asgi.entrypoint(app)
except ImportError:
    Default = None
