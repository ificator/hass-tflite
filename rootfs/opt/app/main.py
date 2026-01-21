import hashlib
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

try:  # Prefer tflite-runtime
    import tflite_runtime.interpreter as tflite  # type: ignore
except ImportError:  # Fallback to TF if available
    try:
        import tensorflow.lite as tflite  # type: ignore
    except ImportError:
        tflite = None  # type: ignore

import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("tflite-server")

MODEL_DIR = Path("/data/models").resolve()
MODEL_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="TFLite Server", version="0.1.0")

SUPERVISOR_TOKEN = os.environ.get("SUPERVISOR_TOKEN")

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if not SUPERVISOR_TOKEN:
        return await call_next(request)
    auth_header = request.headers.get("Authorization")
    sup_header = request.headers.get("X-Supervisor-Token")
    token = None
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ", 1)[1]
    elif sup_header:
        token = sup_header
    if token != SUPERVISOR_TOKEN:
        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
    return await call_next(request)


def compute_sha1(path: Path, chunk_size: int = 8192) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_model_path(name: str) -> Path:
    path = (MODEL_DIR / name).resolve()
    if MODEL_DIR not in path.parents and path != MODEL_DIR:
        raise HTTPException(status_code=400, detail="Invalid model name")
    return path


class ModelMeta(BaseModel):
    name: str
    size: int
    sha1: str
    modified: datetime


class InvokeInput(BaseModel):
    index: Optional[int] = Field(default=None, description="Input tensor index")
    data: Any = Field(..., description="Tensor data as nested lists")
    dtype: Optional[str] = Field(default=None, description="Numpy dtype, e.g., float32")


class InvokeRequest(BaseModel):
    model: str = Field(..., description="Model filename")
    input: Optional[Any] = Field(default=None, description="Single input tensor data")
    inputs: Optional[List[InvokeInput]] = Field(default=None, description="Multiple input tensors")


class InvokeOutput(BaseModel):
    index: int
    data: Any
    dtype: str
    shape: List[int]


_INTERPRETERS: Dict[Path, Dict[str, Any]] = {}
_INTERPRETERS_LOCK = threading.Lock()
_MAX_CACHE_SIZE = 3


def _get_cache_key(path: Path) -> str:
    st = path.stat()
    return f"{st.st_mtime_ns}-{st.st_size}"


def _ensure_interpreter(path: Path):
    if tflite is None:
        return None
    key = _get_cache_key(path)
    with _INTERPRETERS_LOCK:
        cached = _INTERPRETERS.get(path)
        if cached and cached["key"] == key:
            return cached["interpreter"], cached["lock"]
        # Evict if cache too big
        if len(_INTERPRETERS) >= _MAX_CACHE_SIZE:
            _INTERPRETERS.pop(next(iter(_INTERPRETERS)))
        interpreter = tflite.Interpreter(model_path=str(path))
        interpreter.allocate_tensors()
        lock = threading.Lock()
        _INTERPRETERS[path] = {"key": key, "interpreter": interpreter, "lock": lock}
        return interpreter, lock


def _to_numpy(data: Any, dtype: Optional[str]):
    if dtype:
        try:
            return np.array(data, dtype=dtype)
        except TypeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid dtype '{dtype}': {e}")
    return np.array(data)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models", response_model=List[ModelMeta])
def list_models():
    models = []
    for path in MODEL_DIR.glob("*"):
        if path.is_file():
            models.append(
                ModelMeta(
                    name=path.name,
                    size=path.stat().st_size,
                    sha1=compute_sha1(path),
                    modified=datetime.fromtimestamp(path.stat().st_mtime),
                )
            )
    return models


@app.put("/models/{model_name}", response_model=ModelMeta)
async def put_model(model_name: str, request: Request, file: UploadFile = File(default=None)):
    path = safe_model_path(model_name)
    if file is not None:
        content = await file.read()
    else:
        content = await request.body()
    if not content:
        raise HTTPException(status_code=400, detail="Empty model file")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(content)
    meta = ModelMeta(
        name=path.name,
        size=path.stat().st_size,
        sha1=compute_sha1(path),
        modified=datetime.fromtimestamp(path.stat().st_mtime),
    )
    return meta


@app.get("/models/{model_name}")
async def get_model(model_name: str, request: Request):
    path = safe_model_path(model_name)
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Model not found")
    sha1 = compute_sha1(path)
    meta = ModelMeta(
        name=path.name,
        size=path.stat().st_size,
        sha1=sha1,
        modified=datetime.fromtimestamp(path.stat().st_mtime),
    ).dict()
    return JSONResponse(meta)


@app.delete("/models/{model_name}")
def delete_model(model_name: str):
    path = safe_model_path(model_name)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    path.unlink()
    with _INTERPRETERS_LOCK:
        _INTERPRETERS.pop(path, None)
    return {"status": "deleted"}


@app.post("/invoke", response_model=Dict[str, Any])
def invoke(req: InvokeRequest):
    path = safe_model_path(req.model)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Model not found")

    if tflite is None:
        # Fallback: echo input
        input_data = req.input if req.input is not None else (req.inputs[0].data if req.inputs else None)
        if input_data is None:
            raise HTTPException(status_code=400, detail="No input provided")
        arr = _to_numpy(input_data, None)
        return {
            "outputs": [
                {
                    "index": 0,
                    "data": arr.tolist(),
                    "dtype": str(arr.dtype),
                    "shape": list(arr.shape),
                    "note": "tflite runtime not available; returning input"
                }
            ]
        }

    interpreter, lock = _ensure_interpreter(path)
    if interpreter is None:
        raise HTTPException(status_code=500, detail="Failed to load interpreter")

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Prepare inputs
    if req.inputs:
        inputs = req.inputs
    else:
        if req.input is None:
            raise HTTPException(status_code=400, detail="No input provided")
        inputs = [InvokeInput(index=None, data=req.input)]

    if len(inputs) != len(input_details):
        logger.warning("Inputs count (%s) != model inputs (%s)", len(inputs), len(input_details))

    with lock:
        for i, inp in enumerate(inputs):
            detail = input_details[inp.index if inp.index is not None else i]
            arr = _to_numpy(inp.data, inp.dtype)
            arr = arr.astype(detail["dtype"])
            interpreter.set_tensor(detail["index"], arr)
        interpreter.invoke()
        outputs: List[InvokeOutput] = []
        for od in output_details:
            out_arr = interpreter.get_tensor(od["index"])
            outputs.append(
                InvokeOutput(
                    index=od["index"],
                    data=out_arr.tolist(),
                    dtype=str(out_arr.dtype),
                    shape=list(out_arr.shape),
                )
            )
    return {"outputs": [o.dict() for o in outputs]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), reload=False)
