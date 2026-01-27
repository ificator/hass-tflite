import hashlib
import logging
import numpy as np
import tflite_runtime.interpreter as tflite
import threading

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("tflite-server")

MODEL_DIR = Path("/config/models").resolve()
MODEL_DIR.mkdir(parents=True, exist_ok=True)

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="TFLite Server", version="0.1.0")

# Mount static files for UI assets
app.mount("/ui/static", StaticFiles(directory=STATIC_DIR), name="static")

class InvokeInput(BaseModel):
    index: Optional[int] = Field(default=None, description="Input tensor index")
    data: Any = Field(..., description="Tensor data as nested lists")
    dtype: Optional[str] = Field(default=None, description="Numpy dtype, e.g., float32")

class InvokeOutput(BaseModel):
    index: int
    data: Any
    dtype: str
    shape: List[int]

class InitializeRequest(BaseModel):
    model: str = Field(..., description="Model filename")

class InvokeRequest(BaseModel):
    model: str = Field(..., description="Model filename")
    input: Optional[Any] = Field(default=None, description="Single input tensor data")
    inputs: Optional[List[InvokeInput]] = Field(default=None, description="Multiple input tensors")

_INTERPRETERS: Dict[Path, Dict[str, Any]] = {}
_INTERPRETERS_LOCK = threading.Lock()
_MAX_CACHE_SIZE = 3

@app.get("/health")
@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.post("/initialize", response_model=Dict[str, Any])
@app.post("/api/initialize", response_model=Dict[str, Any])
def initialize(req: InitializeRequest):
    """Pre-load a model into the cache to avoid latency on first invoke."""
    if tflite is None:
        raise HTTPException(status_code=500, detail="tflite runtime not available")

    path = _safe_model_path(req.model)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Model not found")

    interpreter, _ = _ensure_interpreter(path)
    if interpreter is None:
        raise HTTPException(status_code=500, detail="Failed to load interpreter")

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return {
        "model": req.model,
        "inputs": [{"index": d["index"], "shape": d["shape"].tolist(), "dtype": str(d["dtype"])} for d in input_details],
        "outputs": [{"index": d["index"], "shape": d["shape"].tolist(), "dtype": str(d["dtype"])} for d in output_details],
    }

@app.post("/invoke", response_model=Dict[str, Any])
@app.post("/api/invoke", response_model=Dict[str, Any])
def invoke(req: InvokeRequest):
    if tflite is None:
        raise HTTPException(status_code=500, detail="tflite runtime not available")

    path = _safe_model_path(req.model)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Model not found")

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

@app.get("/api/models", response_model=List[Dict[str, Any]])
def list_models():
    """List all installed models."""
    models = []
    for path in MODEL_DIR.glob("*.tflite"):
        stat = path.stat()
        models.append({
            "name": path.name,
            "size": stat.st_size,
            "sha256": _compute_sha256(path),
        })
    return sorted(models, key=lambda m: m["name"])

@app.put("/models/{model_name}", response_model=Dict[str, Any])
@app.put("/api/models/{model_name}", response_model=Dict[str, Any])
async def put_model_api(model_name: str, request: Request, file: UploadFile = File(default=None)):
    """Upload a model file."""
    path = _safe_model_path(model_name)
    path.parent.mkdir(parents=True, exist_ok=True)

    if file is not None:
        content = await file.read()
    else:
        content = await request.body()

    if not content:
        raise HTTPException(status_code=400, detail="Empty model file")

    with _INTERPRETERS_LOCK:
        _INTERPRETERS.pop(path, None)

    with path.open("wb") as f:
        f.write(content)

    return {"name": path.name, "size": path.stat().st_size}

@app.delete("/api/models/{model_name}", response_model=Dict[str, Any])
def delete_model(model_name: str):
    """Delete a model file."""
    path = _safe_model_path(model_name)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Model not found")

    # Remove from cache if present
    with _INTERPRETERS_LOCK:
        _INTERPRETERS.pop(path, None)

    path.unlink()
    return {"name": model_name, "deleted": True}

@app.get("/ui")
def ui():
    """Serve the main UI page."""
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

# Private helper functions (alphabetical)

def _compute_sha256(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

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

def _get_cache_key(path: Path) -> str:
    st = path.stat()
    return f"{st.st_mtime_ns}-{st.st_size}"

def _safe_model_path(name: str) -> Path:
    path = (MODEL_DIR / name).resolve()
    if MODEL_DIR not in path.parents and path != MODEL_DIR:
        raise HTTPException(status_code=400, detail="Invalid model name")
    return path

def _to_numpy(data: Any, dtype: Optional[str]):
    if dtype:
        try:
            return np.array(data, dtype=dtype)
        except TypeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid dtype '{dtype}': {e}")
    return np.array(data)
