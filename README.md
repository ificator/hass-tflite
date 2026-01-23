# Home Assistant Add-on: TFLite Server

A Home Assistant add-on that runs a Python FastAPI server to manage TensorFlow Lite models and invoke them via HTTP.

## Features
- ✅ CRUD model files stored in `/data/models`
- ✅ SHA1 hash exposed on read (JSON metadata)
- ✅ Invoke API to run inference with TFLite runtime (fallback echoes input if runtime unavailable)

## Installation
1. Copy this repository into your Home Assistant add-ons folder as `/addons/tflite_server` (or add this repo as a custom add-on repository).
2. In Home Assistant, go to **Settings → Add-ons → Add-on Store**, then the **⋮ menu → Repositories** and add your local repository if needed.
3. Find **TFLite Server** in the add-on list, click **Install**, then **Start**.

## API
Base URL (internal): `http://local-tflite-server:8000`

### Health
```http
GET /health
```

### Models
- **List**: `GET /models`
- **Upload/Update**: `PUT /models/{name}` (binary body or multipart)
- **Read metadata**: `GET /models/{name}` (JSON with sha1)
- **Delete**: `DELETE /models/{name}`

### Invoke
```http
POST /invoke
Content-Type: application/json
{
  "model": "model.tflite",
  "input": [[...]]  // or use "inputs": [{"index":0, "data":..., "dtype":"float32"}]
}
```
Response:
```json
{
  "outputs": [
    {"index": 0, "data": [...], "dtype": "float32", "shape": [..]}
  ]
}
```

## Examples
```bash
BASE=http://local-tflite-server:8000
tflite_model=model.tflite

# Upload model
curl -X PUT --data-binary @${tflite_model} ${BASE}/models/${tflite_model}

# List models
curl ${BASE}/models

# Invoke (single input)
curl -X POST ${BASE}/invoke \
  -H 'Content-Type: application/json' \
  -d '{"model":"'${tflite_model}'","input":[[1,2],[3,4]]}'
```

## Notes
- Models are stored under `/data/models` (persistent across add-on restarts).
- TFLite runtime is attempted via `tflite-runtime`; if unavailable, the server echoes inputs.
- Interpreter caching is used per model to speed up repeated invocations.
- **Inputs must already match the model's expected shapes** (no reshaping or broadcasting performed).

## Development
```bash
# Build image
docker build -t hass-tflite .

# Run for validation (exposes port 8000 on host)
docker run --rm -p 8000:8000 -v $PWD/tmp-data:/data hass-tflite

# Windows PowerShell volume path
docker run --rm -p 8000:8000 -v ${PWD}\tmp-data:/data hass-tflite

# Test
curl http://localhost:8000/health
```

## License
MIT
