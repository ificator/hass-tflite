# Home Assistant Add-on: TFLite Server

A Home Assistant add-on that runs a Python FastAPI server to run TensorFlow Lite model inference via HTTP.

## Features

- Load TFLite models from `/config/tensor_models`
- Invoke API to run inference with TFLite runtime
- Interpreter caching for faster repeated invocations

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

Returns `{"status": "ok"}`.

### Initialize

```http
POST /initialize
Content-Type: application/json
{
  "model": "model.tflite"
}
```

Pre-loads a model into memory to avoid latency on the first `/invoke` call. Returns model metadata:

```json
{
  "model": "model.tflite",
  "inputs": [{ "index": 0, "shape": [1, 224, 224, 3], "dtype": "float32" }],
  "outputs": [{ "index": 0, "shape": [1, 1000], "dtype": "float32" }]
}
```

This is optional - models are automatically loaded on first invoke if not pre-initialized.

### Invoke

```http
POST /invoke
Content-Type: application/json
{
  "model": "model.tflite",
  "input": [[...]]
}
```

For multiple inputs, use `"inputs": [{"index": 0, "data": ..., "dtype": "float32"}, ...]` instead of `"input"`.

Response:

```json
{
  "outputs": [
    {"index": 0, "data": [...], "dtype": "float32", "shape": [...]}
  ]
}
```

## Examples

```bash
BASE=http://local-tflite-server:8000

# Pre-load model (optional, useful at startup)
curl -X POST ${BASE}/initialize \
  -H 'Content-Type: application/json' \
  -d '{"model":"model.tflite"}'

# Invoke (single input)
curl -X POST ${BASE}/invoke \
  -H 'Content-Type: application/json' \
  -d '{"model":"model.tflite","input":[[1,2],[3,4]]}'
```

## Notes

- Models are read from `/config/tensor_models`. Place your `.tflite` model files in this folder within your Home Assistant config directory.
- Interpreters are cached per model (up to 3) to speed up repeated invocations.
- **Inputs must already match the model's expected shapes** (no reshaping or broadcasting performed).

## Development

```bash
# Build image
docker build -t hass-tflite .

# Run for validation (exposes port 8000 on host)
# Note: /config must be mapped as the add-on reads models from /config/tensor_models
docker run --rm -p 8000:8000 -v $PWD/tmp-config:/config hass-tflite

# Windows PowerShell volume path
docker run --rm -p 8000:8000 -v ${PWD}\tmp-config:/config hass-tflite

# Test
curl http://localhost:8000/health
```

## License

MIT
