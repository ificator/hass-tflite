# Home Assistant Add-on: TFLite Server

A Home Assistant add-on that runs a Python FastAPI server to run TensorFlow Lite model inference via HTTP.

## Features
- Upload model files to `/data/models`
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

### Upload Model
```http
PUT /models/{name}
```
Upload a model file (binary body or multipart). Returns `{"name": "...", "size": ...}`.

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
tflite_model=model.tflite

# Upload model
curl -X PUT --data-binary @${tflite_model} ${BASE}/models/${tflite_model}

# Invoke (single input)
curl -X POST ${BASE}/invoke \
  -H 'Content-Type: application/json' \
  -d '{"model":"'${tflite_model}'","input":[[1,2],[3,4]]}'
```

## Notes
- Models are stored under `/data/models` (persistent across add-on restarts).
- Interpreters are cached per model (up to 3) to speed up repeated invocations.
- **Inputs must already match the model's expected shapes** (no reshaping or broadcasting performed).

## Development

### Using the Home Assistant Devcontainer (Recommended)

This repository includes a VS Code devcontainer configuration that provides a full Home Assistant Supervisor environment for add-on development.

**Prerequisites:**
- [VS Code](https://code.visualstudio.com/) with the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
- [Docker](https://www.docker.com/)

**Getting Started:**

1. Open the repository folder in VS Code
2. When prompted "Reopen in Container", click **Reopen in Container** (or use the command palette: `Dev Containers: Reopen in Container`)
3. Wait for the container to build and start
4. Once inside the devcontainer, Home Assistant will be available at `http://localhost:7123`

**Developing the Add-on:**

- The add-on source is mounted at `/mnt/supervisor/addons/local/hass-tflite`
- Install the add-on from the Home Assistant UI: **Settings → Add-ons → Add-on Store → Local add-ons** (click the refresh button if needed)
- After making code changes, rebuild the add-on from the Add-on info page
- View add-on logs in the Home Assistant UI for debugging

**Port Mappings:**
| Host Port | Container Port | Service |
|-----------|----------------|---------|
| 7123      | 8123           | Home Assistant UI |
| 7357      | 4357           | Observer |

### Manual Docker Build

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