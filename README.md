# Home Assistant Add-on: TFLite Server

A Home Assistant add-on that runs a Python FastAPI server to run TensorFlow Lite model inference via HTTP.

## Features

- Load TFLite models from `/config/models`
- Invoke API to run inference with TFLite runtime
- Interpreter caching for faster repeated invocations
- Web UI for model management

## Installation

1. Copy this repository into your Home Assistant add-ons folder as `/addons/tflite_server` (or add this repo as a custom add-on repository).
2. In Home Assistant, go to **Settings → Add-ons → Add-on Store**, then the **⋮ menu → Repositories** and add your local repository if needed.
3. Find **TFLite Server** in the add-on list, click **Install**, then **Start**.

## API

Base URL (internal): `http://local-tflite-server:8000`

### Health

```http
GET /api/health
```

Returns `{"status": "ok"}`.

### Initialize

```http
POST /api/initialize
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
POST /api/invoke
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

## Web UI

The add-on includes a web interface for managing models. Access it from the add-on's info page in Home Assistant by clicking **Open Web UI**.

### Uploading a Model

1. Open the Web UI from the add-on's info page
2. In the **Upload Model** section, click the file input and select a `.tflite` file
3. Optionally enter a custom model name (the original filename is used if left empty)
4. Click **Upload Model**

The uploaded model will appear in the **Installed Models** list and can be used immediately with the `/api/invoke` endpoint.

### Managing Models

The **Installed Models** section displays all models in the `models` directory with their:
- Filename
- File size
- SHA256 hash (for verification)

To delete a model, click the **Delete** button next to it.

## Examples

```bash
BASE=http://local-tflite-server:8000

# Pre-load model (optional, useful at startup)
curl -X POST ${BASE}/api/initialize \
  -H 'Content-Type: application/json' \
  -d '{"model":"model.tflite"}'

# Invoke (single input)
curl -X POST ${BASE}/api/invoke \
  -H 'Content-Type: application/json' \
  -d '{"model":"model.tflite","input":[[1,2],[3,4]]}'
```

## Notes

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
# Note: /config must be mapped as the add-on reads models from /config/tensor_models
docker run --rm -p 8000:8000 -v $PWD/tmp-config:/config hass-tflite

# Windows PowerShell volume path
docker run --rm -p 8000:8000 -v ${PWD}\tmp-config:/config hass-tflite

# Test
curl http://localhost:8000/health
```

## License

MIT
