use axum::{
    body::Bytes,
    extract::{DefaultBodyLimit, Path as AxumPath, State},
    http::StatusCode,
    response::{Html, IntoResponse, Json, Response},
    routing::{get, post, put},
    Router,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    fs,
    io::Read,
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::Instant,
};
use tower_http::services::ServeDir;
use tract_tflite::prelude::*;
use tracing::{info, warn};

const MAX_CACHE_SIZE: usize = 3;

// ---------------------------------------------------------------------------
// Request / Response types
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct InitializeRequest {
    model: String,
}

#[derive(Deserialize)]
struct InvokeRequest {
    model: String,
    input: Option<serde_json::Value>,
    inputs: Option<Vec<InvokeInputItem>>,
}

#[derive(Deserialize)]
struct InvokeInputItem {
    index: Option<usize>,
    data: serde_json::Value,
    #[allow(dead_code)]
    dtype: Option<String>,
}

#[derive(Serialize)]
struct ModelInfo {
    name: String,
    size: u64,
    sha256: String,
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

struct AppError {
    status: StatusCode,
    detail: String,
}

impl AppError {
    fn bad_request(detail: impl Into<String>) -> Self {
        Self { status: StatusCode::BAD_REQUEST, detail: detail.into() }
    }
    fn not_found(detail: impl Into<String>) -> Self {
        Self { status: StatusCode::NOT_FOUND, detail: detail.into() }
    }
    fn internal(detail: impl Into<String>) -> Self {
        Self { status: StatusCode::INTERNAL_SERVER_ERROR, detail: detail.into() }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let body = serde_json::json!({"detail": self.detail});
        (self.status, Json(body)).into_response()
    }
}

// ---------------------------------------------------------------------------
// Application state
// ---------------------------------------------------------------------------

struct CachedModel {
    key: String,
    plan: Arc<TypedRunnableModel<TypedModel>>,
}

struct AppState {
    model_dir: PathBuf,
    static_dir: PathBuf,
    cache: Mutex<HashMap<PathBuf, CachedModel>>,
    start_time: Instant,
    invoke_count: AtomicU64,
}

impl AppState {
    fn new(model_dir: PathBuf, static_dir: PathBuf) -> Self {
        fs::create_dir_all(&model_dir).ok();
        Self {
            model_dir,
            static_dir,
            cache: Mutex::new(HashMap::new()),
            start_time: Instant::now(),
            invoke_count: AtomicU64::new(0),
        }
    }

    fn safe_model_path(&self, name: &str) -> Result<PathBuf, AppError> {
        let path = self.model_dir.join(name);
        let canonical_dir = self.model_dir.canonicalize()
            .map_err(|e| AppError::internal(format!("Model dir error: {e}")))?;

        if let Ok(canonical) = path.canonicalize() {
            if !canonical.starts_with(&canonical_dir) {
                return Err(AppError::bad_request("Invalid model name"));
            }
        } else if let Some(parent) = path.parent() {
            if let Ok(cp) = parent.canonicalize() {
                if !cp.starts_with(&canonical_dir) {
                    return Err(AppError::bad_request("Invalid model name"));
                }
            }
        }
        Ok(path)
    }

    fn ensure_model(&self, path: &Path) -> Result<Arc<TypedRunnableModel<TypedModel>>, AppError> {
        let key = cache_key(path)?;

        // Fast path – cache hit
        {
            let cache = self.cache.lock().unwrap();
            if let Some(cached) = cache.get(path) {
                if cached.key == key {
                    return Ok(Arc::clone(&cached.plan));
                }
            }
        }

        // Slow path – load outside lock
        info!("Loading model: {}", path.display());
        let model = tract_tflite::tflite()
            .model_for_path(path)
            .map_err(|e| AppError::internal(format!("Failed to load model: {e}")))?
            .into_optimized()
            .map_err(|e| AppError::internal(format!("Failed to optimize model: {e}")))?
            .into_runnable()
            .map_err(|e| AppError::internal(format!("Failed to create runnable: {e}")))?;

        let plan = Arc::new(model);

        {
            let mut cache = self.cache.lock().unwrap();
            if cache.len() >= MAX_CACHE_SIZE && !cache.contains_key(path) {
                let evict = cache.keys().next().unwrap().clone();
                cache.remove(&evict);
            }
            cache.insert(path.to_path_buf(), CachedModel {
                key,
                plan: Arc::clone(&plan),
            });
        }

        Ok(plan)
    }

    fn evict_model(&self, path: &Path) {
        let mut cache = self.cache.lock().unwrap();
        cache.remove(path);
    }
}

// ---------------------------------------------------------------------------
// Route handlers
// ---------------------------------------------------------------------------

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({"status": "ok"}))
}

async fn stats(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let uptime_secs = state.start_time.elapsed().as_secs_f64();
    let invoke_count = state.invoke_count.load(Ordering::Relaxed);
    let cached_models = state.cache.lock().unwrap().len();

    let mut resp = serde_json::json!({
        "uptime_seconds": uptime_secs,
        "invoke_count": invoke_count,
        "cached_models": cached_models,
    });

    // Memory from /proc/self/status (Linux)
    if let Ok(status) = fs::read_to_string("/proc/self/status") {
        for line in status.lines() {
            if line.starts_with("VmRSS:") {
                if let Some(kb) = line.split_whitespace().nth(1).and_then(|v| v.parse::<u64>().ok()) {
                    resp["memory_rss_bytes"] = (kb * 1024).into();
                }
            } else if line.starts_with("VmSize:") {
                if let Some(kb) = line.split_whitespace().nth(1).and_then(|v| v.parse::<u64>().ok()) {
                    resp["memory_vms_bytes"] = (kb * 1024).into();
                }
            }
        }
    }

    // CPU times from /proc/self/stat (Linux)
    if let Ok(stat) = fs::read_to_string("/proc/self/stat") {
        let parts: Vec<&str> = stat.split_whitespace().collect();
        if parts.len() > 14 {
            let ticks_per_sec = 100.0_f64;
            if let (Ok(utime), Ok(stime)) = (
                parts[13].parse::<f64>(),
                parts[14].parse::<f64>(),
            ) {
                let user_s = utime / ticks_per_sec;
                let sys_s = stime / ticks_per_sec;
                resp["cpu_user_seconds"] = user_s.into();
                resp["cpu_system_seconds"] = sys_s.into();
                if uptime_secs > 0.0 {
                    resp["cpu_percent"] =
                        (((user_s + sys_s) / uptime_secs) * 100.0).into();
                }
            }
        }
    }

    Json(resp)
}

async fn initialize(
    State(state): State<Arc<AppState>>,
    Json(req): Json<InitializeRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    let path = state.safe_model_path(&req.model)?;
    if !path.exists() {
        return Err(AppError::not_found("Model not found"));
    }

    let plan = state.ensure_model(&path)?;
    let model = plan.model();

    let input_outlets = model.input_outlets()
        .map_err(|e| AppError::internal(e.to_string()))?;
    let inputs: Vec<serde_json::Value> = input_outlets.iter()
        .enumerate()
        .map(|(i, outlet)| {
            let fact = model.input_fact(i).unwrap();
            serde_json::json!({
                "index": outlet.node,
                "shape": fact_shape(fact),
                "dtype": datum_type_to_str(fact.datum_type),
            })
        })
        .collect();

    let output_outlets = model.output_outlets()
        .map_err(|e| AppError::internal(e.to_string()))?;
    let outputs: Vec<serde_json::Value> = output_outlets.iter()
        .enumerate()
        .map(|(i, outlet)| {
            let fact = model.output_fact(i).unwrap();
            serde_json::json!({
                "index": outlet.node,
                "shape": fact_shape(fact),
                "dtype": datum_type_to_str(fact.datum_type),
            })
        })
        .collect();

    Ok(Json(serde_json::json!({
        "model": req.model,
        "inputs": inputs,
        "outputs": outputs,
    })))
}

async fn invoke(
    State(state): State<Arc<AppState>>,
    Json(req): Json<InvokeRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    let path = state.safe_model_path(&req.model)?;
    if !path.exists() {
        return Err(AppError::not_found("Model not found"));
    }

    let plan = state.ensure_model(&path)?;
    state.invoke_count.fetch_add(1, Ordering::Relaxed);
    let model = plan.model();
    let num_inputs = model.input_outlets()
        .map_err(|e| AppError::internal(e.to_string()))?.len();

    // Normalise user-supplied inputs
    let user_inputs: Vec<InvokeInputItem> = if let Some(inputs) = req.inputs {
        inputs
    } else if let Some(input) = req.input {
        vec![InvokeInputItem { index: None, data: input, dtype: None }]
    } else {
        return Err(AppError::bad_request("No input provided"));
    };

    if user_inputs.len() != num_inputs {
        warn!(
            "Inputs count ({}) != model inputs ({})",
            user_inputs.len(),
            num_inputs
        );
    }

    // Build ordered input tensors
    let mut slots: Vec<Option<TValue>> = vec![None; num_inputs];
    for (i, inp) in user_inputs.iter().enumerate() {
        let idx = inp.index.unwrap_or(i);
        if idx >= num_inputs {
            return Err(AppError::bad_request(format!(
                "Input index {idx} out of range (model has {num_inputs} inputs)"
            )));
        }
        let fact = model
            .input_fact(idx)
            .map_err(|e| AppError::bad_request(format!("Input fact error: {e}")))?;
        let tensor = json_to_tensor(&inp.data, fact.datum_type)?;
        slots[idx] = Some(tensor.into());
    }

    let input_tvec: TVec<TValue> = slots
        .into_iter()
        .enumerate()
        .map(|(i, t)| t.ok_or_else(|| AppError::bad_request(format!("Missing input at index {i}"))))
        .collect::<Result<_, _>>()?;

    // Run inference
    let results = plan
        .run(input_tvec)
        .map_err(|e| AppError::internal(format!("Inference failed: {e}")))?;

    // Build response
    let output_outlets = model.output_outlets()
        .map_err(|e| AppError::internal(e.to_string()))?;
    let output_values: Vec<serde_json::Value> = results
        .iter()
        .enumerate()
        .map(|(i, tv)| {
            let t: &Tensor = &*tv;
            let data = tensor_to_json(t).unwrap_or(serde_json::Value::Null);
            let idx = output_outlets.get(i).map(|o| o.node).unwrap_or(i);
            serde_json::json!({
                "index": idx,
                "data": data,
                "dtype": datum_type_to_str(t.datum_type()),
                "shape": t.shape(),
            })
        })
        .collect();

    Ok(Json(serde_json::json!({"outputs": output_values})))
}

async fn list_models(
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<ModelInfo>>, AppError> {
    let mut models = Vec::new();

    let entries = fs::read_dir(&state.model_dir)
        .map_err(|e| AppError::internal(format!("Cannot read model dir: {e}")))?;

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("tflite") {
            if let Ok(meta) = fs::metadata(&path) {
                models.push(ModelInfo {
                    name: path.file_name().unwrap().to_string_lossy().to_string(),
                    size: meta.len(),
                    sha256: compute_sha256(&path)?,
                });
            }
        }
    }

    models.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(Json(models))
}

async fn put_model(
    State(state): State<Arc<AppState>>,
    AxumPath(model_name): AxumPath<String>,
    body: Bytes,
) -> Result<Json<serde_json::Value>, AppError> {
    if body.is_empty() {
        return Err(AppError::bad_request("Empty model file"));
    }

    let path = state.safe_model_path(&model_name)?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).ok();
    }

    state.evict_model(&path);

    fs::write(&path, &body)
        .map_err(|e| AppError::internal(format!("Failed to write model: {e}")))?;

    Ok(Json(serde_json::json!({
        "name": model_name,
        "size": body.len(),
    })))
}

async fn delete_model(
    State(state): State<Arc<AppState>>,
    AxumPath(model_name): AxumPath<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    let path = state.safe_model_path(&model_name)?;
    if !path.exists() {
        return Err(AppError::not_found("Model not found"));
    }

    state.evict_model(&path);

    fs::remove_file(&path)
        .map_err(|e| AppError::internal(format!("Failed to delete model: {e}")))?;

    Ok(Json(serde_json::json!({
        "name": model_name,
        "deleted": true,
    })))
}

async fn serve_ui(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let index = state.static_dir.join("index.html");
    match fs::read_to_string(&index) {
        Ok(html) => Html(html).into_response(),
        Err(_) => (StatusCode::NOT_FOUND, "UI not found").into_response(),
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

fn cache_key(path: &Path) -> Result<String, AppError> {
    let meta = fs::metadata(path)
        .map_err(|e| AppError::internal(format!("Cannot stat model: {e}")))?;
    let modified = meta
        .modified()
        .map_err(|e| AppError::internal(format!("Cannot get mtime: {e}")))?;
    Ok(format!("{modified:?}-{}", meta.len()))
}

fn compute_sha256(path: &Path) -> Result<String, AppError> {
    let mut file =
        fs::File::open(path).map_err(|e| AppError::internal(format!("Cannot open file: {e}")))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 8192];
    loop {
        let n = file.read(&mut buf).map_err(|e| AppError::internal(e.to_string()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn datum_type_to_str(dt: DatumType) -> &'static str {
    match dt {
        DatumType::F16 => "float16",
        DatumType::F32 => "float32",
        DatumType::F64 => "float64",
        DatumType::I8 => "int8",
        DatumType::I16 => "int16",
        DatumType::I32 => "int32",
        DatumType::I64 => "int64",
        DatumType::U8 => "uint8",
        DatumType::U16 => "uint16",
        DatumType::U32 => "uint32",
        DatumType::U64 => "uint64",
        DatumType::Bool => "bool",
        _ => "unknown",
    }
}

fn fact_shape(fact: &TypedFact) -> Vec<i64> {
    fact.shape
        .as_concrete()
        .map(|dims| dims.iter().map(|&d| d as i64).collect())
        .unwrap_or_default()
}

// -- Tensor ↔ JSON conversion -----------------------------------------------

fn json_to_shape(value: &serde_json::Value) -> Vec<usize> {
    match value {
        serde_json::Value::Array(arr) => {
            let mut shape = vec![arr.len()];
            if let Some(first) = arr.first() {
                if first.is_array() {
                    shape.extend(json_to_shape(first));
                }
            }
            shape
        }
        _ => vec![],
    }
}

fn json_flatten_f64(value: &serde_json::Value) -> Vec<f64> {
    match value {
        serde_json::Value::Number(n) => vec![n.as_f64().unwrap_or(0.0)],
        serde_json::Value::Bool(b) => vec![if *b { 1.0 } else { 0.0 }],
        serde_json::Value::Array(arr) => arr.iter().flat_map(json_flatten_f64).collect(),
        _ => vec![0.0],
    }
}

macro_rules! make_tensor {
    ($flat:expr, $shape:expr, $ty:ty) => {{
        let vals: Vec<$ty> = $flat.iter().map(|&v| v as $ty).collect();
        tract_ndarray::ArrayD::<$ty>::from_shape_vec(tract_ndarray::IxDyn(&$shape), vals)
            .map_err(|e| AppError::bad_request(format!("Shape error: {e}")))
            .map(|a| a.into())
    }};
}

fn json_to_tensor(data: &serde_json::Value, target_dt: DatumType) -> Result<Tensor, AppError> {
    let shape = json_to_shape(data);
    let flat = json_flatten_f64(data);

    // Scalar
    if shape.is_empty() {
        return match target_dt {
            DatumType::F32 => Ok(tensor0(flat.first().copied().unwrap_or(0.0) as f32)),
            DatumType::F64 => Ok(tensor0(flat.first().copied().unwrap_or(0.0))),
            DatumType::I32 => Ok(tensor0(flat.first().copied().unwrap_or(0.0) as i32)),
            DatumType::I64 => Ok(tensor0(flat.first().copied().unwrap_or(0.0) as i64)),
            DatumType::U8 => Ok(tensor0(flat.first().copied().unwrap_or(0.0) as u8)),
            DatumType::I8 => Ok(tensor0(flat.first().copied().unwrap_or(0.0) as i8)),
            _ => Err(AppError::bad_request(format!("Unsupported dtype: {target_dt:?}"))),
        };
    }

    match target_dt {
        DatumType::F32 => make_tensor!(flat, shape, f32),
        DatumType::F64 => make_tensor!(flat, shape, f64),
        DatumType::I8 => make_tensor!(flat, shape, i8),
        DatumType::I16 => make_tensor!(flat, shape, i16),
        DatumType::I32 => make_tensor!(flat, shape, i32),
        DatumType::I64 => make_tensor!(flat, shape, i64),
        DatumType::U8 => make_tensor!(flat, shape, u8),
        DatumType::U16 => make_tensor!(flat, shape, u16),
        DatumType::U32 => make_tensor!(flat, shape, u32),
        DatumType::U64 => make_tensor!(flat, shape, u64),
        _ => Err(AppError::bad_request(format!("Unsupported input dtype: {target_dt:?}"))),
    }
}

macro_rules! tensor_slice_to_json {
    ($tensor:expr, $ty:ty) => {
        $tensor
            .as_slice::<$ty>()
            .map_err(|e| AppError::internal(e.to_string()))?
            .iter()
            .map(|v| serde_json::json!(v))
            .collect::<Vec<_>>()
    };
}

fn tensor_to_json(tensor: &Tensor) -> Result<serde_json::Value, AppError> {
    let flat: Vec<serde_json::Value> = match tensor.datum_type() {
        DatumType::F32 => tensor_slice_to_json!(tensor, f32),
        DatumType::F64 => tensor_slice_to_json!(tensor, f64),
        DatumType::I8 => tensor_slice_to_json!(tensor, i8),
        DatumType::I16 => tensor_slice_to_json!(tensor, i16),
        DatumType::I32 => tensor_slice_to_json!(tensor, i32),
        DatumType::I64 => tensor_slice_to_json!(tensor, i64),
        DatumType::U8 => tensor_slice_to_json!(tensor, u8),
        DatumType::U16 => tensor_slice_to_json!(tensor, u16),
        DatumType::U32 => tensor_slice_to_json!(tensor, u32),
        DatumType::U64 => tensor_slice_to_json!(tensor, u64),
        _ => {
            return Err(AppError::internal(format!(
                "Unsupported output dtype: {:?}",
                tensor.datum_type()
            )));
        }
    };

    Ok(reshape_to_nested(&flat, tensor.shape()))
}

fn reshape_to_nested(flat: &[serde_json::Value], shape: &[usize]) -> serde_json::Value {
    if shape.len() <= 1 {
        return serde_json::Value::Array(flat.to_vec());
    }
    let chunk: usize = shape[1..].iter().product();
    serde_json::Value::Array(
        flat.chunks(chunk)
            .map(|c| reshape_to_nested(c, &shape[1..]))
            .collect(),
    )
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let model_dir = PathBuf::from(
        std::env::var("MODEL_DIR").unwrap_or_else(|_| "/config/models".to_string()),
    );
    let static_dir = PathBuf::from(
        std::env::var("STATIC_DIR").unwrap_or_else(|_| "/opt/app/static".to_string()),
    );
    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8000);

    let state = Arc::new(AppState::new(model_dir, static_dir.clone()));

    let app = Router::new()
        .route("/health", get(health))
        .route("/api/health", get(health))
        .route("/api/stats", get(stats))
        .route("/initialize", post(initialize))
        .route("/api/initialize", post(initialize))
        .route("/invoke", post(invoke))
        .route("/api/invoke", post(invoke))
        .route("/api/models", get(list_models))
        .route("/models/:model_name", put(put_model))
        .route(
            "/api/models/:model_name",
            put(put_model).delete(delete_model),
        )
        .route("/ui", get(serve_ui))
        .nest_service("/ui/static", ServeDir::new(&static_dir))
        .layer(DefaultBodyLimit::max(500 * 1024 * 1024))
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    info!("Starting TFLite Server on {addr}");

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
