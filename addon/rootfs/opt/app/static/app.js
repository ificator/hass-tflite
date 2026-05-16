const API_BASE = 'api';

// DOM elements
const healthStatus = document.getElementById('health-status');
const statusText = healthStatus.querySelector('.status-text');
const uploadForm = document.getElementById('upload-form');
const modelFileInput = document.getElementById('model-file');
const modelNameInput = document.getElementById('model-name');
const uploadBtn = document.getElementById('upload-btn');
const uploadStatus = document.getElementById('upload-status');
const modelsLoading = document.getElementById('models-loading');
const modelsEmpty = document.getElementById('models-empty');
const modelsList = document.getElementById('models-list');

// Format file size
function formatSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

// Format uptime
function formatUptime(seconds) {
    const d = Math.floor(seconds / 86400);
    const h = Math.floor((seconds % 86400) / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    if (d > 0) return `${d}d ${h}h ${m}m`;
    if (h > 0) return `${h}h ${m}m ${s}s`;
    if (m > 0) return `${m}m ${s}s`;
    return `${s}s`;
}

// Fetch and display server stats
async function loadStats() {
    try {
        const response = await fetch(`${API_BASE}/stats`);
        if (!response.ok) return;
        const data = await response.json();

        document.getElementById('stat-uptime').textContent =
            formatUptime(data.uptime_seconds);
        document.getElementById('stat-invokes').textContent =
            data.invoke_count.toLocaleString();
        document.getElementById('stat-cached').textContent =
            data.cached_models;

        if (data.memory_rss_bytes !== undefined) {
            document.getElementById('stat-memory').textContent =
                formatSize(data.memory_rss_bytes);
        }
        if (data.cpu_percent !== undefined) {
            document.getElementById('stat-cpu').textContent =
                data.cpu_percent.toFixed(1) + '%';
        }
    } catch (e) {
        // ignore – stats are best-effort
    }
}

// Check server health
async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        if (response.ok) {
            healthStatus.className = 'status status-ok';
            statusText.textContent = 'Server is running';
        } else {
            healthStatus.className = 'status status-error';
            statusText.textContent = 'Server error';
        }
    } catch (error) {
        healthStatus.className = 'status status-error';
        statusText.textContent = 'Cannot connect';
    }
}

// Load models list
async function loadModels() {
    modelsLoading.classList.remove('hidden');
    modelsEmpty.classList.add('hidden');
    modelsList.classList.add('hidden');

    try {
        const response = await fetch(`${API_BASE}/models`);
        if (!response.ok) throw new Error('Failed to load models');

        const models = await response.json();
        modelsLoading.classList.add('hidden');

        if (models.length === 0) {
            modelsEmpty.classList.remove('hidden');
        } else {
            modelsList.innerHTML = '';
            models.forEach(model => {
                const li = document.createElement('li');
                li.className = 'model-item';
                li.innerHTML = `
                    <div class="model-info">
                        <span class="model-name">${escapeHtml(model.name)}</span>
                        <span class="model-size">${formatSize(model.size)}</span>
                        <span class="model-hash">${escapeHtml(model.sha256)}</span>
                    </div>
                    <button class="delete-btn" data-model="${escapeHtml(model.name)}">Delete</button>
                `;
                modelsList.appendChild(li);
            });
            modelsList.classList.remove('hidden');
        }
    } catch (error) {
        modelsLoading.textContent = 'Failed to load models';
    }
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Show upload status message
function showUploadMessage(message, isError) {
    uploadStatus.textContent = message;
    uploadStatus.className = `message ${isError ? 'message-error' : 'message-success'}`;
    uploadStatus.classList.remove('hidden');
    setTimeout(() => uploadStatus.classList.add('hidden'), 5000);
}

// Handle file upload
uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    const file = modelFileInput.files[0];
    if (!file) return;

    // Validate file extension
    if (!file.name.endsWith('.tflite')) {
        showUploadMessage('Only .tflite files are allowed', true);
        return;
    }

    // Determine model name
    let modelName = modelNameInput.value.trim();
    if (!modelName) {
        modelName = file.name;
    }
    if (!modelName.endsWith('.tflite')) {
        modelName += '.tflite';
    }

    uploadBtn.disabled = true;
    uploadBtn.textContent = 'Uploading...';

    try {
        const response = await fetch(`${API_BASE}/models/${encodeURIComponent(modelName)}`, {
            method: 'PUT',
            body: file,
        });

        if (response.ok) {
            const result = await response.json();
            showUploadMessage(`Uploaded ${result.name} (${formatSize(result.size)})`, false);
            modelFileInput.value = '';
            modelNameInput.value = '';
            loadModels();
        } else {
            const error = await response.json();
            showUploadMessage(error.detail || 'Upload failed', true);
        }
    } catch (error) {
        showUploadMessage('Upload failed: ' + error.message, true);
    } finally {
        uploadBtn.disabled = false;
        uploadBtn.textContent = 'Upload Model';
    }
});

// Handle model deletion
modelsList.addEventListener('click', async (e) => {
    if (!e.target.classList.contains('delete-btn')) return;

    const modelName = e.target.dataset.model;
    if (!confirm(`Delete model "${modelName}"?`)) return;

    e.target.disabled = true;
    e.target.textContent = 'Deleting...';

    try {
        const response = await fetch(`${API_BASE}/models/${encodeURIComponent(modelName)}`, {
            method: 'DELETE',
        });

        if (response.ok) {
            loadModels();
        } else {
            const error = await response.json();
            alert(error.detail || 'Delete failed');
            e.target.disabled = false;
            e.target.textContent = 'Delete';
        }
    } catch (error) {
        alert('Delete failed: ' + error.message);
        e.target.disabled = false;
        e.target.textContent = 'Delete';
    }
});

// Initial load
checkHealth();
loadModels();
loadStats();

// Periodically check health and stats
setInterval(checkHealth, 30000);
setInterval(loadStats, 5000);
