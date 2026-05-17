// ═══════════════════════════════════════════════════════════
//  services/api.js  –  All HTTP calls to the Flask backend
// ═══════════════════════════════════════════════════════════

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

async function apiFetch(endpoint, options = {}) {
  const url = `${BASE_URL}${endpoint}`;
  try {
    const res  = await fetch(url, options);
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
    return data;
  } catch (err) {
    if (err.name === 'TypeError')
      throw new Error('Cannot reach the Flask server. Is it running?');
    throw err;
  }
}

export const checkHealth  = () => apiFetch('/health');
export const getModelInfo = () => apiFetch('/model-info');

export async function predictImage(imageFile) {
  const form = new FormData();
  form.append('file', imageFile);
  return apiFetch('/predict', { method: 'POST', body: form });
}

export async function predictWithMask(imageFile, maskFile) {
  const form = new FormData();
  form.append('image', imageFile);
  form.append('mask',  maskFile);
  return apiFetch('/predict-with-mask', { method: 'POST', body: form });
}

export const b64ToDataUri = (b64) => `data:image/png;base64,${b64}`;

export function downloadB64Image(b64, filename = 'result.png') {
  const link = document.createElement('a');
  link.href = b64ToDataUri(b64);
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}