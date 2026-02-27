/**
 * app.js — SkyWatch Frontend Logic
 * 
 * Connects to a Flask backend running predict.py
 * Backend endpoint: POST /predict  (multipart/form-data, field: "image")
 * Response: { detections: [{label, confidence, box: [x1,y1,x2,y2]}] }
 */

const API_URL = 'http://127.0.0.1:5000/predict';

// ── DOM refs ────────────────────────────────────────────────
const uploadZone     = document.getElementById('uploadZone');
const fileInput      = document.getElementById('fileInput');
const browseBtn      = document.getElementById('browseBtn');
const resultContainer= document.getElementById('resultContainer');
const previewImg     = document.getElementById('previewImg');
const canvas         = document.getElementById('detectionCanvas');
const imgMeta        = document.getElementById('imgMeta');
const analyzingState = document.getElementById('analyzingState');
const detectionsList = document.getElementById('detectionsList');
const statsRow       = document.getElementById('statsRow');
const noDetection    = document.getElementById('noDetection');
const resultCount    = document.getElementById('resultCount');
const birdCount      = document.getElementById('birdCount');
const droneCount     = document.getElementById('droneCount');
const avgConf        = document.getElementById('avgConf');
const resetBtn       = document.getElementById('resetBtn');
const downloadBtn    = document.getElementById('downloadBtn');

const ctx = canvas.getContext('2d');

// ── Colors ──────────────────────────────────────────────────
const COLORS = {
  drone: '#ff4f6a',
  bird:  '#76ff6a'
};
const ICONS = { drone: '🚁', bird: '🐦' };

// ── Drag & Drop ──────────────────────────────────────────────
uploadZone.addEventListener('dragover', e => {
  e.preventDefault();
  uploadZone.classList.add('dragover');
});
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  uploadZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) handleFile(file);
});
uploadZone.addEventListener('click', () => fileInput.click());
browseBtn.addEventListener('click', e => { e.stopPropagation(); fileInput.click(); });
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

// ── Reset ────────────────────────────────────────────────────
resetBtn.addEventListener('click', () => {
  resultContainer.classList.remove('visible');
  uploadZone.style.display = 'block';
  fileInput.value = '';
  clearCanvas();
  detectionsList.innerHTML = '';
  hideAll();
});

// ── Download annotated image ──────────────────────────────────
downloadBtn.addEventListener('click', () => {
  const merged = mergeImageAndCanvas();
  const a = document.createElement('a');
  a.href = merged;
  a.download = 'skywatch_result.jpg';
  a.click();
});

// ── Handle file ──────────────────────────────────────────────
function handleFile(file) {
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  previewImg.onload = () => {
    uploadZone.style.display = 'none';
    resultContainer.classList.add('visible');
    showAnalyzing();
    imgMeta.textContent = `${file.name}  ·  ${(file.size / 1024).toFixed(1)} KB  ·  ${previewImg.naturalWidth}×${previewImg.naturalHeight}px`;
    resizeCanvas();
    runDetection(file);
  };
}

// ── Run detection via Flask API ───────────────────────────────
async function runDetection(file) {
  const formData = new FormData();
  formData.append('image', file);

  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) throw new Error(`Server error: ${response.status}`);
    const data = await response.json();
    hideAnalyzing();
    renderResults(data.detections);

  } catch (err) {
    console.error('API Error:', err);
    hideAnalyzing();

    // ── DEMO MODE (no backend) ───────────────────────────────
    // Generates fake detections so the UI is still testable
    const demoDetections = generateDemoDetections();
    renderResults(demoDetections, true);
  }
}

// Demo detections for testing without a backend
function generateDemoDetections() {
  const w = previewImg.naturalWidth;
  const h = previewImg.naturalHeight;
  return [
    { label: 'drone', confidence: 0.91, box: [w*0.1, h*0.1, w*0.35, h*0.4] },
    { label: 'bird',  confidence: 0.78, box: [w*0.6, h*0.2, w*0.85, h*0.55] },
  ];
}

// ── Render results ───────────────────────────────────────────
function renderResults(detections, isDemo = false) {
  detectionsList.innerHTML = '';
  clearCanvas();

  if (!detections || detections.length === 0) {
    noDetection.classList.add('visible');
    resultCount.textContent = '0 objects';
    return;
  }

  let birds = 0, drones = 0, totalConf = 0;

  detections.forEach((det, i) => {
    const label = det.label.toLowerCase();
    const conf  = det.confidence;
    const box   = det.box;
    totalConf += conf;
    if (label === 'bird') birds++;
    else drones++;

    // Draw box on canvas
    setTimeout(() => drawBox(box, label, conf), i * 120);

    // Create list item
    const item = document.createElement('div');
    item.className = `detection-item ${label}`;
    item.style.animationDelay = `${i * 0.08}s`;
    item.innerHTML = `
      <span class="det-icon">${ICONS[label] || '◉'}</span>
      <div class="det-info">
        <div class="det-label ${label}">${label.toUpperCase()}</div>
        <div class="det-box">box: [${box.map(v => Math.round(v)).join(', ')}]</div>
        <div class="conf-bar-wrap">
          <div class="conf-bar ${label}" style="width: ${conf * 100}%"></div>
        </div>
      </div>
      <span class="det-conf ${label}">${(conf * 100).toFixed(1)}%</span>
    `;
    detectionsList.appendChild(item);
  });

  // Update stats
  birdCount.textContent  = birds;
  droneCount.textContent = drones;
  avgConf.textContent    = `${((totalConf / detections.length) * 100).toFixed(0)}%`;
  resultCount.textContent = `${detections.length} object${detections.length > 1 ? 's' : ''} found`;

  statsRow.classList.add('visible');
  downloadBtn.classList.add('visible');

  if (isDemo) {
    const notice = document.createElement('div');
    notice.style.cssText = 'text-align:center;font-family:var(--font-mono);font-size:0.65rem;color:var(--muted);padding:8px;';
    notice.textContent = '⚠ Demo mode — connect Flask backend for real inference';
    detectionsList.appendChild(notice);
  }
}

// ── Canvas drawing ───────────────────────────────────────────
function resizeCanvas() {
  const rect = previewImg.getBoundingClientRect();
  canvas.width  = rect.width;
  canvas.height = rect.height;
}

function drawBox(box, label, conf) {
  resizeCanvas();
  const scaleX = canvas.width  / previewImg.naturalWidth;
  const scaleY = canvas.height / previewImg.naturalHeight;

  const [x1, y1, x2, y2] = box;
  const sx = x1 * scaleX, sy = y1 * scaleY;
  const sw = (x2 - x1) * scaleX, sh = (y2 - y1) * scaleY;

  const color = COLORS[label] || '#00e5ff';

  // Glow effect
  ctx.shadowColor = color;
  ctx.shadowBlur  = 12;

  // Box
  ctx.strokeStyle = color;
  ctx.lineWidth   = 2;
  ctx.strokeRect(sx, sy, sw, sh);

  // Corner accents
  ctx.shadowBlur = 0;
  const cs = 12;
  ctx.lineWidth = 3;
  [[sx, sy], [sx+sw, sy], [sx, sy+sh], [sx+sw, sy+sh]].forEach(([cx, cy], i) => {
    ctx.beginPath();
    const dx = i % 2 === 0 ? cs : -cs;
    const dy = i < 2 ? cs : -cs;
    ctx.moveTo(cx, cy + dy); ctx.lineTo(cx, cy); ctx.lineTo(cx + dx, cy);
    ctx.stroke();
  });

  // Label background
  const text  = `${label.toUpperCase()}  ${(conf*100).toFixed(0)}%`;
  ctx.font    = 'bold 11px DM Mono, monospace';
  const tw    = ctx.measureText(text).width;
  const pad   = 6;
  const lh    = 20;
  const lx    = sx;
  const ly    = sy > lh + pad ? sy - lh - pad : sy + sh + pad;

  ctx.fillStyle = color;
  ctx.fillRect(lx, ly, tw + pad * 2, lh);
  ctx.fillStyle = '#060810';
  ctx.fillText(text, lx + pad, ly + lh - 5);
}

function clearCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

// ── Merge image + canvas for download ────────────────────────
function mergeImageAndCanvas() {
  const offscreen = document.createElement('canvas');
  offscreen.width  = previewImg.naturalWidth;
  offscreen.height = previewImg.naturalHeight;
  const oc = offscreen.getContext('2d');
  oc.drawImage(previewImg, 0, 0);

  const scaleX = previewImg.naturalWidth  / canvas.width;
  const scaleY = previewImg.naturalHeight / canvas.height;
  oc.scale(scaleX, scaleY);
  oc.drawImage(canvas, 0, 0);
  return offscreen.toDataURL('image/jpeg', 0.95);
}

// ── UI State helpers ─────────────────────────────────────────
function showAnalyzing() {
  analyzingState.classList.add('visible');
  noDetection.classList.remove('visible');
  statsRow.classList.remove('visible');
  downloadBtn.classList.remove('visible');
  detectionsList.innerHTML = '';
  resultCount.textContent = '—';
}
function hideAnalyzing() { analyzingState.classList.remove('visible'); }
function hideAll() {
  analyzingState.classList.remove('visible');
  statsRow.classList.remove('visible');
  noDetection.classList.remove('visible');
  downloadBtn.classList.remove('visible');
  resultCount.textContent = '—';
}

// Resize canvas when window resizes
window.addEventListener('resize', () => {
  if (resultContainer.classList.contains('visible')) resizeCanvas();
});
