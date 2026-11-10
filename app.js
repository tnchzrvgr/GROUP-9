// ============================================================
// AI Real-Time Object Detection + Voice Announcement
// Camera (phone-as-webcam) -> COCO-SSD (TensorFlow.js) -> UI -> SpeechSynthesis
// ============================================================

// ---------- DOM ----------
const $ = (id) => document.getElementById(id);
const video = $('video');
const overlay = $('overlay');
const ctx = overlay.getContext('2d');
const videoWrap = $('videoWrap');
const deviceSelect = $('deviceSelect');
const startBtn = $('startBtn');
const stopBtn = $('stopBtn');
const modelBadge = $('modelBadge');
const camBadge = $('camBadge');
const fpsBadge = $('fpsBadge');
const detectList = $('detectList');
const logList = $('logList');
const confRange = $('confRange');
const coolRange = $('coolRange');
const voiceToggle = $('voiceToggle');

// ---------- Settings ----------
const CONFIRM_FRAMES = 3;     // object must be seen in N consecutive detections before announcing
const GONE_AFTER_MS = 2000;   // object must be missing this long before it can be announced again
const MIN_GAP_MS = 800;       // minimum gap between any two spoken announcements
const DETECT_INTERVAL_MS = 100;

let minConfidence = 0.6;
let cooldownMs = 5000;

// ---------- State ----------
let model = null;
let stream = null;
let running = false;
let lastDetectTime = 0;
let lastSpeechTime = 0;
let fpsFrames = 0;
let fpsStart = performance.now();

// per-label tracking: { streak, lastSeen, announced, lastAnnounced }
const tracker = new Map();

// ---------- Badges ----------
function setBadge(el, text, state) {
  el.textContent = text;
  el.className = 'badge ' + state;
}

// ---------- Model ----------
async function loadModel() {
  setBadge(modelBadge, 'Model: loading…', 'wait');
  try {
    await tf.ready();
    model = await cocoSsd.load({ base: 'mobilenet_v2' });
    setBadge(modelBadge, 'Model: COCO-SSD ready', 'on');
  } catch (err) {
    console.error(err);
    setBadge(modelBadge, 'Model: failed to load (check internet)', 'off');
  }
}

// ---------- Camera ----------
async function listCameras() {
  const devices = await navigator.mediaDevices.enumerateDevices();
  const cams = devices.filter((d) => d.kind === 'videoinput');
  const previous = deviceSelect.value;
  deviceSelect.innerHTML = '';
  cams.forEach((cam, i) => {
    const opt = document.createElement('option');
    opt.value = cam.deviceId;
    opt.textContent = cam.label || `Camera ${i + 1}`;
    deviceSelect.appendChild(opt);
  });
  // Prefer a phone-as-webcam source automatically if its name is recognisable
  const phoneMatch = cams.find((c) => /droidcam|iriun|camo|epoccam|obs|virtual|phone|ivcam|continuity/i.test(c.label));
  if (previous && cams.some((c) => c.deviceId === previous)) deviceSelect.value = previous;
  else if (phoneMatch) deviceSelect.value = phoneMatch.deviceId;
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    alert('Camera API not available. Open this page via http://localhost or HTTPS.');
    return;
  }
  stopCamera();
  setBadge(camBadge, 'Camera: connecting…', 'wait');
  try {
    // First call (no deviceId) grants permission so device labels become visible
    const constraints = {
      video: deviceSelect.value
        ? { deviceId: { exact: deviceSelect.value }, width: { ideal: 1280 }, height: { ideal: 720 } }
        : { width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    };
    stream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = stream;
    await video.play();

    await listCameras();
    const activeId = stream.getVideoTracks()[0].getSettings().deviceId;
    if (activeId) deviceSelect.value = activeId;

    const label = stream.getVideoTracks()[0].label || 'camera';
    setBadge(camBadge, `Camera: active (${label})`, 'on');
    videoWrap.classList.add('live');
    startBtn.disabled = true;
    stopBtn.disabled = false;

    // If the phone app disconnects, update status
    stream.getVideoTracks()[0].addEventListener('ended', () => {
      setBadge(camBadge, 'Camera: disconnected', 'off');
      stopCamera();
    });

    // Resize overlay to the real video resolution
    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;

    // Unlock speech synthesis with a user gesture (silent utterance)
    if ('speechSynthesis' in window) speechSynthesis.speak(new SpeechSynthesisUtterance(''));

    running = true;
    requestAnimationFrame(detectLoop);
  } catch (err) {
    console.error(err);
    setBadge(camBadge, 'Camera: ' + (err.name === 'NotAllowedError' ? 'permission denied' : 'not found / busy'), 'off');
  }
}

function stopCamera() {
  running = false;
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null;
  video.srcObject = null;
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  videoWrap.classList.remove('live');
  startBtn.disabled = false;
  stopBtn.disabled = true;
  tracker.clear();
  renderDetections([]);
  setBadge(camBadge, 'Camera: disconnected', 'off');
  setBadge(fpsBadge, '0 FPS', 'off');
}

// ---------- Detection loop ----------
async function detectLoop(now) {
  if (!running) return;
  if (model && video.readyState >= 2 && now - lastDetectTime >= DETECT_INTERVAL_MS) {
    lastDetectTime = now;
    try {
      if (overlay.width !== video.videoWidth) {
        overlay.width = video.videoWidth;
        overlay.height = video.videoHeight;
      }
      const raw = await model.detect(video);
      const preds = raw.filter((p) => p.score >= minConfidence);
      drawBoxes(preds);
      renderDetections(preds);
      updateAnnouncements(preds);
      updateFps();
    } catch (err) {
      console.error('Detection error', err);
    }
  }
  requestAnimationFrame(detectLoop);
}

function updateFps() {
  fpsFrames++;
  const t = performance.now();
  if (t - fpsStart >= 1000) {
    setBadge(fpsBadge, `${fpsFrames} FPS`, 'on');
    fpsFrames = 0;
    fpsStart = t;
  }
}

// ---------- Drawing ----------
function drawBoxes(preds) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  const lw = Math.max(2, overlay.width / 320);
  const fs = Math.max(14, overlay.width / 45);
  ctx.lineWidth = lw;
  ctx.font = `600 ${fs}px system-ui, sans-serif`;
  ctx.textBaseline = 'top';
  for (const p of preds) {
    const [x, y, w, h] = p.bbox;
    ctx.strokeStyle = '#22c55e';
    ctx.strokeRect(x, y, w, h);
    const text = `${p.class} ${(p.score * 100).toFixed(0)}%`;
    const tw = ctx.measureText(text).width + 10;
    const th = fs + 6;
    const ly = y - th < 0 ? y : y - th;
    ctx.fillStyle = '#22c55e';
    ctx.fillRect(x, ly, tw, th);
    ctx.fillStyle = '#052e16';
    ctx.fillText(text, x + 5, ly + 3);
  }
}

// ---------- Detection list UI ----------
function renderDetections(preds) {
  if (!preds.length) {
    detectList.innerHTML = '<li class="empty">Nothing detected</li>';
    return;
  }
  // best score per label
  const best = new Map();
  for (const p of preds) {
    if (!best.has(p.class) || best.get(p.class) < p.score) best.set(p.class, p.score);
  }
  detectList.innerHTML = '';
  [...best.entries()]
    .sort((a, b) => b[1] - a[1])
    .forEach(([name, score]) => {
      const pct = Math.round(score * 100);
      const li = document.createElement('li');
      li.innerHTML = `<span class="name">${name}</span>
        <span style="display:flex;align-items:center">
          <span class="bar"><i style="width:${pct}%"></i></span>
          <span class="pct">${pct}%</span>
        </span>`;
      detectList.appendChild(li);
    });
}

function addLog(label) {
  const empty = logList.querySelector('.empty');
  if (empty) empty.remove();
  const li = document.createElement('li');
  const time = new Date().toLocaleTimeString();
  li.innerHTML = `<span style="text-transform:capitalize">${label} detected</span><span class="time">${time}</span>`;
  logList.prepend(li);
  while (logList.children.length > 50) logList.lastChild.remove();
}

// ---------- Voice announcement logic ----------
// Rules to avoid repeated announcements:
//  1. CONFIRMATION: a label must appear in CONFIRM_FRAMES consecutive detections.
//  2. ONCE PER APPEARANCE: after announcing, it is not announced again while it stays in view.
//  3. RESET: it can be announced again only after it has been gone for GONE_AFTER_MS...
//  4. COOLDOWN: ...and at least `cooldownMs` has passed since its last announcement.
//  5. GLOBAL GAP: never speak two things within MIN_GAP_MS; skip while speech is playing.
function updateAnnouncements(preds) {
  const now = performance.now();
  const present = new Set(preds.map((p) => p.class));

  // update streaks for labels currently visible
  for (const label of present) {
    const t = tracker.get(label) || { streak: 0, lastSeen: 0, announced: false, lastAnnounced: -Infinity };
    t.streak++;
    t.lastSeen = now;
    tracker.set(label, t);
  }
  // reset streaks / announced flag for labels not visible
  for (const [label, t] of tracker) {
    if (present.has(label)) continue;
    t.streak = 0;
    if (t.announced && now - t.lastSeen > GONE_AFTER_MS) t.announced = false;
    if (!t.announced && now - t.lastSeen > 60000) tracker.delete(label);
  }

  // announce confirmed, not-yet-announced labels (one per cycle)
  for (const label of present) {
    const t = tracker.get(label);
    if (t.streak < CONFIRM_FRAMES || t.announced) continue;
    if (now - t.lastAnnounced < cooldownMs) continue;
    if (now - lastSpeechTime < MIN_GAP_MS) break;
    if (speak(`${label} detected`)) {
      t.announced = true;
      t.lastAnnounced = now;
      lastSpeechTime = now;
      addLog(label);
    }
    break;
  }
}

function speak(text) {
  if (!voiceToggle.checked) return true; // still log/mark, just no sound
  if (!('speechSynthesis' in window)) return true;
  if (speechSynthesis.speaking) return false; // try again next cycle
  const u = new SpeechSynthesisUtterance(text);
  u.lang = 'en-US';
  u.rate = 1;
  speechSynthesis.speak(u);
  return true;
}

// ---------- Wire up UI ----------
startBtn.addEventListener('click', startCamera);
stopBtn.addEventListener('click', stopCamera);
deviceSelect.addEventListener('change', () => { if (running) startCamera(); });
confRange.addEventListener('input', () => {
  minConfidence = confRange.value / 100;
  $('confVal').textContent = confRange.value;
});
coolRange.addEventListener('input', () => {
  cooldownMs = coolRange.value * 1000;
  $('coolVal').textContent = coolRange.value;
});
$('testVoiceBtn').addEventListener('click', () => {
  speechSynthesis.cancel();
  speechSynthesis.speak(new SpeechSynthesisUtterance('Voice test. Apple detected.'));
});
navigator.mediaDevices?.addEventListener?.('devicechange', listCameras);

setBadge(camBadge, 'Camera: disconnected', 'off');
loadModel();
