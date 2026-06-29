/**
 * CricketEye — Ball Tracker (Starter / Educational Version)
 *
 * Pipeline:
 *   Camera → Frame → HSV Conversion → Color Mask → Blob Centroid → Trail → Render
 *
 * Key concepts:
 *   - RGB→HSV: isolates color from brightness so lighting doesn't break detection
 *   - Color thresholding: each pixel tested against a ball-specific HSV range
 *   - Blob detection: grid scan finds the densest region of matching pixels
 *   - Ring buffer: stores last N positions for the fading trail
 */

"use strict";

// ─────────────────────────────────────────────────────────
//  CONSTANTS — HSV Detection Ranges per Ball Color
//  HSV: Hue [0-360], Saturation [0-100], Value [0-100]
// ─────────────────────────────────────────────────────────
const BALL_PROFILES = {
  red: {
    // Red wraps around 0°/360° in HSV, so we need two ranges
    ranges: [
      { hMin: 0, hMax: 18, sMin: 45, sMax: 100, vMin: 30, vMax: 100 },
      { hMin: 340, hMax: 360, sMin: 45, sMax: 100, vMin: 30, vMax: 100 },
    ],
    color: '#ff3b3b',
    glowColor: 'rgba(255,59,59,0.6)',
    trailStart: 'rgba(255,100,100,0.9)',
    trailEnd: 'rgba(255,59,59,0)',
  },
  white: {
    ranges: [
      { hMin: 0, hMax: 360, sMin: 0, sMax: 25, vMin: 75, vMax: 100 },
    ],
    color: '#e8e8e8',
    glowColor: 'rgba(220,220,255,0.5)',
    trailStart: 'rgba(230,230,255,0.9)',
    trailEnd: 'rgba(200,200,255,0)',
  },
  pink: {
    ranges: [
      { hMin: 300, hMax: 340, sMin: 40, sMax: 100, vMin: 40, vMax: 100 },
    ],
    color: '#ff69b4',
    glowColor: 'rgba(255,105,180,0.5)',
    trailStart: 'rgba(255,130,190,0.9)',
    trailEnd: 'rgba(255,69,150,0)',
  },
  green: {
    ranges: [
      { hMin: 90, hMax: 150, sMin: 40, sMax: 100, vMin: 25, vMax: 100 },
    ],
    color: '#39d353',
    glowColor: 'rgba(57,211,83,0.6)',
    trailStart: 'rgba(80,220,100,0.9)',
    trailEnd: 'rgba(57,211,83,0)',
  },
};

// ─────────────────────────────────────────────────────────
//  GLOBAL STATE
// ─────────────────────────────────────────────────────────
const state = {
  isTracking: true,
  sessionMaxY: -Infinity,
  ballColor: 'red',
  sensitivity: 40,         // tolerance around HSV ranges (0-80)
  maxTrail: 30,         // how many positions to keep in trail
  trail: [],         // ring buffer of {x, y, t} positions
  lastPos: null,       // {x, y, t} — previous frame's ball position
  frameCount: 0,
  lastFpsTime: performance.now(),
  fps: 0,
  detected: false,
  line: {
    status: 'none', // 'none', 'drawing', 'tracking', 'lost'
    p1: null,
    p2: null,
  }
};

// OpenCV variables
let oldGray = null;
let p0 = null;
let maxLevel = 2;
let winSize = null;
let criteria = null;

// ─────────────────────────────────────────────────────────
//  DOM REFS
// ─────────────────────────────────────────────────────────
const video = document.getElementById('video');
const canvas = document.getElementById('mainCanvas');
const ctx = canvas.getContext('2d', { willReadFrequently: true });
const fpsLabel = document.getElementById('fpsLabel');
const ballLabel = document.getElementById('ballLabel');
const sensitivitySlider = document.getElementById('sensitivity');
const sensitivityVal = document.getElementById('sensitivityVal');
const trailSlider = document.getElementById('trailLength');
const trailVal = document.getElementById('trailVal');
const resetBtn = document.getElementById('resetBtn');
const statSpeed = document.getElementById('statSpeed');
const statX = document.getElementById('statX');
const statY = document.getElementById('statY');
const statRadius = document.getElementById('statRadius');
const sideLabel = document.getElementById('sideLabel');
const lineInstruction = document.getElementById('lineInstruction');
const autoDetectBtn = document.getElementById('autoDetectBtn');
const toggleTrackingBtn = document.getElementById('toggleTrackingBtn');
const screenshotModal = document.getElementById('screenshotModal');
const screenshotImg = document.getElementById('screenshotImg');
const closeModalBtn = document.getElementById('closeModalBtn');
const settingsToggle = document.getElementById('settingsToggle');
const settingsPanel = document.getElementById('settingsPanel');
const settingsClose = document.getElementById('settingsClose');
const settingsBackdrop = document.getElementById('settingsBackdrop');

const bestFrameCanvas = document.createElement('canvas');
const bestFrameCtx = bestFrameCanvas.getContext('2d', { willReadFrequently: true });

// ─────────────────────────────────────────────────────────
//  LINE DRAWING EVENTS
// ─────────────────────────────────────────────────────────

function getEventPos(e) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  let clientX = e.clientX;
  let clientY = e.clientY;
  if (e.touches && e.touches.length > 0) {
    clientX = e.touches[0].clientX;
    clientY = e.touches[0].clientY;
  }
  return {
    x: (clientX - rect.left) * scaleX,
    y: (clientY - rect.top) * scaleY
  };
}

function handleStart(e) {
  e.preventDefault();
  const pos = getEventPos(e);
  state.line.status = 'drawing';
  state.line.p1 = pos;
  state.line.p2 = { ...pos };
}

function handleMove(e) {
  if (state.line.status !== 'drawing') return;
  e.preventDefault();
  const pos = getEventPos(e);
  state.line.p2 = pos;
}

function handleEnd(e) {
  if (state.line.status === 'drawing') {
    const dx = state.line.p2.x - state.line.p1.x;
    const dy = state.line.p2.y - state.line.p1.y;
    if (dx * dx + dy * dy > 100) {
      state.line.status = 'tracking';
      if (p0) p0.delete();
      if (typeof cv !== 'undefined' && cv.Mat) {
        p0 = cv.matFromArray(2, 1, cv.CV_32FC2, [state.line.p1.x, state.line.p1.y, state.line.p2.x, state.line.p2.y]);
      }
    } else {
      state.line.status = 'none';
      state.line.p1 = null;
      state.line.p2 = null;
    }
  }
}

canvas.addEventListener('mousedown', handleStart);
canvas.addEventListener('mousemove', handleMove);
canvas.addEventListener('mouseup', handleEnd);
canvas.addEventListener('touchstart', handleStart, { passive: false });
canvas.addEventListener('touchmove', handleMove, { passive: false });
canvas.addEventListener('touchend', handleEnd);

// ─────────────────────────────────────────────────────────
//  LANDSCAPE ORIENTATION LOCK
// ─────────────────────────────────────────────────────────
function lockLandscape() {
  // Try to lock orientation to landscape using the Screen Orientation API
  if (screen.orientation && screen.orientation.lock) {
    screen.orientation.lock('landscape').then(() => {
      console.log('Orientation locked to landscape');
    }).catch(err => {
      console.warn('Could not lock orientation:', err.message);
    });
  }
}

// ─────────────────────────────────────────────────────────
//  SETTINGS PANEL TOGGLE
// ─────────────────────────────────────────────────────────
function openSettings() {
  settingsPanel.classList.add('open');
  settingsBackdrop.classList.add('open');
  settingsToggle.classList.add('open');
}

function closeSettings() {
  settingsPanel.classList.remove('open');
  settingsBackdrop.classList.remove('open');
  settingsToggle.classList.remove('open');
}

settingsToggle.addEventListener('click', () => {
  if (settingsPanel.classList.contains('open')) {
    closeSettings();
  } else {
    openSettings();
  }
});

settingsClose.addEventListener('click', closeSettings);
settingsBackdrop.addEventListener('click', closeSettings);

// ─────────────────────────────────────────────────────────
//  STAGE 1 — CAMERA SETUP (Mobile-First, Rear Camera)
// ─────────────────────────────────────────────────────────

/**
 * Resize the canvas to match the actual video dimensions.
 * This is called on initial load AND on orientation change
 * to prevent any stretching.
 */
function resizeCanvas() {
  if (!video.videoWidth || !video.videoHeight) return;

  const vw = video.videoWidth;
  const vh = video.videoHeight;

  canvas.width = vw;
  canvas.height = vh;
  bestFrameCanvas.width = vw;
  bestFrameCanvas.height = vh;

  // Reset OpenCV oldGray so it re-initializes at new size
  if (oldGray) {
    oldGray.delete();
    oldGray = null;
  }

  console.log(`Canvas resized: ${vw}×${vh} (aspect ${(vw/vh).toFixed(2)})`);
}

async function startCamera() {
  try {
    // Force rear camera on mobile (facingMode: 'environment' exact)
    // Falls back to ideal if exact fails (e.g. on desktop/webcam)
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { exact: 'environment' },
          width: { ideal: 1920 },
          height: { ideal: 1080 },
        },
        audio: false,
      });
    } catch (exactErr) {
      console.warn('Exact rear camera failed, falling back to ideal:', exactErr.message);
      stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: 'environment' },
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });
    }

    video.srcObject = stream;

    // Lock landscape after camera is granted
    lockLandscape();

    // Once video metadata loads, size the canvas to match
    video.addEventListener('loadedmetadata', () => {
      resizeCanvas();
      console.log(`Camera ready: ${canvas.width}×${canvas.height}`);
      requestAnimationFrame(tick); // ← start the main loop
    });

    // Handle orientation changes — camera dimensions may swap
    if (screen.orientation) {
      screen.orientation.addEventListener('change', () => {
        setTimeout(resizeCanvas, 300);
      });
    }
    window.addEventListener('resize', () => {
      clearTimeout(window._resizeTimer);
      window._resizeTimer = setTimeout(resizeCanvas, 200);
    });

  } catch (err) {
    console.error('Camera error:', err);
    alert('Camera access denied or unavailable. Please allow camera and refresh.');
  }
}

// ─────────────────────────────────────────────────────────
//  STAGE 2 — RGB → HSV CONVERSION
//
//  Why HSV?
//  - Hue separates "color" from "brightness"
//  - A red ball looks red whether lit or in shadow
//  - RGB mixes all three, making thresholding unreliable under changing light
// ─────────────────────────────────────────────────────────

/**
 * Convert a single RGB pixel to HSV.
 * @param {number} r 0-255
 * @param {number} g 0-255
 * @param {number} b 0-255
 * @returns {{ h: number, s: number, v: number }} h=[0,360], s=[0,100], v=[0,100]
 */
function rgbToHsv(r, g, b) {
  r /= 255; g /= 255; b /= 255;

  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const delta = max - min;

  // Value = brightness
  const v = max * 100;

  // Saturation = how "colorful" vs. grey
  const s = max === 0 ? 0 : (delta / max) * 100;

  // Hue = which color (0–360°)
  let h = 0;
  if (delta !== 0) {
    if (max === r) h = 60 * (((g - b) / delta) % 6);
    else if (max === g) h = 60 * (((b - r) / delta) + 2);
    else h = 60 * (((r - g) / delta) + 4);
  }
  if (h < 0) h += 360;

  return { h, s, v };
}

// ─────────────────────────────────────────────────────────
//  STAGE 3 — COLOR THRESHOLDING
//
//  For each pixel, check if its HSV falls within any of
//  the ball color's defined ranges (with added sensitivity).
// ─────────────────────────────────────────────────────────

/**
 * Returns true if the given HSV matches the current ball color profile.
 */
function isMatchingPixel(h, s, v, profile, sensitivity) {
  const tol = sensitivity / 2; // expand ranges by this much

  return profile.ranges.some(range => {
    const hMatch = (h >= range.hMin - tol) && (h <= range.hMax + tol);
    const sMatch = s >= Math.max(0, range.sMin - tol)
      && s <= Math.min(100, range.sMax + tol);
    const vMatch = v >= Math.max(0, range.vMin - tol)
      && v <= Math.min(100, range.vMax + tol);
    return hMatch && sMatch && vMatch;
  });
}

// ─────────────────────────────────────────────────────────
//  STAGE 4 — BLOB DETECTION (Find Ball Center)
//
//  Strategy: divide canvas into a grid of cells.
//  For each cell, count matching pixels.
//  The cell with the most matches = likely ball location.
//  Then refine: compute the weighted centroid of matching pixels in that cell.
// ─────────────────────────────────────────────────────────

const GRID_COLS = 40; // how many horizontal divisions
const GRID_ROWS = 30; // how many vertical divisions

/**
 * Scan ImageData and return the ball's detected {x, y, radius, confidence}.
 * Returns null if no ball is found.
 */
function detectBall(imageData, profile) {
  const { data, width, height } = imageData;
  const cellW = width / GRID_COLS;
  const cellH = height / GRID_ROWS;

  // Count matching pixels per grid cell
  const grid = new Float32Array(GRID_COLS * GRID_ROWS);

  for (let y = 0; y < height; y += 2) {  // step 2 = sample every other row (speed)
    for (let x = 0; x < width; x += 2) { // step 2 = sample every other col
      const i = (y * width + x) * 4;
      const r = data[i], g = data[i + 1], b = data[i + 2];

      const hsv = rgbToHsv(r, g, b);

      if (isMatchingPixel(hsv.h, hsv.s, hsv.v, profile, state.sensitivity)) {
        const col = Math.floor(x / cellW);
        const row = Math.floor(y / cellH);
        grid[row * GRID_COLS + col]++;
      }
    }
  }

  // Find the cell with the most matching pixels
  let bestIdx = 0;
  for (let i = 1; i < grid.length; i++) {
    if (grid[i] > grid[bestIdx]) bestIdx = i;
  }

  const bestCount = grid[bestIdx];

  // Minimum threshold: need at least this many matching pixels
  const minThreshold = 4;
  if (bestCount < minThreshold) return null;

  // Get the bounding box of the best cell
  const bestCol = bestIdx % GRID_COLS;
  const bestRow = Math.floor(bestIdx / GRID_COLS);
  const cellX0 = Math.floor(bestCol * cellW);
  const cellY0 = Math.floor(bestRow * cellH);
  const cellX1 = Math.min(width, cellX0 + Math.ceil(cellW));
  const cellY1 = Math.min(height, cellY0 + Math.ceil(cellH));

  // Refine: compute centroid of matching pixels in that cell + neighbors
  let sumX = 0, sumY = 0, count = 0;
  const searchX0 = Math.max(0, cellX0 - Math.ceil(cellW));
  const searchY0 = Math.max(0, cellY0 - Math.ceil(cellH));
  const searchX1 = Math.min(width, cellX1 + Math.ceil(cellW));
  const searchY1 = Math.min(height, cellY1 + Math.ceil(cellH));

  for (let y = searchY0; y < searchY1; y++) {
    for (let x = searchX0; x < searchX1; x++) {
      const i = (y * width + x) * 4;
      const r = data[i], g = data[i + 1], b = data[i + 2];
      const hsv = rgbToHsv(r, g, b);
      if (isMatchingPixel(hsv.h, hsv.s, hsv.v, profile, state.sensitivity)) {
        sumX += x;
        sumY += y;
        count++;
      }
    }
  }

  if (count === 0) return null;

  const cx = sumX / count;
  const cy = sumY / count;
  // Rough radius: sqrt of pixel count (circle area = π·r²)
  const radius = Math.max(8, Math.sqrt(count / Math.PI));
  const confidence = Math.min(100, Math.round(bestCount * 5));

  return { x: cx, y: cy, radius, confidence };
}

// ─────────────────────────────────────────────────────────
//  STAGE 5 — TRAIL RENDERING
// ─────────────────────────────────────────────────────────

function drawTrail(profile) {
  const n = state.trail.length;
  if (n < 2) return;

  for (let i = 1; i < n; i++) {
    const alpha = i / n; // fade from 0 (oldest) to 1 (newest)
    const prev = state.trail[i - 1];
    const curr = state.trail[i];
    const lineWidth = 2 + alpha * 6; // thin→thick

    ctx.beginPath();
    ctx.moveTo(prev.x, prev.y);
    ctx.lineTo(curr.x, curr.y);
    ctx.strokeStyle = `rgba(${hexToRgbStr(profile.color)}, ${alpha * 0.9})`;
    ctx.lineWidth = lineWidth;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.stroke();
  }
}

function drawBallCircle(ball, profile) {
  const { x, y, radius } = ball;

  // Outer glow ring
  ctx.beginPath();
  ctx.arc(x, y, radius + 10, 0, Math.PI * 2);
  ctx.strokeStyle = profile.glowColor;
  ctx.lineWidth = 4;
  ctx.shadowColor = profile.color;
  ctx.shadowBlur = 20;
  ctx.stroke();
  ctx.shadowBlur = 0;

  // Main detection circle
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.strokeStyle = profile.color;
  ctx.lineWidth = 2.5;
  ctx.stroke();

  // Center dot
  ctx.beginPath();
  ctx.arc(x, y, 3, 0, Math.PI * 2);
  ctx.fillStyle = profile.color;
  ctx.fill();

  // Crosshair lines
  ctx.strokeStyle = `rgba(${hexToRgbStr(profile.color)}, 0.5)`;
  ctx.lineWidth = 1;
  const len = radius + 16;
  ctx.beginPath(); ctx.moveTo(x - len, y); ctx.lineTo(x + len, y); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(x, y - len); ctx.lineTo(x, y + len); ctx.stroke();
}

// ─────────────────────────────────────────────────────────
//  MAIN LOOP
// ─────────────────────────────────────────────────────────

function tick(timestamp) {
  requestAnimationFrame(tick);

  if (video.readyState < 2) return; // video not ready

  // ── Draw current video frame ──────────────────────────
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  // ── FPS counter ───────────────────────────────────────
  state.frameCount++;
  if (timestamp - state.lastFpsTime > 500) {
    state.fps = Math.round(state.frameCount / ((timestamp - state.lastFpsTime) / 1000));
    state.frameCount = 0;
    state.lastFpsTime = timestamp;
    fpsLabel.textContent = `FPS: ${state.fps}`;
  }

  // ── Get pixel data ────────────────────────────────────
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

  // ── OpenCV Optical Flow ───────────────────────────────
  if (typeof cv !== 'undefined' && cv.Mat) {
    if (!winSize) {
      winSize = new cv.Size(31, 31);
      criteria = new cv.TermCriteria(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01);
    }

    if (!oldGray) {
      oldGray = new cv.Mat(canvas.height, canvas.width, cv.CV_8UC1);
      // Let's populate oldGray right away so first frame has it
      let initialSrc = cv.matFromImageData(imageData);
      cv.cvtColor(initialSrc, oldGray, cv.COLOR_RGBA2GRAY);
      initialSrc.delete();
    }

    let newGray = new cv.Mat(canvas.height, canvas.width, cv.CV_8UC1);
    let src = cv.matFromImageData(imageData);
    cv.cvtColor(src, newGray, cv.COLOR_RGBA2GRAY);

    if (state.line.status === 'tracking' && p0) {
      let p1_cv = new cv.Mat();
      let st = new cv.Mat();
      let err = new cv.Mat();

      try {
        cv.calcOpticalFlowPyrLK(oldGray, newGray, p0, p1_cv, st, err, winSize, 3, criteria);

        let pt0_status = st.data[0];
        let pt1_status = st.data[1];

        if (pt0_status === 1 && pt1_status === 1) {
          state.line.p1.x = p1_cv.data32F[0];
          state.line.p1.y = p1_cv.data32F[1];
          state.line.p2.x = p1_cv.data32F[2];
          state.line.p2.y = p1_cv.data32F[3];

          const margin = 100; // Allow points to go slightly off-screen
          if (state.line.p1.x < -margin || state.line.p1.x > canvas.width + margin ||
            state.line.p1.y < -margin || state.line.p1.y > canvas.height + margin ||
            state.line.p2.x < -margin || state.line.p2.x > canvas.width + margin ||
            state.line.p2.y < -margin || state.line.p2.y > canvas.height + margin) {
            state.line.status = 'lost';
          } else {
            p0.delete();
            p0 = p1_cv.clone();
          }
        } else {
          state.line.status = 'lost';
        }
      } catch (e) {
        console.error("Optical flow error", e);
        state.line.status = 'lost';
      }

      p1_cv.delete();
      st.delete();
      err.delete();
    }

    newGray.copyTo(oldGray);
    newGray.delete();
    src.delete();
  }

  // ── Detect ball ───────────────────────────────────────
  const profile = BALL_PROFILES[state.ballColor];
  let ball = null;
  let isNewHighest = false;

  if (state.isTracking) {
    ball = detectBall(imageData, profile);
  }

  if (ball) {
    // Push to trail (ring buffer)
    state.trail.push({ x: ball.x, y: ball.y, t: timestamp });
    if (state.trail.length > state.maxTrail) state.trail.shift();

    if (ball.y > state.sessionMaxY) {
      state.sessionMaxY = ball.y;
      isNewHighest = true; // reusing variable name, actually means isNewLowest
    }

    // Speed estimate (pixels/sec between last two positions)
    if (state.trail.length >= 2) {
      const a = state.trail[state.trail.length - 2];
      const b = state.trail[state.trail.length - 1];
      const dt = (b.t - a.t) / 1000; // seconds
      if (dt > 0) {
        const dx = b.x - a.x, dy = b.y - a.y;
        const speed = Math.round(Math.sqrt(dx * dx + dy * dy) / dt);
        statSpeed.textContent = speed;
      }
    }

    // Update stats
    statX.textContent = Math.round(ball.x);
    statY.textContent = Math.round(ball.y);
    statRadius.textContent = Math.round(ball.radius);

    // Update HUD
    ballLabel.textContent = `🔴 Tracking`;
    ballLabel.classList.add('tracking');

    state.detected = true;
    state.lastPos = { x: ball.x, y: ball.y, t: timestamp };

    // Draw trail and circle
    drawTrail(profile);
    drawBallCircle(ball, profile);

  } else {
    // Ball not found — still draw existing trail (it'll auto-clear on reset)
    drawTrail(profile);

    ballLabel.textContent = 'No Ball';
    ballLabel.classList.remove('tracking');
    state.detected = false;

    // Slowly expire old trail points (1 second timeout)
    const now = timestamp;
    state.trail = state.trail.filter(p => now - p.t < 1000);
  }

  // ── Draw Line ─────────────────────────────────────────
  if (state.line.status !== 'none') {
    ctx.beginPath();
    ctx.moveTo(state.line.p1.x, state.line.p1.y);
    ctx.lineTo(state.line.p2.x, state.line.p2.y);
    ctx.strokeStyle = state.line.status === 'lost' ? 'rgba(255, 0, 0, 0.5)' : 'rgba(255, 204, 0, 0.8)';
    ctx.lineWidth = 4;
    if (state.line.status === 'drawing') {
      ctx.setLineDash([10, 10]);
    } else {
      ctx.setLineDash([]);
    }
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = 'yellow';
    ctx.beginPath(); ctx.arc(state.line.p1.x, state.line.p1.y, 5, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.arc(state.line.p2.x, state.line.p2.y, 5, 0, Math.PI * 2); ctx.fill();
  }

  // ── Side Detection ────────────────────────────────────
  if (ball && state.line.status === 'tracking') {
    const A = state.line.p1;
    const B = state.line.p2;
    const C = ball;

    const crossProduct = (B.x - A.x) * (C.y - A.y) - (B.y - A.y) * (C.x - A.x);
    const side = crossProduct > 0 ? 'Left Side' : 'Right Side';
    sideLabel.textContent = `Side: ${side}`;
    sideLabel.classList.add('active');
  } else {
    sideLabel.textContent = `Side: --`;
    sideLabel.classList.remove('active');
  }

  // ── UI Prompts ────────────────────────────────────────
  if (state.line.status === 'none' || state.line.status === 'lost') {
    lineInstruction.textContent = state.line.status === 'lost' ? 'Line lost! Draw again or Auto Detect' : 'Draw Line or click Auto Detect';
    lineInstruction.classList.add('active');
  } else {
    lineInstruction.classList.remove('active');
  }

  // ── Save Best Frame ───────────────────────────────────
  if (isNewHighest) {
    bestFrameCtx.drawImage(canvas, 0, 0);
  }
}

// ─────────────────────────────────────────────────────────
//  UTILITY
// ─────────────────────────────────────────────────────────

/** Convert hex color like '#ff3b3b' → '255,59,59' for rgba() */
function hexToRgbStr(hex) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `${r},${g},${b}`;
}

/** 
 * Automatically detect a white line nearest to the center of the frame.
 */
function autoDetectLine() {
  if (typeof cv === 'undefined' || !cv.Mat || video.readyState < 2) return;

  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  let src = cv.matFromImageData(imageData);
  
  let hsv = new cv.Mat();
  cv.cvtColor(src, hsv, cv.COLOR_RGBA2RGB);
  cv.cvtColor(hsv, hsv, cv.COLOR_RGB2HSV);

  // Helper function to detect a vertical line using specific HSV color bounds
  function detectWithBounds(lowArr, highArr) {
    let mask = new cv.Mat();
    let low = cv.matFromArray(1, 1, cv.CV_8UC3, lowArr);
    let high = cv.matFromArray(1, 1, cv.CV_8UC3, highArr);
    
    cv.inRange(hsv, low, high, mask);
    
    let lines = new cv.Mat();
    // More relaxed Hough parameters: lower threshold and minLineLength to 20, allow gap up to 15
    cv.HoughLinesP(mask, lines, 1, Math.PI / 180, 20, 20, 15);
    
    let bestLine = null;
    console.log(`[Line Detection] Found ${lines.rows} raw lines under current color filter.`);
    if (lines.rows > 0) {
      let minDistanceToCenter = Infinity;
      const cx = canvas.width / 2;
      const cy = canvas.height / 2;

      for (let i = 0; i < lines.rows; ++i) {
        let x1 = lines.data32S[i * 4];
        let y1 = lines.data32S[i * 4 + 1];
        let x2 = lines.data32S[i * 4 + 2];
        let y2 = lines.data32S[i * 4 + 3];
        
        // Filter out horizontal lines
        let dx = Math.abs(x2 - x1);
        let dy = Math.abs(y2 - y1);
        if (dy < dx) {
          console.log(`[Line Detection] Skipped horizontal-ish line: (${x1}, ${y1}) -> (${x2}, ${y2})`);
          continue;
        }
        
        // Calculate distance from center to this line's midpoint
        let mx = (x1 + x2) / 2;
        let my = (y1 + y2) / 2;
        let dist = Math.sqrt(Math.pow(mx - cx, 2) + Math.pow(my - cy, 2));
        
        if (dist < minDistanceToCenter) {
          minDistanceToCenter = dist;
          bestLine = { p1: { x: x1, y: y1 }, p2: { x: x2, y: y2 } };
        }
      }
    }
    
    // Cleanup local mats
    mask.delete();
    low.delete();
    high.delete();
    lines.delete();
    
    return bestLine;
  }

  // 1st Preference: White line on Green pitch (H: 0-180, S: 0-90, V: 130-255)
  // Grass is generally highly saturated (S > 100). White lines are desaturated (S < 90).
  // Lowered V threshold to 130 to catch lines in shadows.
  console.log("[Line Detection] Checking 1st preference: White line on Green pitch...");
  let bestLine = detectWithBounds([0, 0, 130], [180, 90, 255]);
  
  // 2nd Preference: Blue line on White pitch (H: 90-135, S: 50-255, V: 50-255)
  // Lowered S/V limits to 50 to ensure we catch blue lines in shadows or under exposure.
  if (!bestLine) {
    console.log("[Line Detection] White line detection failed, checking 2nd preference: Blue line on White pitch...");
    bestLine = detectWithBounds([90, 50, 50], [135, 255, 255]);
  }
  
  if (bestLine) {
    state.line.p1 = bestLine.p1;
    state.line.p2 = bestLine.p2;
    state.line.status = 'tracking';
    
    if (p0) p0.delete();
    p0 = cv.matFromArray(2, 1, cv.CV_32FC2, [state.line.p1.x, state.line.p1.y, state.line.p2.x, state.line.p2.y]);
    console.log('Auto-detected line:', state.line);
  } else {
    alert("Could not automatically detect a vertical line (tried White/Green and Blue/White). Please draw it manually on the canvas.");
  }

  // Cleanup top-level mats
  src.delete();
  hsv.delete();
}

// ─────────────────────────────────────────────────────────
//  UI EVENT LISTENERS
// ─────────────────────────────────────────────────────────

// Ball color picker
document.querySelectorAll('.color-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.color-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    state.ballColor = btn.dataset.color;
    state.trail = []; // clear trail on color switch
    console.log('Ball color set to:', state.ballColor);
  });
});

// Sensitivity slider
sensitivitySlider.addEventListener('input', () => {
  state.sensitivity = Number(sensitivitySlider.value);
  sensitivityVal.textContent = state.sensitivity;
});

// Trail length slider
trailSlider.addEventListener('input', () => {
  state.maxTrail = Number(trailSlider.value);
  trailVal.textContent = state.maxTrail;
});

// Toggle Tracking
if (toggleTrackingBtn) {
  toggleTrackingBtn.addEventListener('click', () => {
    state.isTracking = !state.isTracking;
    if (state.isTracking) {
      toggleTrackingBtn.querySelector('.action-icon').textContent = '⏹';
      toggleTrackingBtn.classList.add('tracking-active');
      toggleTrackingBtn.title = 'Stop Tracking';
      state.sessionMaxY = -Infinity; // reset best point
    } else {
      toggleTrackingBtn.querySelector('.action-icon').textContent = '▶';
      toggleTrackingBtn.classList.remove('tracking-active');
      toggleTrackingBtn.title = 'Start Tracking';
      
      if (state.sessionMaxY !== -Infinity) {
        screenshotImg.src = bestFrameCanvas.toDataURL('image/jpeg', 0.8);
        screenshotModal.classList.remove('hidden');
      }

      // Clear trail when tracking stops
      state.trail = [];
      statSpeed.textContent = '--';
      statX.textContent = '--';
      statY.textContent = '--';
      statRadius.textContent = '--';
      ballLabel.textContent = 'No Ball';
      ballLabel.classList.remove('tracking');
      state.detected = false;
    }
  });
}

// Reset
resetBtn.addEventListener('click', () => {
  state.trail = [];
  statSpeed.textContent = '--';
  statX.textContent = '--';
  statY.textContent = '--';
  statRadius.textContent = '--';
});

// Auto Detect Line
if (autoDetectBtn) {
  autoDetectBtn.addEventListener('click', () => {
    autoDetectLine();
  });
}

// Modal Close
if (closeModalBtn) {
  closeModalBtn.addEventListener('click', () => {
    screenshotModal.classList.add('hidden');
  });
}

// ─────────────────────────────────────────────────────────
//  BOOT
// ─────────────────────────────────────────────────────────
function waitForOpenCV() {
  return new Promise(resolve => {
    if (typeof cv !== 'undefined' && cv.Mat) {
      resolve();
    } else {
      const check = setInterval(() => {
        if (typeof cv !== 'undefined' && cv.Mat) {
          clearInterval(check);
          resolve();
        }
      }, 100);
    }
  });
}

waitForOpenCV().then(() => {
  console.log("OpenCV is ready.");
  startCamera();
});
