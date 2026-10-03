// Procedural low-poly models for every device type in catalog.js.
//
// Public API (CONTRACT.md, "Scene"):
//   createDeviceModel(device) -> { group, port, update(device, dtSec, tSec), setStatus(status), dispose() }
//
// scene.js places `group` at device.pos and turns it by device.rotY, so every
// model is built around its local origin, facing +z, at real-world size in
// metres: floor and table items stand on y = 0, wall items have their mount
// point at the origin with the wall behind them (-z), and ceiling items hang
// down from it (negative y). `port` is a group-local point near the top or
// the antenna where scene.js attaches the network link.
//
// Keeping update() cheap:
//  - geometries and plain materials are cached at module level, shared by all
//    instances and kept across a sim reset; only materials whose colour or
//    glow changes per device, canvas screens and status effects are per instance;
//  - update() never allocates: state lives in closures and scratch objects,
//    canvas screens are redrawn only when what they show changes (or at ~4 Hz
//    for animated ones), and the GPU upload skips mipmaps;
//  - status effects (alert halo, blocked cage, offline greying) are built once,
//    lazily, by one shared helper sized from each model's bounding box.
//
// Device animations follow simulation time (dtSec/tSec, frozen while paused);
// fades, LED decay and the status effects follow real time.
import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { HOUSE, RESEARCH } from './catalog.js';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const TAU = Math.PI * 2;
const COLOR = {
  saffron: 0xf2b134,
  allowed: 0x5ad1e6,
  alert: 0xffb347,
  blocked: 0xff4d5e,
  ledGreen: 0x3dff8a,
  ledRed: 0xff3b3b,
  ledAmber: 0xffa21f,
  ledBlue: 0x3b8bff,
  ledWhite: 0xe8f1ff,
};
const IST_OFFSET_MS = 5.5 * 3600 * 1000;
const FALLBACK_DAY_SEC = 19 * 3600 + 42 * 60;   // 19:42 IST, the sim's start time
const SCREEN_MIN_MS = 250;                      // animated screens redraw at most ~4x per second
const FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif';
const MONO = 'ui-monospace, "DejaVu Sans Mono", Menlo, Consolas, monospace';

// Living-room floor the robot vacuum may roam (world metres): the room minus
// the sofa, TV console, speaker side table and plant placed by scene.js.
const VACUUM_AREA = { x: [-5.75, -0.25], z: [-5.2, -1.5] };

// ---------------------------------------------------------------------------
// Small math helpers
// ---------------------------------------------------------------------------

const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
/** Frame-rate independent smoothing factor for x += (target - x) * damp(k, dt). */
const damp = (k, dt) => 1 - Math.exp(-k * dt);
/** Deterministic hash in [0, 1). */
function hash1(n) {
  const s = Math.sin(n * 127.1 + 311.7) * 43758.5453;
  return s - Math.floor(s);
}
function wrapAngle(a) {
  a = (a + Math.PI) % TAU;
  if (a < 0) a += TAU;
  return a - Math.PI;
}
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Seconds since midnight IST on the simulation clock (falls back to tSec). */
function simDaySeconds(tSec) {
  const time = globalThis.__shield?.state?.time;
  if (time && Number.isFinite(time.epochMs) && Number.isFinite(time.simMs)) {
    const s = ((time.epochMs + time.simMs + IST_OFFSET_MS) / 1000) % 86400;
    return s < 0 ? s + 86400 : s;
  }
  return (FALLBACK_DAY_SEC + tSec) % 86400;
}

function formatTime(daySec, twelve) {
  const total = Math.floor(daySec / 60);
  const h = Math.floor(total / 60) % 24;
  const m = total % 60;
  const mm = m < 10 ? `0${m}` : `${m}`;
  if (!twelve) return `${h < 10 ? '0' : ''}${h}:${mm}`;
  const h12 = h % 12 === 0 ? 12 : h % 12;
  return `${h12}:${mm} ${h < 12 ? 'AM' : 'PM'}`;
}

// ---------------------------------------------------------------------------
// Shared resource caches (module level, shared by every instance)
// ---------------------------------------------------------------------------

const geoCache = new Map();
const matCache = new Map();
const texCache = new Map();

function cached(map, key, make) {
  let value = map.get(key);
  if (value === undefined) {
    value = make();
    map.set(key, value);
  }
  return value;
}
const sharedGeo = (key, make) => cached(geoCache, key, make);
const sharedTex = (key, make) => cached(texCache, key, make);
function sharedMat(key, make) {
  return cached(matCache, key, () => {
    const m = make();
    m.userData.shared = true;
    return m;
  });
}

// Geometry shortcuts. Every call with the same arguments returns the same geometry.
const box = (w, h, d) => sharedGeo(`box|${w}|${h}|${d}`, () => new THREE.BoxGeometry(w, h, d));
const rbox = (w, h, d, r, seg = 2) =>
  sharedGeo(`rbox|${w}|${h}|${d}|${r}|${seg}`, () => new RoundedBoxGeometry(w, h, d, seg, r));
const cyl = (rt, rb, h, seg = 16, open = false, ts = 0, tl = TAU) =>
  sharedGeo(`cyl|${rt}|${rb}|${h}|${seg}|${open}|${ts}|${tl}`, () => new THREE.CylinderGeometry(rt, rb, h, seg, 1, open, ts, tl));
/** Cylinder whose axis points along +z (front-facing discs, lenses, knobs). */
const cylZ = (rt, rb, h, seg = 16) =>
  sharedGeo(`cylZ|${rt}|${rb}|${h}|${seg}`, () => new THREE.CylinderGeometry(rt, rb, h, seg).rotateX(Math.PI / 2));
const sphere = (r, ws = 16, hs = 12) => sharedGeo(`sph|${r}|${ws}|${hs}`, () => new THREE.SphereGeometry(r, ws, hs));
/** Torus in the xy plane (faces +z). */
const torus = (r, tube, rs = 8, ts = 32) => sharedGeo(`tor|${r}|${tube}|${rs}|${ts}`, () => new THREE.TorusGeometry(r, tube, rs, ts));
/** Torus lying flat in the xz plane. */
const flatTorus = (r, tube, rs = 8, ts = 32) =>
  sharedGeo(`ftor|${r}|${tube}|${rs}|${ts}`, () => new THREE.TorusGeometry(r, tube, rs, ts).rotateX(Math.PI / 2));
const circle = (r, seg = 32) => sharedGeo(`circ|${r}|${seg}`, () => new THREE.CircleGeometry(r, seg));      // faces +z
const plane = (w, h) => sharedGeo(`plane|${w}|${h}`, () => new THREE.PlaneGeometry(w, h));                  // faces +z
const ring = (ri, ro, seg = 32) => sharedGeo(`ring|${ri}|${ro}|${seg}`, () => new THREE.RingGeometry(ri, ro, seg)); // faces +z
/** Horizontal shapes facing +y; texture "up" points to -z so text reads from the front. */
const flatCircle = (r, seg = 32) => sharedGeo(`fcirc|${r}|${seg}`, () => new THREE.CircleGeometry(r, seg).rotateX(-Math.PI / 2));
const flatPlane = (w, d) => sharedGeo(`fplane|${w}|${d}`, () => new THREE.PlaneGeometry(w, d).rotateX(-Math.PI / 2));
const flatRing = (ri, ro, seg = 32, ts = 0, tl = TAU) =>
  sharedGeo(`fring|${ri}|${ro}|${seg}|${ts}|${tl}`, () => new THREE.RingGeometry(ri, ro, seg, 1, ts, tl).rotateX(-Math.PI / 2));

// Material shortcuts (shared).
const matte = (hex, roughness = 0.75) =>
  sharedMat(`matte|${hex}|${roughness}`, () => new THREE.MeshStandardMaterial({ color: hex, roughness, metalness: 0 }));
const metal = (hex, roughness = 0.35, metalness = 0.5) =>
  sharedMat(`metal|${hex}|${roughness}|${metalness}`, () => new THREE.MeshStandardMaterial({ color: hex, roughness, metalness }));
/** A static lit LED or emblem. */
const led = (hex, intensity = 1.6) =>
  sharedMat(`led|${hex}|${intensity}`, () => new THREE.MeshStandardMaterial({ color: 0x111214, emissive: hex, emissiveIntensity: intensity, roughness: 0.4 }));
/** Shared textured material; params() is called once. */
const texMat = (key, params) => sharedMat(`tex|${key}`, () => new THREE.MeshStandardMaterial(params()));

// Desaturated, darker twins of shared materials, used while a device is offline.
const GREY = new THREE.Color(0x8b909a);
const greyCache = new WeakMap();
const _hsl = { h: 0, s: 0, l: 0 };
function greyed(mat) {
  let g = greyCache.get(mat);
  if (!g) {
    g = mat.clone();
    if (g.color) {
      g.color.getHSL(_hsl);
      g.color.setHSL(_hsl.h, _hsl.s * 0.3, _hsl.l * 0.78);
    }
    if (g.emissive) g.emissiveIntensity = mat.emissiveIntensity * 0.2;
    greyCache.set(mat, g);
  }
  return g;
}

// ---------------------------------------------------------------------------
// Canvas helpers and shared textures
// ---------------------------------------------------------------------------

function makeCanvas(w, h) {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  return c;
}

function canvasTexture(canvas, repeat = false) {
  const t = new THREE.CanvasTexture(canvas);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 4;
  if (repeat) t.wrapS = t.wrapT = THREE.RepeatWrapping;
  return t;
}

/** Texture for a screen that is redrawn at runtime: no mipmaps, so uploads stay cheap. */
function screenTexture(canvas) {
  const t = canvasTexture(canvas);
  t.generateMipmaps = false;
  t.minFilter = THREE.LinearFilter;
  return t;
}

function roundRect(g, x, y, w, h, r) {
  g.beginPath();
  g.moveTo(x + r, y);
  g.arcTo(x + w, y, x + w, y + h, r);
  g.arcTo(x + w, y + h, x, y + h, r);
  g.arcTo(x, y + h, x, y, r);
  g.arcTo(x, y, x + w, y, r);
  g.closePath();
}

/** Heater-shield outline centred on (cx, cy), w wide and h tall (canvas y grows down). */
function shieldPath(g, cx, cy, w, h) {
  const sx = w, sy = h / 1.2;
  g.beginPath();
  g.moveTo(cx, cy - 0.6 * sy);
  g.quadraticCurveTo(cx + 0.25 * sx, cy - 0.47 * sy, cx + 0.5 * sx, cy - 0.5 * sy);
  g.lineTo(cx + 0.5 * sx, cy - 0.05 * sy);
  g.quadraticCurveTo(cx + 0.48 * sx, cy + 0.4 * sy, cx, cy + 0.6 * sy);
  g.quadraticCurveTo(cx - 0.48 * sx, cy + 0.4 * sy, cx - 0.5 * sx, cy - 0.05 * sy);
  g.lineTo(cx - 0.5 * sx, cy - 0.5 * sy);
  g.quadraticCurveTo(cx - 0.25 * sx, cy - 0.47 * sy, cx, cy - 0.6 * sy);
  g.closePath();
}

/** The same outline as a THREE.Shape (y up), centred on the origin. */
function shieldShape(w, h) {
  const sx = w, sy = h / 1.2;
  const s = new THREE.Shape();
  s.moveTo(0, 0.6 * sy);
  s.quadraticCurveTo(0.25 * sx, 0.47 * sy, 0.5 * sx, 0.5 * sy);
  s.lineTo(0.5 * sx, 0.05 * sy);
  s.quadraticCurveTo(0.48 * sx, -0.4 * sy, 0, -0.6 * sy);
  s.quadraticCurveTo(-0.48 * sx, -0.4 * sy, -0.5 * sx, 0.05 * sy);
  s.lineTo(-0.5 * sx, 0.5 * sy);
  s.quadraticCurveTo(-0.25 * sx, 0.47 * sy, 0, 0.6 * sy);
  return s;
}

/** Saffron SHIELD-IoT emblem: shield, dark field and a check mark. */
function shieldEmblem(g, cx, cy, w, h, field = '#1a1f2b') {
  shieldPath(g, cx, cy, w, h);
  g.fillStyle = '#f2b134';
  g.fill();
  shieldPath(g, cx, cy + h * 0.02, w * 0.72, h * 0.72);
  g.fillStyle = field;
  g.fill();
  g.strokeStyle = '#f2b134';
  g.lineWidth = Math.max(1.5, w * 0.11);
  g.lineCap = 'round';
  g.lineJoin = 'round';
  g.beginPath();
  g.moveTo(cx - w * 0.17, cy + h * 0.02);
  g.lineTo(cx - w * 0.03, cy + h * 0.15);
  g.lineTo(cx + w * 0.2, cy - h * 0.14);
  g.stroke();
}

function heartPath(g, x, y, s) {
  g.beginPath();
  g.moveTo(x, y + s * 0.62);
  g.bezierCurveTo(x - s * 1.25, y - s * 0.1, x - s * 0.6, y - s * 1.05, x, y - s * 0.38);
  g.bezierCurveTo(x + s * 0.6, y - s * 1.05, x + s * 1.25, y - s * 0.1, x, y + s * 0.62);
  g.closePath();
}

// Seven-segment digits: segment letters lit per character.
const SEGMENTS = { 0: 'abcdef', 1: 'bc', 2: 'abdeg', 3: 'abcdg', 4: 'bcfg', 5: 'acdfg', 6: 'acdefg', 7: 'abc', 8: 'abcdefg', 9: 'abcdfg', '-': 'g' };

function segPoly(g, x, y, horizontal, len, th) {
  const t = th / 2;
  g.beginPath();
  if (horizontal) {
    g.moveTo(x, y); g.lineTo(x + t, y - t); g.lineTo(x + len - t, y - t);
    g.lineTo(x + len, y); g.lineTo(x + len - t, y + t); g.lineTo(x + t, y + t);
  } else {
    g.moveTo(x, y); g.lineTo(x + t, y + t); g.lineTo(x + t, y + len - t);
    g.lineTo(x, y + len); g.lineTo(x - t, y + len - t); g.lineTo(x - t, y + t);
  }
  g.closePath();
  g.fill();
}

/** One seven-segment digit in a w x h cell at (x, y); unlit segments use `off`. */
function sevenSeg(g, ch, x, y, w, h, th, on, off) {
  const lit = SEGMENTS[ch] || '';
  const half = h / 2, gap = th * 0.55, vlen = half - gap * 2;
  const segs = [
    ['a', x + gap, y, true, w - gap * 2],
    ['b', x + w, y + gap, false, vlen],
    ['c', x + w, y + half + gap, false, vlen],
    ['d', x + gap, y + h, true, w - gap * 2],
    ['e', x, y + half + gap, false, vlen],
    ['f', x, y + gap, false, vlen],
    ['g', x + gap, y + half, true, w - gap * 2],
  ];
  for (const [k, sx, sy, hor, len] of segs) {
    g.fillStyle = lit.includes(k) ? on : off;
    segPoly(g, sx, sy, hor, len, th);
  }
}

/** Soft radial glow used by sprites (halo, bulbs, LEDs, steam, cloud). */
function glowTexture() {
  return sharedTex('glow', () => {
    const c = makeCanvas(64, 64), g = c.getContext('2d');
    const grd = g.createRadialGradient(32, 32, 0, 32, 32, 32);
    grd.addColorStop(0, 'rgba(255,255,255,1)');
    grd.addColorStop(0.22, 'rgba(255,255,255,0.6)');
    grd.addColorStop(0.55, 'rgba(255,255,255,0.14)');
    grd.addColorStop(1, 'rgba(255,255,255,0)');
    g.fillStyle = grd;
    g.fillRect(0, 0, 64, 64);
    return canvasTexture(c);
  });
}

function ventTexture() {
  return sharedTex('vents', () => {
    const c = makeCanvas(128, 64), g = c.getContext('2d');
    g.fillStyle = '#262a33';
    g.fillRect(0, 0, 128, 64);
    g.fillStyle = '#0c0d10';
    for (let y = 8; y < 60; y += 7) {
      roundRect(g, 10, y, 108, 3, 1.5);
      g.fill();
    }
    return canvasTexture(c);
  });
}

function fabricTexture() {
  return sharedTex('fabric', () => {
    const c = makeCanvas(64, 64), g = c.getContext('2d');
    g.fillStyle = '#4a5468';
    g.fillRect(0, 0, 64, 64);
    g.fillStyle = 'rgba(15,18,26,0.55)';
    for (let y = 2; y < 64; y += 4) for (let x = (y % 8 ? 2 : 0); x < 64; x += 4) g.fillRect(x, y, 2, 2);
    const t = canvasTexture(c, true);
    t.repeat.set(10, 3);
    return t;
  });
}

function dialTexture() {
  return sharedTex('dial', () => {
    const S = 512, c = makeCanvas(S, S), g = c.getContext('2d'), m = S / 2;
    const grd = g.createRadialGradient(m, m * 0.8, 20, m, m, m);
    grd.addColorStop(0, '#fbf8f1');
    grd.addColorStop(1, '#e9e3d6');
    g.fillStyle = grd;
    g.fillRect(0, 0, S, S);
    g.translate(m, m);
    for (let i = 0; i < 60; i++) {
      const five = i % 5 === 0;
      g.save();
      g.rotate((i / 60) * TAU);
      g.fillStyle = five ? '#1d2027' : '#6b6f78';
      g.fillRect(five ? -5 : -1.5, -m + 14, five ? 10 : 3, five ? 34 : 16);
      g.restore();
    }
    g.fillStyle = '#1d2027';
    g.font = `600 52px ${FONT}`;
    g.textAlign = 'center';
    g.textBaseline = 'middle';
    for (let n = 1; n <= 12; n++) {
      const a = (n / 12) * TAU;
      g.fillText(String(n), Math.sin(a) * (m - 92), -Math.cos(a) * (m - 92));
    }
    g.font = `600 22px ${FONT}`;
    g.fillStyle = '#b08433';
    g.fillText('SHIELD · NTP', 0, -m * 0.34);
    return canvasTexture(c);
  });
}

function shieldLabelTextures() {
  return sharedTex('shield-label', () => {
    const W = 256, H = 160;
    const paint = (glowOnly) => {
      const c = makeCanvas(W, H), g = c.getContext('2d');
      if (glowOnly) {
        g.fillStyle = '#000';
        g.fillRect(0, 0, W, H);
      } else {
        const grd = g.createLinearGradient(0, 0, W, H);
        grd.addColorStop(0, '#2b303b');
        grd.addColorStop(1, '#1d2129');
        g.fillStyle = grd;
        g.fillRect(0, 0, W, H);
        g.strokeStyle = 'rgba(255,255,255,0.06)';
        for (let x = -H; x < W; x += 6) { g.beginPath(); g.moveTo(x, 0); g.lineTo(x + H, H); g.stroke(); }
      }
      shieldEmblem(g, 58, 80, 62, 76, glowOnly ? '#000' : '#1a1f2b');
      g.fillStyle = '#f2b134';
      g.font = `700 34px ${FONT}`;
      g.textBaseline = 'middle';
      g.fillText('SHIELD', 102, 66);
      g.font = `600 22px ${FONT}`;
      g.fillStyle = glowOnly ? '#7a5a1a' : '#f2c564';
      g.fillText('IoT  IDS/IPS', 104, 100);
      return canvasTexture(c);
    };
    return { map: paint(false), glow: paint(true) };
  });
}

function raspberryTexture() {
  return sharedTex('raspberry', () => {
    const c = makeCanvas(64, 64), g = c.getContext('2d');
    g.fillStyle = '#4caf50';
    g.beginPath(); g.ellipse(24, 14, 10, 5, -0.5, 0, TAU); g.fill();
    g.beginPath(); g.ellipse(40, 14, 10, 5, 0.5, 0, TAU); g.fill();
    g.fillStyle = '#c51a4a';
    const berries = [[32, 26], [24, 32], [40, 32], [32, 38], [26, 45], [38, 45], [32, 52]];
    for (const [x, y] of berries) { g.beginPath(); g.arc(x, y, 6.5, 0, TAU); g.fill(); }
    return canvasTexture(c);
  });
}

function keyboardTexture() {
  return sharedTex('keyboard', () => {
    const W = 256, H = 100, c = makeCanvas(W, H), g = c.getContext('2d');
    g.fillStyle = '#8e95a0';
    g.fillRect(0, 0, W, H);
    g.fillStyle = '#1b1d22';
    const rows = 5, kw = W / 14.5;
    for (let r = 0; r < rows; r++) {
      const y = 6 + r * 18.5;
      let x = 4 + (r % 2) * 5;
      for (let k = 0; x < W - 10; k++) {
        const w = r === 4 && k === 3 ? kw * 5.4 : kw - 2.5;
        roundRect(g, x, y, w, 15, 2.5);
        g.fill();
        x += w + 2.5;
      }
    }
    return canvasTexture(c);
  });
}

function drumTexture() {
  return sharedTex('drum', () => {
    const S = 128, c = makeCanvas(S, S), g = c.getContext('2d');
    const grd = g.createRadialGradient(64, 64, 4, 64, 64, 64);
    grd.addColorStop(0, '#d7dbe0');
    grd.addColorStop(1, '#7d838c');
    g.fillStyle = grd;
    g.fillRect(0, 0, S, S);
    g.fillStyle = '#3b4048';
    for (let r = 10; r < 62; r += 9) {
      const n = Math.floor((TAU * r) / 9);
      for (let i = 0; i < n; i++) {
        const a = (i / n) * TAU;
        g.beginPath(); g.arc(64 + Math.cos(a) * r, 64 + Math.sin(a) * r, 1.6, 0, TAU); g.fill();
      }
    }
    return canvasTexture(c);
  });
}

function stickerTexture() {
  return sharedTex('sticker', () => {
    const c = makeCanvas(64, 80), g = c.getContext('2d');
    shieldEmblem(g, 32, 40, 54, 70);
    return canvasTexture(c);
  });
}

function hazardTexture() {
  return sharedTex('hazard', () => {
    const c = makeCanvas(64, 56), g = c.getContext('2d');
    g.fillStyle = '#ffd23f';
    g.strokeStyle = '#111';
    g.lineWidth = 4;
    g.lineJoin = 'round';
    g.beginPath(); g.moveTo(32, 4); g.lineTo(60, 52); g.lineTo(4, 52); g.closePath(); g.fill(); g.stroke();
    g.fillStyle = '#111';
    g.beginPath(); g.moveTo(35, 16); g.lineTo(24, 36); g.lineTo(32, 36); g.lineTo(28, 48); g.lineTo(41, 28); g.lineTo(33, 28); g.closePath(); g.fill();
    return canvasTexture(c);
  });
}

function tagTexture() {
  return sharedTex('tag', () => {
    const c = makeCanvas(64, 80), g = c.getContext('2d');
    g.fillStyle = '#f6efd9';
    g.beginPath();
    g.moveTo(32, 2); g.lineTo(60, 18); g.lineTo(60, 76); g.lineTo(4, 76); g.lineTo(4, 18); g.closePath();
    g.fill();
    g.fillStyle = '#5b4a2a';
    g.beginPath(); g.arc(32, 15, 4.5, 0, TAU); g.fill();
    g.fillStyle = '#e08a00';
    g.font = `800 48px ${FONT}`;
    g.textAlign = 'center';
    g.textBaseline = 'middle';
    g.fillText('?', 32, 50);
    return canvasTexture(c);
  });
}

/** Soft horizontal bands that scroll down the AC airflow sheet. */
function breezeImage() {
  return sharedTex('breeze', () => {
    const c = makeCanvas(32, 128), g = c.getContext('2d');
    const grd = g.createLinearGradient(0, 0, 0, 128);
    for (let i = 0; i <= 8; i++) grd.addColorStop(i / 8, i % 2 ? 'rgba(255,255,255,0.0)' : 'rgba(255,255,255,0.9)');
    g.fillStyle = grd;
    g.fillRect(0, 0, 32, 128);
    // Fade towards the far end so the sheet dissolves into the room.
    const fade = g.createLinearGradient(0, 0, 0, 128);
    fade.addColorStop(0, 'rgba(0,0,0,0)');
    fade.addColorStop(1, 'rgba(0,0,0,1)');
    g.globalCompositeOperation = 'destination-out';
    g.fillStyle = fade;
    g.fillRect(0, 0, 32, 128);
    return c;
  });
}

/** Bright spot on a dark strip: scrolled along the SHIELD light bar. */
function scanImage() {
  return sharedTex('scan', () => {
    const c = makeCanvas(128, 4), g = c.getContext('2d');
    const grd = g.createLinearGradient(0, 0, 128, 0);
    grd.addColorStop(0, 'rgb(40,40,40)');
    grd.addColorStop(0.38, 'rgb(70,70,70)');
    grd.addColorStop(0.5, 'rgb(255,255,255)');
    grd.addColorStop(0.62, 'rgb(70,70,70)');
    grd.addColorStop(1, 'rgb(40,40,40)');
    g.fillStyle = grd;
    g.fillRect(0, 0, 128, 4);
    return c;
  });
}

// ---------------------------------------------------------------------------
// Screen painters. Each paints a whole canvas; callers decide when to repaint.
// ---------------------------------------------------------------------------

// ---- TV channels (512 x 288) ------------------------------------------------

const CROWD_COLORS = ['#f2b134', '#5ad1e6', '#ff6b6b', '#ffffff', '#7dd87d', '#3b7ddb'];
let crowdDots = null;
function crowd() {
  if (!crowdDots) {
    const rnd = mulberry32(7);
    crowdDots = [];
    for (let i = 0; i < 300; i++) crowdDots.push([rnd() * 512, 40 + rnd() * 62, Math.floor(rnd() * CROWD_COLORS.length)]);
  }
  return crowdDots;
}

function figure(g, x, y, s, shirt) {
  g.fillStyle = '#f2f2f2';
  g.fillRect(x - 3 * s, y, 2.4 * s, 8 * s);
  g.fillRect(x + 0.6 * s, y, 2.4 * s, 8 * s);
  g.fillStyle = shirt;
  roundRect(g, x - 3.6 * s, y - 10 * s, 7.2 * s, 10.5 * s, 2 * s);
  g.fill();
  g.fillStyle = '#c68d63';
  g.beginPath(); g.arc(x, y - 13.2 * s, 3 * s, 0, TAU); g.fill();
}

function liveBadge(g, x, y, t) {
  g.fillStyle = '#e0242f';
  roundRect(g, x, y, 54, 20, 4);
  g.fill();
  g.fillStyle = (t % 1.2) < 0.8 ? '#fff' : 'rgba(255,255,255,0.35)';
  g.beginPath(); g.arc(x + 11, y + 10, 3.5, 0, TAU); g.fill();
  g.fillStyle = '#fff';
  g.font = `700 13px ${FONT}`;
  g.textBaseline = 'middle';
  g.textAlign = 'left';
  g.fillText('LIVE', x + 19, y + 10.5);
}

function paintCricket(g, w, h, t) {
  let grd = g.createLinearGradient(0, 0, 0, 120);
  grd.addColorStop(0, '#050a18');
  grd.addColorStop(1, '#1a2850');
  g.fillStyle = grd;
  g.fillRect(0, 0, w, h);
  for (let i = 0; i < 2; i++) {
    const x = i ? w - 80 : 80;
    const r = g.createRadialGradient(x, 12, 0, x, 12, 70);
    r.addColorStop(0, 'rgba(255,250,225,0.95)');
    r.addColorStop(1, 'rgba(255,250,225,0)');
    g.fillStyle = r;
    g.fillRect(x - 70, 0, 140, 82);
  }
  // Stands with a flickering crowd.
  g.fillStyle = '#1f2947';
  g.fillRect(0, 38, w, 68);
  const dots = crowd(), flick = Math.floor(t * 4);
  for (let i = 0; i < dots.length; i++) {
    const d = dots[i];
    g.globalAlpha = 0.4 + 0.6 * hash1(i * 3 + flick);
    g.fillStyle = CROWD_COLORS[d[2]];
    g.fillRect(d[0], d[1], 3, 4);
  }
  g.globalAlpha = 1;
  // Advertising boards.
  g.fillStyle = '#0c1430';
  g.fillRect(0, 104, w, 12);
  g.fillStyle = '#f2b134';
  g.font = `700 9px ${FONT}`;
  g.textBaseline = 'middle';
  g.textAlign = 'left';
  const adX = -((t * 30) % 200);
  for (let x = adX; x < w; x += 200) g.fillText('SHIELD-IoT  ·  SAFE SMART HOMES  ·', x, 110.5);
  // Outfield with mowing stripes.
  g.save();
  g.beginPath();
  g.ellipse(w / 2, h + 60, w * 0.82, h * 0.82, 0, 0, TAU);
  g.clip();
  for (let i = 0; i < 12; i++) {
    g.fillStyle = i % 2 ? '#2d8a39' : '#37a145';
    g.fillRect(0, 114 + i * 15, w, 15);
  }
  g.restore();
  // Pitch, creases and stumps.
  g.fillStyle = '#cdb47e';
  g.beginPath();
  g.moveTo(w / 2 - 15, 118); g.lineTo(w / 2 + 15, 118); g.lineTo(w / 2 + 38, 246); g.lineTo(w / 2 - 38, 246);
  g.closePath();
  g.fill();
  g.strokeStyle = 'rgba(255,255,255,0.9)';
  g.lineWidth = 1.5;
  g.beginPath();
  g.moveTo(w / 2 - 19, 126); g.lineTo(w / 2 + 19, 126);
  g.moveTo(w / 2 - 42, 230); g.lineTo(w / 2 + 42, 230);
  g.stroke();
  g.fillStyle = '#f4e3c0';
  for (let i = 0; i < 3; i++) g.fillRect(w / 2 - 4 + i * 3, 112, 1.6, 12);
  for (let i = 0; i < 3; i++) g.fillRect(w / 2 - 7 + i * 5, 210, 3, 20);
  // Fielders.
  const fielders = [[92, 150], [418, 142], [150, 220], [360, 214], [250, 166], [60, 236]];
  for (const [x, y] of fielders) figure(g, x, y, 0.75, '#1d3f8f');
  // One delivery every 4 s: run-up, ball down the pitch, shot to the boundary.
  const p = (t % 4) / 4, n = Math.floor(t / 4);
  const by = p < 0.42 ? 78 + (p / 0.42) * 40 : 118;
  figure(g, w / 2 + 9, by, 0.85, '#1d3f8f');
  const swing = p > 0.66 ? Math.min(1, (p - 0.66) / 0.08) : 0;
  figure(g, w / 2 - 15, 208, 1.4, '#2f6fdb');
  g.save();
  g.translate(w / 2 - 9, 200);
  g.rotate(-0.6 + swing * 2.1);
  g.fillStyle = '#e8d2a0';
  g.fillRect(-2, 0, 4, 22);
  g.restore();
  let bx = -1, bySp = 0, br = 3;
  if (p >= 0.42 && p < 0.7) {
    const f = (p - 0.42) / 0.28;
    bx = w / 2 + 6 - f * 14;
    bySp = 118 + f * 92 - Math.sin(Math.min(1, f * 1.4) * Math.PI) * 10;
    br = 2 + f * 2;
  } else if (p >= 0.7) {
    const f = (p - 0.7) / 0.3, dir = hash1(n) > 0.5 ? 1 : -1;
    bx = w / 2 - 10 + dir * f * 250;
    bySp = 204 - f * 110 + f * f * 60;
    br = 4 - f * 1.5;
  }
  if (bx > 0) {
    g.fillStyle = '#d81e2c';
    g.beginPath(); g.arc(bx, bySp, br, 0, TAU); g.fill();
  }
  // Score bug.
  const overs = 15 + Math.floor(n / 6), ball = n % 6;
  const runs = 142 + Math.floor(n * 1.45);
  const wickets = Math.min(9, 3 + Math.floor(n / 26));
  g.fillStyle = 'rgba(5,10,30,0.92)';
  roundRect(g, 10, h - 40, w - 20, 30, 6);
  g.fill();
  g.fillStyle = '#ff9933';
  roundRect(g, 14, h - 36, 50, 22, 4);
  g.fill();
  g.fillStyle = '#fff';
  g.font = `800 14px ${FONT}`;
  g.textAlign = 'center';
  g.fillText('IND', 39, h - 24.5);
  g.textAlign = 'left';
  g.font = `700 17px ${FONT}`;
  g.fillText(`${runs}/${wickets}`, 74, h - 24.5);
  g.font = `500 13px ${FONT}`;
  g.fillStyle = '#b9c6e6';
  g.fillText(`${overs}.${ball} ov`, 146, h - 24.5);
  g.fillStyle = '#f2b134';
  g.fillText(`CRR ${(runs / (overs + ball / 6)).toFixed(2)}`, 214, h - 24.5);
  g.textAlign = 'right';
  g.fillStyle = '#dfe6f5';
  g.fillText('Target 201 · T20', w - 20, h - 24.5);
  liveBadge(g, 12, 10, t);
  g.textAlign = 'right';
  g.font = `800 13px ${FONT}`;
  g.fillStyle = 'rgba(255,255,255,0.85)';
  g.fillText('SPORTS HD', w - 14, 20);
}

const NEWS_TICKER = 'Monsoon reaches Himachal two days early  •  Markets close higher  •  ' +
  'NIT Hamirpur: protocol invariants flag malformed IoT packets with zero false positives  •  ' +
  'Keep a device inventory, say security experts  •  ';
let newsTickerWidth = 0;

function paintNews(g, w, h, t) {
  const grd = g.createLinearGradient(0, 0, w, h);
  grd.addColorStop(0, '#123375');
  grd.addColorStop(1, '#071233');
  g.fillStyle = grd;
  g.fillRect(0, 0, w, h);
  g.fillStyle = 'rgba(120,170,255,0.07)';
  for (let i = 0; i < 6; i++) {
    g.beginPath();
    g.moveTo(80 + i * 90, 0); g.lineTo(130 + i * 90, 0); g.lineTo(40 + i * 90, h); g.lineTo(-10 + i * 90, h);
    g.closePath();
    g.fill();
  }
  // Anchor at the desk.
  g.fillStyle = '#1a2340';
  g.beginPath();
  g.moveTo(80, 200); g.quadraticCurveTo(84, 150, 140, 146); g.quadraticCurveTo(196, 150, 200, 200);
  g.closePath();
  g.fill();
  g.fillStyle = '#f3f3f3';
  g.beginPath(); g.moveTo(128, 147); g.lineTo(152, 147); g.lineTo(140, 172); g.closePath(); g.fill();
  g.fillStyle = '#b3262e';
  g.fillRect(138, 152, 4, 22);
  g.fillStyle = '#c68d63';
  g.beginPath(); g.arc(140, 120, 22, 0, TAU); g.fill();
  g.fillStyle = '#2a1d16';
  g.beginPath(); g.arc(140, 112, 23, Math.PI * 1.05, Math.PI * 1.95); g.fill();
  g.fillStyle = '#0d1d44';
  g.fillRect(40, 196, 220, 14);
  // Story panel.
  g.fillStyle = '#132a5c';
  roundRect(g, 290, 50, 205, 124, 8);
  g.fill();
  g.strokeStyle = '#4c78d0';
  g.lineWidth = 2;
  g.stroke();
  g.strokeStyle = '#dfe8ff';
  g.lineWidth = 4;
  g.lineJoin = 'round';
  g.beginPath();
  g.moveTo(330, 120); g.lineTo(365, 88); g.lineTo(400, 120); g.lineTo(400, 150); g.lineTo(330, 150); g.closePath();
  g.stroke();
  shieldEmblem(g, 440, 112, 46, 56, '#132a5c');
  g.fillStyle = '#dfe8ff';
  g.font = `700 12px ${FONT}`;
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  g.fillText('IoT SECURITY', 392, 164);
  // Lower third.
  g.fillStyle = '#d61f2c';
  g.fillRect(12, 212, 104, 28);
  g.fillStyle = '#fff';
  g.font = `800 14px ${FONT}`;
  g.fillText('BREAKING', 64, 226.5);
  g.fillStyle = '#f4f6fb';
  g.fillRect(116, 212, 384, 28);
  g.fillStyle = '#0b1736';
  g.font = `700 14px ${FONT}`;
  g.textAlign = 'left';
  g.fillText('Inline IDS keeps home IoT networks safe', 126, 226.5);
  // Ticker.
  g.fillStyle = '#050c22';
  g.fillRect(0, 246, w, 30);
  g.font = `600 13px ${FONT}`;
  if (!newsTickerWidth) newsTickerWidth = g.measureText(NEWS_TICKER).width;
  g.fillStyle = '#ffd36b';
  const x0 = -((t * 60) % newsTickerWidth);
  for (let x = x0; x < w; x += newsTickerWidth) g.fillText(NEWS_TICKER, x, 261.5);
  g.fillStyle = '#d61f2c';
  g.fillRect(w - 66, 246, 66, 30);
  g.fillStyle = '#fff';
  g.textAlign = 'center';
  g.font = `700 14px ${FONT}`;
  g.fillText(formatTime(simDaySeconds(t), false), w - 33, 261.5);
  g.fillStyle = '#d61f2c';
  g.fillRect(12, 12, 92, 26);
  g.fillStyle = '#fff';
  g.font = `800 16px ${FONT}`;
  g.fillText('NEWS 24', 58, 25.5);
  liveBadge(g, 112, 15, t);
}

const MOVIE_LINES = [
  '"The lights are on. Someone is home."',
  '"Every packet tells a story."',
  '"Trust the house, but check the guest list."',
  '"Not every device belongs here."',
];
let movieRidges = null;

function paintMovies(g, w, h, t) {
  const grd = g.createLinearGradient(0, 0, 0, h);
  grd.addColorStop(0, '#1a0f35');
  grd.addColorStop(0.45, '#6a2a5e');
  grd.addColorStop(0.68, '#ec7b4c');
  grd.addColorStop(0.8, '#ffd08a');
  g.fillStyle = grd;
  g.fillRect(0, 0, w, h);
  const sunY = 128 + 6 * Math.sin(t * 0.05);
  const sun = g.createRadialGradient(330, sunY, 10, 330, sunY, 120);
  sun.addColorStop(0, 'rgba(255,240,200,1)');
  sun.addColorStop(0.35, 'rgba(255,214,150,0.55)');
  sun.addColorStop(1, 'rgba(255,190,120,0)');
  g.fillStyle = sun;
  g.fillRect(200, 40, 260, 230);
  g.fillStyle = '#fff1cf';
  g.beginPath(); g.arc(330, sunY, 38, 0, TAU); g.fill();
  if (!movieRidges) {
    const rnd = mulberry32(21);
    movieRidges = [0, 1, 2].map(layer => {
      const pts = [];
      for (let x = 0; x <= 512; x += 32) pts.push(x, 168 + layer * 18 - rnd() * (48 - layer * 12));
      return pts;
    });
  }
  const fills = ['#3a1d4d', '#26123a', '#140a22'];
  movieRidges.forEach((pts, i) => {
    g.fillStyle = fills[i];
    g.beginPath();
    g.moveTo(0, h);
    for (let k = 0; k < pts.length; k += 2) g.lineTo(pts[k], pts[k + 1]);
    g.lineTo(w, h);
    g.closePath();
    g.fill();
  });
  // A car crossing on the road.
  g.fillStyle = '#0b0611';
  g.fillRect(0, 214, w, 12);
  const cx = ((t * 34) % (w + 120)) - 60;
  g.fillStyle = '#05030a';
  roundRect(g, cx - 26, 200, 52, 14, 4); g.fill();
  roundRect(g, cx - 14, 191, 26, 12, 4); g.fill();
  g.fillStyle = '#ffcf6b';
  g.fillRect(cx + 24, 204, 4, 3);
  // Birds.
  g.strokeStyle = '#1b0e22';
  g.lineWidth = 1.6;
  for (let i = 0; i < 3; i++) {
    const bx = ((t * 12 + i * 40) % (w + 60)) - 30, by = 70 + i * 9 + 3 * Math.sin(t * 2 + i);
    const flap = 3 + 2 * Math.sin(t * 6 + i);
    g.beginPath(); g.moveTo(bx - 6, by - flap); g.lineTo(bx, by); g.lineTo(bx + 6, by - flap); g.stroke();
  }
  // Letterbox and subtitles.
  g.fillStyle = '#000';
  g.fillRect(0, 0, w, 32);
  g.fillRect(0, h - 40, w, 40);
  g.fillStyle = '#f5f5f5';
  g.font = `italic 500 15px ${FONT}`;
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  g.fillText(MOVIE_LINES[Math.floor(t / 5) % MOVIE_LINES.length], w / 2, h - 20);
}

function paintCartoons(g, w, h, t) {
  const grd = g.createLinearGradient(0, 0, 0, h);
  grd.addColorStop(0, '#4fc3ff');
  grd.addColorStop(1, '#c9f2ff');
  g.fillStyle = grd;
  g.fillRect(0, 0, w, h);
  // Smiling sun with turning rays.
  g.save();
  g.translate(440, 60);
  g.rotate(t * 0.4);
  g.fillStyle = '#ffd23f';
  for (let i = 0; i < 12; i++) {
    g.rotate(TAU / 12);
    g.beginPath(); g.moveTo(-6, -40); g.lineTo(0, -58); g.lineTo(6, -40); g.closePath(); g.fill();
  }
  g.restore();
  g.fillStyle = '#ffd23f';
  g.beginPath(); g.arc(440, 60, 34, 0, TAU); g.fill();
  g.fillStyle = '#5a3b00';
  g.beginPath(); g.arc(430, 54, 4, 0, TAU); g.arc(450, 54, 4, 0, TAU); g.fill();
  g.strokeStyle = '#5a3b00';
  g.lineWidth = 3;
  g.beginPath(); g.arc(440, 64, 12, 0.2, Math.PI - 0.2); g.stroke();
  // Drifting clouds.
  g.fillStyle = '#ffffff';
  for (let i = 0; i < 3; i++) {
    const x = ((i * 190 + t * (14 + i * 4)) % (w + 140)) - 70, y = 50 + i * 26;
    g.beginPath();
    g.arc(x, y, 18, 0, TAU); g.arc(x + 20, y - 8, 22, 0, TAU); g.arc(x + 42, y, 17, 0, TAU);
    g.fill();
  }
  // Hills and flowers.
  g.fillStyle = '#6fd66f';
  g.beginPath(); g.ellipse(120, h + 40, 260, 120, 0, 0, TAU); g.fill();
  g.fillStyle = '#4cbc58';
  g.beginPath(); g.ellipse(420, h + 60, 260, 120, 0, 0, TAU); g.fill();
  const petals = ['#ff6fb5', '#ffffff', '#ffb347', '#b06cff'];
  for (let i = 0; i < 7; i++) {
    const x = 40 + i * 70, y = 250 + (i % 2) * 14;
    g.fillStyle = petals[i % petals.length];
    g.beginPath(); g.arc(x, y, 6, 0, TAU); g.fill();
    g.fillStyle = '#ffe66b';
    g.beginPath(); g.arc(x, y, 2.5, 0, TAU); g.fill();
  }
  // Bouncing pink blob with googly eyes.
  const hop = Math.abs(Math.sin(t * 3));
  const bx = 220 + 120 * Math.sin(t * 0.7), by = 226 - hop * 80;
  const squash = 1 + 0.28 * Math.pow(1 - hop, 6);
  g.fillStyle = 'rgba(0,0,0,0.12)';
  g.beginPath(); g.ellipse(bx, 252, 30 - hop * 10, 6, 0, 0, TAU); g.fill();
  g.fillStyle = '#ff6fb5';
  g.beginPath(); g.ellipse(bx, by, 30 * squash, 30 / squash, 0, 0, TAU); g.fill();
  const look = Math.cos(t * 0.7) > 0 ? 3 : -3;
  g.fillStyle = '#fff';
  g.beginPath(); g.arc(bx - 10, by - 8, 9, 0, TAU); g.arc(bx + 10, by - 8, 9, 0, TAU); g.fill();
  g.fillStyle = '#1b1b2f';
  g.beginPath(); g.arc(bx - 10 + look, by - 7, 4, 0, TAU); g.arc(bx + 10 + look, by - 7, 4, 0, TAU); g.fill();
  g.strokeStyle = '#8a1f55';
  g.lineWidth = 3;
  g.beginPath(); g.arc(bx, by + 6, 9, 0.2, Math.PI - 0.2); g.stroke();
  // A spinning star.
  g.save();
  g.translate(80, 90 + 10 * Math.sin(t * 2));
  g.rotate(t);
  g.fillStyle = '#ffe14d';
  g.beginPath();
  for (let i = 0; i < 10; i++) {
    const r = i % 2 ? 9 : 20, a = (i / 10) * TAU;
    g.lineTo(Math.sin(a) * r, -Math.cos(a) * r);
  }
  g.closePath();
  g.fill();
  g.restore();
  // Channel logo.
  const letters = 'TOONS', colours = ['#ff5a5a', '#ffb347', '#ffe14d', '#6fd66f', '#5ab0ff'];
  g.font = `900 22px ${FONT}`;
  g.textAlign = 'left';
  g.textBaseline = 'middle';
  for (let i = 0; i < letters.length; i++) {
    g.fillStyle = colours[i];
    g.fillText(letters[i], 14 + i * 17, 24 + 2 * Math.sin(t * 4 + i));
  }
}

function paintDashboard(g, w, h, t, series, head) {
  g.fillStyle = '#0a0f1e';
  g.fillRect(0, 0, w, h);
  g.fillStyle = '#121a30';
  g.fillRect(0, 0, w, 42);
  shieldEmblem(g, 26, 21, 22, 28, '#121a30');
  g.fillStyle = '#f2b134';
  g.font = `700 19px ${FONT}`;
  g.textAlign = 'left';
  g.textBaseline = 'middle';
  g.fillText('SHIELD Dashboard', 46, 22);
  const ids = globalThis.__shield?.state?.ids;
  const m = ids?.metrics;
  const mode = ids?.mode === 'detect' ? 'DETECT' : 'PREVENT';
  g.fillStyle = mode === 'PREVENT' ? 'rgba(61,220,151,0.18)' : 'rgba(255,179,71,0.18)';
  roundRect(g, w - 104, 11, 92, 22, 11);
  g.fill();
  g.fillStyle = mode === 'PREVENT' ? '#3ddc97' : '#ffb347';
  g.font = `700 12px ${FONT}`;
  g.textAlign = 'center';
  g.fillText(mode, w - 58, 22.5);
  // Throughput chart.
  const cx = 14, cy = 54, cw = 300, ch = 168;
  g.fillStyle = '#0f1628';
  roundRect(g, cx, cy, cw, ch, 8);
  g.fill();
  g.strokeStyle = 'rgba(160,190,255,0.08)';
  g.lineWidth = 1;
  for (let i = 1; i < 4; i++) {
    g.beginPath(); g.moveTo(cx + 8, cy + (ch * i) / 4); g.lineTo(cx + cw - 8, cy + (ch * i) / 4); g.stroke();
  }
  const n = series.length;
  let max = 10;
  for (let i = 0; i < n; i++) if (series[i] > max) max = series[i];
  max *= 1.2;
  g.beginPath();
  for (let i = 0; i < n; i++) {
    const v = series[(head + i) % n];
    const x = cx + 10 + ((cw - 20) * i) / (n - 1), y = cy + ch - 12 - ((ch - 34) * v) / max;
    if (i === 0) g.moveTo(x, y); else g.lineTo(x, y);
  }
  g.strokeStyle = '#5ad1e6';
  g.lineWidth = 2.2;
  g.stroke();
  g.lineTo(cx + cw - 10, cy + ch - 12);
  g.lineTo(cx + 10, cy + ch - 12);
  g.closePath();
  g.fillStyle = 'rgba(90,209,230,0.16)';
  g.fill();
  g.fillStyle = '#8ea0c4';
  g.font = `600 11px ${FONT}`;
  g.textAlign = 'left';
  g.fillText('packets / s', cx + 12, cy + 14);
  // Stat tiles.
  const pps = m ? Math.round(m.pps) : Math.round(series[(head + n - 1) % n]);
  const alerts = ids ? ids.alerts.length : 0;
  const blocked = ids ? ids.blocked.size : 0;
  const threat = (m?.threat || 'low').toUpperCase();
  const threatCol = threat === 'LOW' ? '#3ddc97' : threat === 'ELEVATED' ? '#ffb347' : '#ff4d5e';
  const tiles = [['Packets/s', String(pps), '#5ad1e6'], ['Alerts', String(alerts), '#ffb347'], ['Blocked', String(blocked), '#ff4d5e'], ['Threat', threat, threatCol]];
  for (let i = 0; i < tiles.length; i++) {
    const ty = 54 + i * 43;
    g.fillStyle = '#0f1628';
    roundRect(g, 326, ty, 172, 37, 7);
    g.fill();
    g.fillStyle = '#8ea0c4';
    g.font = `600 11px ${FONT}`;
    g.textAlign = 'left';
    g.fillText(tiles[i][0], 336, ty + 19);
    g.fillStyle = tiles[i][2];
    g.font = `800 17px ${FONT}`;
    g.textAlign = 'right';
    g.fillText(tiles[i][1], 488, ty + 19.5);
  }
  const b = RESEARCH.baseline;
  g.fillStyle = '#6f7fa3';
  g.font = `600 11px ${FONT}`;
  g.textAlign = 'left';
  g.fillText(`${b.model} baseline · macro-F1 ${b.macroF1.toFixed(4)} · ROC-AUC ${b.rocAuc.toFixed(4)} · ${b.latencyUs} µs/flow`, 14, h - 34);
  g.fillStyle = '#3ddc97';
  g.beginPath(); g.arc(20, h - 15, 4, 0, TAU); g.fill();
  g.fillStyle = '#8ea0c4';
  g.fillText(`Inline at 192.168.0.1 · ${formatTime(simDaySeconds(t), false)} IST`, 30, h - 14.5);
}

const CHANNEL_PAINTERS = {
  'Cricket Live': paintCricket,
  News: paintNews,
  Movies: paintMovies,
  Cartoons: paintCartoons,
};

function paintChannelBanner(g, w, name, index) {
  g.fillStyle = 'rgba(5,8,18,0.82)';
  roundRect(g, w / 2 - 110, 112, 220, 50, 10);
  g.fill();
  g.fillStyle = '#f2b134';
  g.font = `700 13px ${FONT}`;
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  g.fillText(`CH ${index + 1}`, w / 2, 127);
  g.fillStyle = '#fff';
  g.font = `700 18px ${FONT}`;
  g.fillText(name, w / 2, 148);
}

// ---- small appliance displays --------------------------------------------------

function paintClockLine(g, w, h, text) {
  g.fillStyle = '#04090c';
  g.fillRect(0, 0, w, h);
  g.fillStyle = '#7df9ff';
  g.font = `700 28px ${MONO}`;
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  g.fillText(text, w / 2 - 14, h / 2 + 1);
  g.fillStyle = '#3fa6b0';
  g.font = `700 13px ${FONT}`;
  g.fillText('IST', w - 20, h / 2 + 1);
}

function paintAc(g, w, h, on, temp, mode) {
  g.fillStyle = '#05070a';
  g.fillRect(0, 0, w, h);
  if (!on) return;
  g.fillStyle = '#7fd8ff';
  g.font = `700 34px ${MONO}`;
  g.textAlign = 'left';
  g.textBaseline = 'middle';
  g.fillText(`${Math.round(temp)}°`, 8, h / 2 + 2);
  g.font = `700 13px ${FONT}`;
  g.textAlign = 'right';
  g.fillStyle = mode === 'Cool' ? '#7fd8ff' : mode === 'Dry' ? '#8ff0d8' : '#e8eef7';
  g.fillText(String(mode).toUpperCase(), w - 8, h / 2 + 1);
}

function paintFridge(g, w, h, temp, open) {
  g.fillStyle = '#060a12';
  g.fillRect(0, 0, w, h);
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  g.fillStyle = '#8fd3ff';
  g.font = `700 13px ${FONT}`;
  g.fillText('FRIDGE', w / 2, 14);
  g.fillStyle = '#ffffff';
  g.font = `700 36px ${FONT}`;
  g.fillText(`${temp}°C`, w / 2, h / 2 + 4);
  if (open) {
    g.fillStyle = '#ffb347';
    roundRect(g, 10, h - 24, w - 20, 18, 5);
    g.fill();
    g.fillStyle = '#1a1206';
    g.font = `800 11px ${FONT}`;
    g.fillText('DOOR OPEN', w / 2, h - 14.5);
  } else {
    g.strokeStyle = '#5ab8ff';
    g.lineWidth = 2;
    const sx = w / 2, sy = h - 15;
    for (let i = 0; i < 3; i++) {
      const a = (i / 3) * Math.PI;
      g.beginPath();
      g.moveTo(sx - Math.cos(a) * 7, sy - Math.sin(a) * 7);
      g.lineTo(sx + Math.cos(a) * 7, sy + Math.sin(a) * 7);
      g.stroke();
    }
  }
}

function paintWasher(g, w, h, running, minutes, program) {
  g.fillStyle = '#05070a';
  g.fillRect(0, 0, w, h);
  g.textBaseline = 'middle';
  const done = !running && minutes <= 0;
  g.fillStyle = running ? '#ffb347' : '#b9895a';
  g.font = `700 26px ${MONO}`;
  g.textAlign = 'left';
  const m = Math.max(0, Math.round(minutes));
  g.fillText(done ? 'End' : `${Math.floor(m / 60)}:${m % 60 < 10 ? '0' : ''}${m % 60}`, 6, h / 2 + 2);
  g.font = `600 10px ${FONT}`;
  g.textAlign = 'right';
  g.fillStyle = '#d7dde8';
  g.fillText(String(program || ''), w - 6, 14);
  g.fillStyle = running ? '#3ddc97' : '#6b7280';
  g.fillText(running ? 'RUNNING' : done ? 'DONE' : 'READY', w - 6, h - 12);
}

function paintThermostat(g, s, current, target) {
  const c = s / 2;
  const cooling = target < current - 0.15, heating = target > current + 0.15;
  const accent = cooling ? '#3b9bff' : heating ? '#ff7a2f' : '#9aa3b2';
  g.fillStyle = '#04060a';
  g.fillRect(0, 0, s, s);
  const grd = g.createRadialGradient(c, c, s * 0.05, c, c, s * 0.5);
  grd.addColorStop(0, cooling ? '#0f2f5e' : heating ? '#4d2510' : '#161b24');
  grd.addColorStop(1, '#04060a');
  g.fillStyle = grd;
  g.beginPath(); g.arc(c, c, s * 0.5, 0, TAU); g.fill();
  // Tick ring from 16 to 30 °C over 270 degrees; the span between now and target lights up.
  const angleOf = v => Math.PI * 0.75 + ((clamp(v, 16, 30) - 16) / 14) * Math.PI * 1.5;
  const a0 = Math.min(angleOf(current), angleOf(target)), a1 = Math.max(angleOf(current), angleOf(target));
  for (let i = 0; i <= 70; i++) {
    const a = Math.PI * 0.75 + (i / 70) * Math.PI * 1.5;
    const lit = a >= a0 - 0.01 && a <= a1 + 0.01;
    g.strokeStyle = lit ? accent : 'rgba(255,255,255,0.22)';
    g.lineWidth = lit ? 2.6 : 1.6;
    g.beginPath();
    g.moveTo(c + Math.cos(a) * s * 0.4, c + Math.sin(a) * s * 0.4);
    g.lineTo(c + Math.cos(a) * s * 0.46, c + Math.sin(a) * s * 0.46);
    g.stroke();
  }
  const at = angleOf(target);
  g.strokeStyle = '#ffffff';
  g.lineWidth = 4;
  g.beginPath();
  g.moveTo(c + Math.cos(at) * s * 0.37, c + Math.sin(at) * s * 0.37);
  g.lineTo(c + Math.cos(at) * s * 0.48, c + Math.sin(at) * s * 0.48);
  g.stroke();
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  g.fillStyle = accent;
  g.font = `700 ${Math.round(s * 0.085)}px ${FONT}`;
  g.fillText(cooling ? 'COOLING' : heating ? 'HEATING' : 'HOLD', c, c - s * 0.2);
  g.fillStyle = '#ffffff';
  g.font = `600 ${Math.round(s * 0.28)}px ${FONT}`;
  g.fillText(Number(target).toFixed(1), c, c + s * 0.01);
  g.fillStyle = '#c9d1dd';
  g.font = `500 ${Math.round(s * 0.085)}px ${FONT}`;
  g.fillText(`Now ${Number(current).toFixed(1)}°`, c, c + s * 0.22);
}

function paintWatch(g, s, hr, steps, bigHeart, time) {
  const c = s / 2;
  g.fillStyle = '#000';
  g.fillRect(0, 0, s, s);
  g.lineCap = 'round';
  g.lineWidth = s * 0.06;
  g.strokeStyle = '#183326';
  g.beginPath(); g.arc(c, c, s * 0.44, 0, TAU); g.stroke();
  g.strokeStyle = '#3ddc97';
  g.beginPath(); g.arc(c, c, s * 0.44, -Math.PI / 2, -Math.PI / 2 + clamp(steps / 10000, 0, 1) * TAU); g.stroke();
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  g.fillStyle = '#9fb0c8';
  g.font = `600 ${Math.round(s * 0.1)}px ${FONT}`;
  g.fillText(time, c, s * 0.2);
  heartPath(g, c, s * 0.38, bigHeart ? s * 0.13 : s * 0.1);
  g.fillStyle = '#ff4d6d';
  g.fill();
  g.fillStyle = '#ffffff';
  g.font = `700 ${Math.round(s * 0.26)}px ${FONT}`;
  g.fillText(String(hr), c, s * 0.63);
  g.fillStyle = '#ff8fa3';
  g.font = `700 ${Math.round(s * 0.085)}px ${FONT}`;
  g.fillText('BPM', c, s * 0.8);
}

function paintMeter(g, w, h, kw) {
  g.fillStyle = '#030805';
  g.fillRect(0, 0, w, h);
  const text = Math.min(99.99, Math.max(0, Number(kw) || 0)).toFixed(2).padStart(5, ' ');
  const on = '#5dff8f', off = 'rgba(93,255,143,0.07)';
  let x = 14;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (ch === '.') {
      g.fillStyle = on;
      g.beginPath(); g.arc(x - 6, 78, 4, 0, TAU); g.fill();
      continue;
    }
    sevenSeg(g, ch === ' ' ? '8' : ch, x, 16, 30, 62, 7, ch === ' ' ? off : on, off);
    x += 46;
  }
  g.fillStyle = '#5dff8f';
  g.font = `700 22px ${FONT}`;
  g.textAlign = 'left';
  g.textBaseline = 'middle';
  g.fillText('kW', 204, 66);
  g.font = `600 11px ${FONT}`;
  g.fillStyle = '#2f9e5a';
  g.fillText('IMPORT', 204, 26);
}

// Laptop terminal: lines typed out over time, numbers quoted from the research book.
const TERMINAL_LINES = [
  '$ ssh pi@192.168.0.10',
  'pi@hub:~ $ mosquitto_sub -v -t "home/#"',
  'home/light-living/state {"power":true,"brightness":80}',
  'home/fridge/state {"temp":4,"door":"closed"}',
  'home/thermostat/state {"current":25.1,"target":23.5}',
  '^C',
  'pi@hub:~ $ exit',
  '$ cd ~/shield-iot && python eval.py --seed 42',
  `test split: ${RESEARCH.baseline.test} flows, ${RESEARCH.baseline.classes} classes`,
  `${RESEARCH.baseline.model}  macro-F1 ${RESEARCH.baseline.macroF1.toFixed(4)}  ROC-AUC ${RESEARCH.baseline.rocAuc.toFixed(4)}`,
  `INV_TCP_02 hits ${RESEARCH.invariants.INV_TCP_02.test}  precision ${RESEARCH.tcpEngineTest.precision.toFixed(3)}`,
  `[OK] latency ${RESEARCH.baseline.latencyUs} us/flow`,
  '$ git status',
  'On branch main, nothing to commit',
];
let terminalStarts = null;   // cumulative character offsets, built on first use
const TERMINAL_CPS = 24;     // characters typed per second
const TERMINAL_PAUSE = 4;    // seconds to rest on the finished screen before looping

function terminalOffsets() {
  if (!terminalStarts) {
    terminalStarts = [];
    let sum = 0;
    for (const line of TERMINAL_LINES) { terminalStarts.push(sum); sum += line.length + 6; }
    terminalStarts.push(sum);
  }
  return terminalStarts;
}

/** Characters typed so far at sim time t (the session loops after a pause). */
function terminalCursor(t) {
  const starts = terminalOffsets();
  const total = starts[starts.length - 1];
  return Math.floor(((t % (total / TERMINAL_CPS + TERMINAL_PAUSE)) * TERMINAL_CPS));
}

function paintTerminal(g, w, h, typed, wifi, blink) {
  g.fillStyle = '#0c1016';
  g.fillRect(0, 0, w, h);
  g.fillStyle = '#1b2230';
  g.fillRect(0, 0, w, 22);
  const dots = ['#ff5f57', '#febc2e', '#28c840'];
  for (let i = 0; i < 3; i++) { g.fillStyle = dots[i]; g.beginPath(); g.arc(14 + i * 16, 11, 5, 0, TAU); g.fill(); }
  g.fillStyle = '#8b96aa';
  g.font = `600 12px ${FONT}`;
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  g.fillText('anirudh@study: ~/shield-iot', w / 2, 11.5);
  g.font = `14px ${MONO}`;
  g.textAlign = 'left';
  const starts = terminalOffsets();
  const lines = [];
  let n = 0;
  while (n < TERMINAL_LINES.length && starts[n] <= typed) n++;
  for (let i = 0; i < n; i++) {
    const line = TERMINAL_LINES[i];
    lines.push(i === n - 1 ? line.slice(0, Math.max(0, typed - starts[i])) : line);
  }
  if (!wifi) lines.push('!ssh: connect to host 192.168.0.10: Network is unreachable');
  const rows = 15, start = Math.max(0, lines.length - rows);
  for (let i = start; i < lines.length; i++) {
    const s = lines[i], y = 36 + (i - start) * 18.5;
    if (s.startsWith('!')) { g.fillStyle = '#ff6b6b'; g.fillText(s.slice(1), 10, y); continue; }
    if (s.startsWith('$') || s.startsWith('pi@')) {
      const cut = s.indexOf('$') + 1;
      g.fillStyle = '#3ddc97';
      g.fillText(s.slice(0, cut), 10, y);
      g.fillStyle = '#e6edf3';
      g.fillText(s.slice(cut), 10 + g.measureText(s.slice(0, cut)).width, y);
    } else {
      g.fillStyle = s.startsWith('[OK]') ? '#3ddc97' : s.startsWith('LightGBM') ? '#f2b134' : '#9fb3c8';
      g.fillText(s, 10, y);
    }
    if (i === lines.length - 1 && blink) {
      g.fillStyle = '#e6edf3';
      g.fillRect(12 + g.measureText(s).width, y - 8, 8, 16);
    }
  }
}

function paintPhone(g, w, h, info) {
  const grd = g.createLinearGradient(0, 0, 0, h);
  grd.addColorStop(0, info.ringing ? '#1d2a66' : '#121b36');
  grd.addColorStop(1, '#070b18');
  g.fillStyle = grd;
  g.fillRect(0, 0, w, h);
  g.textBaseline = 'middle';
  g.fillStyle = '#ffffff';
  g.font = `600 13px ${FONT}`;
  g.textAlign = 'left';
  g.fillText(info.time, 12, 14);
  g.fillStyle = '#000';
  g.beginPath(); g.arc(w / 2, 14, 5, 0, TAU); g.fill();
  // Battery and Wi-Fi.
  g.strokeStyle = '#ffffff';
  g.lineWidth = 1.4;
  g.strokeRect(w - 30, 9, 18, 10);
  g.fillStyle = '#ffffff';
  g.fillRect(w - 28, 11, 11, 6);
  g.fillRect(w - 11, 12, 2, 4);
  if (info.wifi) {
    g.strokeStyle = '#ffffff';
    for (let i = 1; i <= 3; i++) { g.beginPath(); g.arc(w - 44, 20, i * 3.5, -Math.PI * 0.75, -Math.PI * 0.25); g.stroke(); }
  } else {
    g.fillStyle = '#ff6b6b';
    g.font = `700 10px ${FONT}`;
    g.textAlign = 'right';
    g.fillText('No Wi-Fi', w - 36, 14);
  }
  g.textAlign = 'center';
  if (info.ringing) {
    for (let i = 3; i >= 1; i--) {
      g.fillStyle = `rgba(90,209,230,${0.08 * i})`;
      g.beginPath(); g.arc(w / 2, 120, 22 + i * 16, 0, TAU); g.fill();
    }
    g.fillStyle = '#5ad1e6';
    g.beginPath(); g.arc(w / 2, 120, 26, 0, TAU); g.fill();
    g.fillStyle = '#0b1530';
    g.beginPath();
    g.moveTo(w / 2 - 12, 128); g.quadraticCurveTo(w / 2 - 12, 106, w / 2, 106); g.quadraticCurveTo(w / 2 + 12, 106, w / 2 + 12, 128);
    g.closePath();
    g.fill();
    g.fillRect(w / 2 - 14, 127, 28, 3);
    g.beginPath(); g.arc(w / 2, 133, 3, 0, TAU); g.fill();
    g.fillStyle = '#ffffff';
    g.font = `700 17px ${FONT}`;
    g.fillText('Find my phone', w / 2, 196);
    g.fillStyle = '#9fb3d8';
    g.font = `500 12px ${FONT}`;
    g.fillText('Smartwatch is ringing', w / 2, 218);
    g.fillStyle = '#ff4d5e';
    roundRect(g, w / 2 - 44, h - 64, 88, 30, 15);
    g.fill();
    g.fillStyle = '#ffffff';
    g.font = `700 13px ${FONT}`;
    g.fillText('Dismiss', w / 2, h - 48.5);
    return;
  }
  g.textAlign = 'left';
  g.fillStyle = '#8ea0c4';
  g.font = `500 11px ${FONT}`;
  g.fillText('Good evening', 12, 44);
  g.fillStyle = '#ffffff';
  g.font = `700 18px ${FONT}`;
  g.fillText('Anirudh', 12, 63);
  const tiles = info.tiles;
  for (let i = 0; i < tiles.length; i++) {
    const [label, value, on, accent] = tiles[i];
    const x = 10 + (i % 2) * ((w - 26) / 2 + 6), y = 84 + Math.floor(i / 2) * 64, tw = (w - 26) / 2, th = 56;
    g.fillStyle = on ? '#1f2d57' : '#141b30';
    roundRect(g, x, y, tw, th, 9);
    g.fill();
    if (accent) { g.strokeStyle = accent; g.lineWidth = 1.5; g.stroke(); }
    g.fillStyle = on ? (accent || '#5ad1e6') : '#4a5675';
    g.beginPath(); g.arc(x + 13, y + 15, 5, 0, TAU); g.fill();
    g.fillStyle = '#dfe6f5';
    g.font = `600 11px ${FONT}`;
    g.fillText(label, x + 8, y + 33);
    g.fillStyle = on ? '#ffffff' : '#7d89a6';
    g.font = `700 11px ${FONT}`;
    g.fillText(value, x + 8, y + 47);
  }
  g.fillStyle = 'rgba(255,255,255,0.5)';
  roundRect(g, w / 2 - 30, h - 10, 60, 4, 2);
  g.fill();
}

/** Tiles for the phone's home app, read from the live state when the app is running. */
function phoneTiles() {
  const devs = globalThis.__shield?.state?.devices;
  const p = id => devs?.get(id)?.props;
  const light = p('light-living'), fan = p('fan'), lock = p('lock'), tv = p('tv'), ac = p('ac');
  const mode = globalThis.__shield?.state?.ids?.mode;
  return [
    ['Lights', light ? (light.power ? `On · ${light.brightness}%` : 'Off') : 'On · 80%', light ? !!light.power : true],
    ['Fan', fan ? (fan.power ? `Speed ${fan.speed}` : 'Off') : 'Speed 3', fan ? !!fan.power : true],
    ['Front door', lock ? (lock.locked ? 'Locked' : 'Unlocked') : 'Locked', true, lock && !lock.locked ? '#ff4d5e' : null],
    ['TV', tv ? (tv.power ? String(tv.channel) : 'Off') : 'Cricket Live', tv ? !!tv.power : true],
    ['AC', ac ? (ac.power ? `${ac.temp}°C ${ac.mode}` : 'Off') : 'Off', ac ? !!ac.power : false],
    ['SHIELD-IoT', mode === 'detect' ? 'Detect only' : 'Protected', true, '#f2b134'],
  ];
}

// ---------------------------------------------------------------------------
// Per-instance builder kit
// ---------------------------------------------------------------------------

class Kit {
  constructor(device) {
    this.device = device;
    this.group = new THREE.Group();
    this.group.name = `device:${device.id}`;
    this.body = new THREE.Group();
    this.group.add(this.body);
    this.port = new THREE.Vector3(0, 0.1, 0);
    this.fxParent = this.body;     // status effects attach here (the vacuum's moves)
    this.mount = 'floor';          // 'floor' | 'wall' | 'ceiling': orients the status rings
    this.dim = 1;                  // < 1 while offline; multiplies every dynamic glow
    this.owned = [];               // per-instance GPU resources freed by dispose()
    this.mats = [];                // per-instance materials restyled by setStatus()
    this.pending = [];             // decorations attached on the first update (see decor())
  }

  own(resource) {
    this.owned.push(resource);
    return resource;
  }

  /** Per-instance standard material. dynamic: its emissive is driven every frame by update(). */
  mat(params, dynamic = false) {
    const m = new THREE.MeshStandardMaterial(params);
    m.userData.dynamic = dynamic;
    m.userData.baseEI = m.emissiveIntensity;
    m.userData.baseColor = m.color.getHex();
    this.mats.push(m);
    return this.own(m);
  }

  add(parent, geo, mat, x = 0, y = 0, z = 0, rx = 0, ry = 0, rz = 0) {
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(x, y, z);
    if (rx || ry || rz) mesh.rotation.set(rx, ry, rz);
    parent.add(mesh);
    return mesh;
  }

  pivot(parent, x = 0, y = 0, z = 0) {
    const g = new THREE.Group();
    g.position.set(x, y, z);
    parent.add(g);
    return g;
  }

  /** Canvas-backed emissive screen (dynamic material; glow driven by update()). */
  screen(w, h, intensity = 1.15) {
    const canvas = makeCanvas(w, h);
    const ctx = canvas.getContext('2d');
    const tex = this.own(screenTexture(canvas));
    const mat = this.mat({ color: 0x050608, roughness: 0.22, metalness: 0.1, emissive: 0xffffff, emissiveMap: tex, emissiveIntensity: intensity }, true);
    return { canvas, ctx, tex, mat, w, h };
  }

  /**
   * Decorations (glow sprites, airflow, blur discs) that must not count towards
   * the size scene.js measures when it builds the device: they are attached on
   * the first update() and excluded from the status-effect bounds.
   */
  decor(obj, parent) {
    obj.userData.noBounds = true;
    this.pending.push(obj, parent);
    return obj;
  }

  attachPending() {
    for (let i = 0; i < this.pending.length; i += 2) this.pending[i + 1].add(this.pending[i]);
    this.pending.length = 0;
  }

  /** Additive glow sprite with its own material (opacity animated per instance). */
  glowSprite(parent, hex, size, opacity, x, y, z) {
    const m = this.own(new THREE.SpriteMaterial({
      map: glowTexture(), color: hex, transparent: true, opacity, depthWrite: false, blending: THREE.AdditiveBlending,
    }));
    const s = new THREE.Sprite(m);
    s.scale.set(size, size, 1);
    s.position.set(x, y, z);
    return this.decor(s, parent);
  }

  glow(mat, intensity) {
    mat.emissiveIntensity = intensity * this.dim;
  }

  dispose() {
    for (let i = 0; i < this.owned.length; i++) this.owned[i].dispose();
    this.owned.length = 0;
  }
}

// ---------------------------------------------------------------------------
// Status effects: alert halo, blocked cage, offline greying (one shared helper)
// ---------------------------------------------------------------------------

/** Flat unit annulus in the xz plane with angular UVs (u runs around the ring). */
function bandGeometry() {
  return sharedGeo('fx-band', () => {
    const segs = 72, inner = 0.9;
    const pos = [], uv = [], idx = [];
    for (let i = 0; i <= segs; i++) {
      const a = (i / segs) * TAU, c = Math.cos(a), s = Math.sin(a);
      pos.push(c * inner, 0, s * inner, c, 0, s);
      uv.push(i / segs, 0, i / segs, 1);
    }
    for (let i = 0; i < segs; i++) {
      const a = i * 2;
      idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
    g.setIndex(idx);
    g.computeVertexNormals();
    return g;
  });
}

function dashTexture() {
  return sharedTex('dash', () => {
    const c = makeCanvas(64, 8), g = c.getContext('2d');
    g.fillStyle = '#fff';
    g.fillRect(0, 0, 40, 8);
    const t = canvasTexture(c, true);
    t.repeat.set(16, 1);
    return t;
  });
}

const cageBarMat = () => sharedMat('fx-cage-bar', () => new THREE.MeshBasicMaterial({ color: COLOR.blocked, transparent: true, opacity: 0.9, depthWrite: false }));
const cageFaceMat = () => sharedMat('fx-cage-face', () => new THREE.MeshBasicMaterial({
  color: COLOR.blocked, transparent: true, opacity: 0.1, depthWrite: false, side: THREE.DoubleSide,
}));
const cageRingMat = () => sharedMat('fx-cage-ring', () => new THREE.MeshBasicMaterial({
  color: COLOR.blocked, map: dashTexture(), transparent: true, opacity: 0.95, depthWrite: false, side: THREE.DoubleSide,
}));

const _inv = new THREE.Matrix4();
const _rel = new THREE.Matrix4();
const _bb = new THREE.Box3();

/** Bounding box of root's meshes in root's local space, skipping decorations and effects. */
function localBounds(root, out) {
  out.makeEmpty();
  root.updateWorldMatrix(true, true);
  _inv.copy(root.matrixWorld).invert();
  const walk = (o) => {
    if (o.userData.noBounds || o.userData.fx) return;
    if (o.isMesh && o.geometry) {
      if (!o.geometry.boundingBox) o.geometry.computeBoundingBox();
      _bb.copy(o.geometry.boundingBox).applyMatrix4(_rel.multiplyMatrices(_inv, o.matrixWorld));
      out.union(_bb);
    }
    for (let i = 0; i < o.children.length; i++) walk(o.children[i]);
  };
  walk(root);
  return out;
}

class StatusFx {
  constructor(kit) {
    this.kit = kit;
    this.status = 'ok';
    this.box = localBounds(kit.fxParent, new THREE.Box3());
    if (this.box.isEmpty()) this.box.set(new THREE.Vector3(-0.1, 0, -0.1), new THREE.Vector3(0.1, 0.2, 0.1));
    this.size = this.box.getSize(new THREE.Vector3());
    this.center = this.box.getCenter(new THREE.Vector3());
    this.halo = null;
    this.cage = null;
    this.swapped = [];
  }

  set(status) {
    if (status !== 'alert' && status !== 'blocked' && status !== 'offline') status = 'ok';
    if (status === this.status) return;
    this.status = status;
    if (status === 'alert' && !this.halo) this.buildHalo();
    if (status === 'blocked' && !this.cage) this.buildCage();
    if (this.halo) this.halo.root.visible = status === 'alert';
    if (this.cage) this.cage.root.visible = status === 'blocked';
    this.setGrey(status === 'offline');
  }

  /** Offline: shared materials swap to grey twins, per-instance glows dim. */
  setGrey(on) {
    const kit = this.kit;
    kit.dim = on ? 0.22 : 1;
    if (on && this.swapped.length === 0) {
      kit.body.traverse(o => {
        if (!o.isMesh || o.userData.fx || !o.material?.userData?.shared) return;
        o.userData.baseMaterial = o.material;
        o.material = greyed(o.material);
        this.swapped.push(o);
      });
    } else if (!on) {
      for (const o of this.swapped) o.material = o.userData.baseMaterial;
      this.swapped.length = 0;
    }
    for (const m of kit.mats) {
      if (m.userData.dynamic) continue;
      m.emissiveIntensity = m.userData.baseEI * kit.dim;
      m.color.setHex(m.userData.baseColor);
      if (on) m.color.lerp(GREY, 0.35);
    }
  }

  /** Put a unit band around the model: at its base (halo) or middle (cage). */
  placeRing(mesh, pad, atBase, minR = 0.06) {
    const { box, size, center } = this;
    const mount = this.kit.mount;
    if (mount === 'wall') {
      const r = Math.max(minR, 0.5 * Math.hypot(size.x, size.y) * pad);
      mesh.position.set(center.x, center.y, box.max.z + 0.012);
      mesh.rotation.x = Math.PI / 2;
      mesh.userData.r = r;
    } else {
      const r = Math.max(minR, 0.5 * Math.hypot(size.x, size.z) * pad);
      const y = !atBase ? center.y : mount === 'ceiling' ? box.min.y - 0.01 : box.min.y + 0.006;
      mesh.position.set(center.x, y, center.z);
      mesh.userData.r = r;
    }
    mesh.scale.setScalar(mesh.userData.r);
  }

  buildHalo() {
    const { kit, size, center, box } = this;
    const root = new THREE.Group();
    root.userData.fx = true;
    const maxDim = Math.max(size.x, size.y, size.z);
    const spriteMat = kit.own(new THREE.SpriteMaterial({
      map: glowTexture(), color: COLOR.alert, transparent: true, opacity: 0.5, depthWrite: false, blending: THREE.AdditiveBlending,
    }));
    const sprite = new THREE.Sprite(spriteMat);
    sprite.userData.fx = true;
    sprite.position.copy(center);
    if (kit.mount === 'wall') sprite.position.z = box.max.z + 0.03;
    const s = clamp(maxDim * 1.8, 0.45, 6);   // tiny devices still get a halo visible from the street
    sprite.scale.set(s, s, 1);
    // A steady amber ring that breathes, plus a "ping" ring that expands and fades.
    const ringParams = { color: COLOR.alert, transparent: true, opacity: 0.8, depthWrite: false, side: THREE.DoubleSide };
    const ringMat = kit.own(new THREE.MeshBasicMaterial(ringParams));
    const pingMat = kit.own(new THREE.MeshBasicMaterial(ringParams));
    const ring = new THREE.Mesh(bandGeometry(), ringMat);
    const ping = new THREE.Mesh(bandGeometry(), pingMat);
    const pad = kit.mount === 'wall' ? 1.08 : 1.25;
    this.placeRing(ring, pad, true, 0.16);
    this.placeRing(ping, pad, true, 0.16);
    for (const m of [sprite, ring, ping]) { m.userData.fx = true; root.add(m); }
    kit.fxParent.add(root);
    this.halo = { root, spriteMat, ringMat, ping, pingMat };
  }

  buildCage() {
    const { kit, box, size } = this;
    const maxDim = Math.max(size.x, size.y, size.z);
    const pad = clamp(maxDim * 0.08, 0.012, 0.15);
    const wall = kit.mount === 'wall', ceiling = kit.mount === 'ceiling';
    let x0 = box.min.x - pad, x1 = box.max.x + pad;
    let y0 = box.min.y - (ceiling ? pad : pad * 0.4), y1 = box.max.y + (ceiling ? 0 : pad);
    let z0 = wall ? box.min.z : box.min.z - pad, z1 = box.max.z + pad;
    // Small devices get a cage of at least ~22 cm so it reads from the street.
    const MIN = 0.22;
    if (x1 - x0 < MIN) { const c = (x0 + x1) / 2; x0 = c - MIN / 2; x1 = c + MIN / 2; }
    if (!wall && z1 - z0 < MIN) { const c = (z0 + z1) / 2; z0 = c - MIN / 2; z1 = c + MIN / 2; }
    if (y1 - y0 < MIN * 0.8) {
      if (ceiling) y0 = y1 - MIN * 0.8;
      else if (wall) { const c = (y0 + y1) / 2; y0 = c - MIN * 0.4; y1 = c + MIN * 0.4; }
      else y1 = y0 + MIN * 0.8;
    }
    const W = x1 - x0, H = y1 - y0, D = z1 - z0, my = (y0 + y1) / 2;
    const cageDim = Math.max(W, H, D);
    const t = clamp(cageDim * 0.012, 0.003, 0.022);
    const spacing = clamp(cageDim / 6, 0.03, 0.3);
    const parts = [];
    const bar = (x, z) => parts.push(new THREE.BoxGeometry(t, H, t).translate(x, my, z));
    const nx = Math.max(2, Math.round(W / spacing)), nz = Math.max(2, Math.round(D / spacing));
    for (let i = 0; i <= nx; i++) { bar(x0 + (W * i) / nx, z0); bar(x0 + (W * i) / nx, z1); }
    for (let i = 1; i < nz; i++) { bar(x0, z0 + (D * i) / nz); bar(x1, z0 + (D * i) / nz); }
    for (const y of [y0, y1]) {
      parts.push(new THREE.BoxGeometry(W + t, t, t).translate((x0 + x1) / 2, y, z0));
      parts.push(new THREE.BoxGeometry(W + t, t, t).translate((x0 + x1) / 2, y, z1));
      parts.push(new THREE.BoxGeometry(t, t, D).translate(x0, y, (z0 + z1) / 2));
      parts.push(new THREE.BoxGeometry(t, t, D).translate(x1, y, (z0 + z1) / 2));
    }
    const barsGeo = kit.own(mergeGeometries(parts));
    for (const p of parts) p.dispose();
    const root = new THREE.Group();
    root.userData.fx = true;
    const bars = new THREE.Mesh(barsGeo, cageBarMat());
    const faces = new THREE.Mesh(box1(), cageFaceMat());
    faces.position.set((x0 + x1) / 2, my, (z0 + z1) / 2);
    faces.scale.set(W, H, D);
    const band = new THREE.Mesh(bandGeometry(), cageRingMat());
    this.placeRing(band, 1.1, false, 0.5 * (wall ? Math.hypot(W, H) : Math.hypot(W, D)) * 1.04);
    if (!wall) band.position.set((x0 + x1) / 2, my, (z0 + z1) / 2);
    for (const m of [bars, faces, band]) { m.userData.fx = true; root.add(m); }
    kit.fxParent.add(root);
    this.cage = { root, band };
  }

  /** Per-frame animation in real seconds; only runs while the status is not 'ok'. */
  update(now) {
    if (this.status === 'alert' && this.halo) {
      const h = this.halo, p = (now * 0.8) % 1, beat = 0.5 + 0.5 * Math.sin(now * 5.5);
      h.ping.scale.setScalar(h.ping.userData.r * (1 + 0.6 * p));
      h.pingMat.opacity = 0.75 * (1 - p);
      h.ringMat.opacity = 0.5 + 0.4 * beat;
      h.spriteMat.opacity = 0.3 + 0.35 * beat;
    } else if (this.status === 'blocked' && this.cage) {
      const band = this.cage.band, spin = now * 0.6;
      if (this.kit.mount === 'wall') band.rotation.set(Math.PI / 2, spin, 0);
      else band.rotation.y = spin;
    }
  }
}

const box1 = () => box(1, 1, 1);

// ---------------------------------------------------------------------------
// Device builders. Each fills kit.body, sets kit.port and returns
// step(device, dtSec, tSec, realDtSec, nowMs).
// ---------------------------------------------------------------------------

const FEET = (x, z) => [[-x, -z], [x, -z], [-x, z], [x, z]];

/** Packets seen since the last call: the device's own counters plus the sensor's total. */
function trafficProbe() {
  let own = -1, all = -1;
  return (d) => {
    const st = d.stats;
    const o = st ? st.tx + st.rx : 0;
    const a = globalThis.__shield?.state?.ids?.metrics?.inspected ?? 0;
    const busy = own >= 0 && (o !== own || a !== all);
    own = o;
    all = a;
    return busy;
  };
}

// ---- infrastructure ------------------------------------------------------------

function buildRouter(kit) {
  const b = kit.body;
  const shell = metal(0x1c1f26, 0.45, 0.2);
  for (const [x, z] of FEET(0.115, 0.07)) kit.add(b, cyl(0.009, 0.01, 0.008, 10), matte(0x0b0c0f, 0.9), x, 0.004, z);
  kit.add(b, rbox(0.27, 0.042, 0.176, 0.012), shell, 0, 0.029, 0);
  kit.add(b, flatPlane(0.23, 0.125), texMat('router-vents', () => ({ map: ventTexture(), roughness: 0.6, metalness: 0.2 })), 0, 0.0503, -0.014);
  // Four paddle antennas, hinged at the back and splayed.
  const antenna = matte(0x14161b, 0.5);
  const AX = [-0.105, -0.035, 0.035, 0.105], SPLAY = [0.22, 0.07, -0.07, -0.22];
  for (let i = 0; i < 4; i++) {
    const p = kit.pivot(b, AX[i], 0.034, -0.094);
    p.rotation.set(-0.12, 0, SPLAY[i]);
    kit.add(p, cyl(0.0075, 0.0075, 0.018, 10), antenna, 0, 0, 0, 0, 0, Math.PI / 2);
    kit.add(p, rbox(0.017, 0.165, 0.008, 0.004), antenna, 0, 0.088, 0);
  }
  // Status LEDs on the front: power, internet, 2.4 GHz, 5 GHz, LAN.
  const LED_HEX = [0x7dffb0, 0x7dffb0, COLOR.allowed, COLOR.allowed, 0x7dffb0];
  const leds = LED_HEX.map((hex, i) => {
    const m = kit.mat({ color: 0x0d0f12, emissive: hex, emissiveIntensity: 0, roughness: 0.4 }, true);
    kit.add(b, rbox(0.014, 0.005, 0.004, 0.0015, 1), m, -0.06 + i * 0.03, 0.03, 0.0885);
    return m;
  });
  // A light strip along the top front edge, readable from above.
  const strip = kit.mat({ color: 0x0d0f12, emissive: COLOR.allowed, emissiveIntensity: 0, roughness: 0.4 }, true);
  kit.add(b, flatPlane(0.2, 0.005), strip, 0, 0.0504, 0.074);
  kit.port.set(0, 0.13, -0.03);

  const probe = trafficProbe();
  let activity = 0, powerHex = 0;
  return (d, dt, t, rdt) => {
    if (probe(d)) activity = 1;
    activity *= Math.exp(-rdt * 1.2);
    const up = d.status !== 'offline' && d.online !== false;
    const hex = up ? 0x7dffb0 : COLOR.ledAmber;
    if (hex !== powerHex) { powerHex = hex; leds[0].emissive.setHex(hex); }
    // While rebooting the power LED blinks amber at full strength (not dimmed).
    leds[0].emissiveIntensity = up ? 1.6 * kit.dim : ((t * 2.5) % 1 < 0.5 ? 2 : 0.1);
    const tick = Math.floor(t * 14);
    for (let i = 1; i < 5; i++) {
      const lit = !up ? 0 : activity < 0.05 ? 1 : hash1(tick * 5 + i) > 0.38 ? 1 : 0.12;
      kit.glow(leds[i], 1.6 * lit);
    }
    kit.glow(strip, up ? 0.35 + 1.1 * activity * (0.6 + 0.4 * hash1(tick)) : 0);
  };
}

function buildShield(kit) {
  const b = kit.body;
  const shell = metal(0x23272f, 0.32, 0.35);
  for (const [x, z] of FEET(0.1, 0.065)) kit.add(b, cyl(0.009, 0.01, 0.008, 10), matte(0x0b0c0f, 0.9), x, 0.004, z);
  kit.add(b, rbox(0.25, 0.068, 0.17, 0.018, 3), shell, 0, 0.042, 0);
  const label = shieldLabelTextures();
  kit.add(b, flatPlane(0.21, 0.13), texMat('shield-top', () => ({
    map: label.map, emissive: 0xffffff, emissiveMap: label.glow, emissiveIntensity: 1.1, roughness: 0.45, metalness: 0.2,
  })), 0, 0.0763, -0.006);
  // Extruded emblem on the front: saffron shield, dark field, saffron core.
  const outer = sharedGeo('shield-emblem-outer', () => new THREE.ExtrudeGeometry(shieldShape(0.036, 0.044), { depth: 0.003, bevelEnabled: false, curveSegments: 6 }));
  const inner = sharedGeo('shield-emblem-inner', () => new THREE.ExtrudeGeometry(shieldShape(0.025, 0.031), { depth: 0.002, bevelEnabled: false, curveSegments: 6 }));
  kit.add(b, outer, led(COLOR.saffron, 1.5), -0.083, 0.043, 0.085);
  kit.add(b, inner, matte(0x1a1e26, 0.4), -0.083, 0.0435, 0.087);
  kit.add(b, cylZ(0.0045, 0.0045, 0.002, 14), led(COLOR.saffron, 1.5), -0.083, 0.0445, 0.0895);
  // Status light bar along the front and the top front edge; a bright spot scans along it.
  const scan = kit.own(new THREE.CanvasTexture(scanImage()));
  scan.wrapS = THREE.RepeatWrapping;
  scan.colorSpace = THREE.SRGBColorSpace;
  const bar = kit.mat({ color: 0x101216, emissive: COLOR.saffron, emissiveMap: scan, emissiveIntensity: 2.2, roughness: 0.3 }, true);
  kit.add(b, plane(0.15, 0.007), bar, 0.035, 0.024, 0.0852);
  kit.add(b, flatPlane(0.2, 0.006), bar, 0, 0.0765, 0.075);
  const vent = matte(0x111317, 0.8);
  for (let i = 0; i < 4; i++) {
    kit.add(b, box(0.002, 0.005, 0.1), vent, -0.1252, 0.026 + i * 0.011, 0);
    kit.add(b, box(0.002, 0.005, 0.1), vent, 0.1252, 0.026 + i * 0.011, 0);
  }
  kit.port.set(0, 0.11, 0);

  let scanPos = 0, barHex = COLOR.saffron;
  return (d, dt, t, rdt) => {
    const metrics = globalThis.__shield?.state?.ids?.metrics;
    const threat = metrics?.threat || 'low';
    const hex = threat === 'critical' ? COLOR.blocked : threat === 'high' ? 0xff7a3d : threat === 'elevated' ? COLOR.alert : COLOR.saffron;
    if (hex !== barHex) { barHex = hex; bar.emissive.setHex(hex); }
    const pps = metrics?.pps ?? 20;
    scanPos = (scanPos + rdt * (0.35 + Math.min(pps, 200) / 200)) % 1;
    scan.offset.x = -scanPos;
    kit.glow(bar, 2.2);
  };
}

function buildHub(kit) {
  const b = kit.body;
  kit.add(b, rbox(0.094, 0.02, 0.068, 0.006), matte(0xc51a4a, 0.45), 0, 0.01, 0);
  kit.add(b, rbox(0.092, 0.013, 0.066, 0.005), matte(0xf2f0ec, 0.4), 0, 0.0255, 0);
  kit.add(b, flatPlane(0.026, 0.026), texMat('pi-logo', () => ({ map: raspberryTexture(), alphaTest: 0.4, roughness: 0.5 })), 0.022, 0.0322, 0);
  // USB and Ethernet on the end, a Zigbee stick in one USB port, small ports on the side.
  const silver = metal(0xc3c8cf, 0.3, 0.6);
  kit.add(b, box(0.006, 0.013, 0.014), silver, 0.0455, 0.012, -0.022);
  kit.add(b, box(0.006, 0.013, 0.014), silver, 0.0455, 0.012, -0.004);
  kit.add(b, box(0.006, 0.013, 0.016), silver, 0.0455, 0.012, 0.018);
  kit.add(b, rbox(0.034, 0.008, 0.016, 0.003, 1), matte(0x2a6fdb, 0.5), 0.063, 0.012, -0.004);
  const dark = matte(0x111214, 0.6);
  for (const x of [-0.026, -0.01, 0.006]) kit.add(b, box(0.009, 0.005, 0.003), dark, x, 0.011, -0.0345);
  // Light pipes: red power (steady), green activity (flickers with traffic).
  kit.add(b, box(0.004, 0.003, 0.002), led(COLOR.ledRed, 1.4), -0.03, 0.016, 0.0345);
  const act = kit.mat({ color: 0x0d0f12, emissive: COLOR.ledGreen, emissiveIntensity: 0, roughness: 0.4 }, true);
  kit.add(b, box(0.004, 0.003, 0.002), act, -0.022, 0.016, 0.0345);
  const actGlow = kit.glowSprite(b, COLOR.ledGreen, 0.05, 0.6, -0.022, 0.017, 0.038);
  kit.port.set(0, 0.06, 0);

  const probe = trafficProbe();
  let activity = 0;
  return (d, dt, t, rdt) => {
    if (probe(d)) activity = 1;
    activity *= Math.exp(-rdt * 1.5);
    const up = d.status !== 'offline';
    const lit = up ? (activity > 0.05 ? (hash1(Math.floor(t * 16)) > 0.4 ? 1 : 0.1) : 0.35) : 0;
    kit.glow(act, 2 * lit);
    actGlow.material.opacity = 0.6 * lit * kit.dim;
  };
}

// ---- living room --------------------------------------------------------------------

function buildTv(kit) {
  kit.mount = 'wall';
  const b = kit.body;
  const W = 1.21, H = 0.68;   // 55-inch 16:9 panel
  kit.add(b, box(0.42, 0.3, 0.03), matte(0x15171b, 0.7), 0, 0, -0.025);
  kit.add(b, rbox(W + 0.024, H + 0.024, 0.024, 0.006), metal(0x111216, 0.35, 0.3), 0, 0, 0);
  const scr = kit.screen(512, 288, 1.15);
  kit.add(b, plane(W, H), scr.mat, 0, 0, 0.0122);
  kit.add(b, box(0.05, 0.006, 0.003), metal(0x9aa0aa, 0.3, 0.6), 0, -H / 2 - 0.006, 0.0115);
  const standby = kit.mat({ color: 0x110808, emissive: 0xff2a2a, emissiveIntensity: 0 }, true);
  kit.add(b, box(0.006, 0.003, 0.003), standby, W / 2 - 0.03, -H / 2 - 0.006, 0.0115);
  kit.port.set(0, H / 2 + 0.05, 0.02);

  const channels = ['Cricket Live', 'News', 'Movies', 'Cartoons', 'SHIELD Dashboard'];
  const series = new Float32Array(48);
  let head = 0, seeded = false, level = 0, lastDraw = -1e9, lastT = -1, channel = null, bannerUntil = 0;
  return (d, dt, t, rdt, now) => {
    const on = d.props.power !== false;
    level += ((on ? 1 : 0) - level) * damp(5, rdt);
    kit.glow(scr.mat, 1.15 * level);
    kit.glow(standby, on ? 0 : 1.2);
    if (!on && level < 0.01) return;
    const ch = d.props.channel;
    const switched = ch !== channel;
    if (switched) {
      if (channel !== null) bannerUntil = now + 1600;
      channel = ch;
    }
    if (!switched && (now - lastDraw < SCREEN_MIN_MS || (t === lastT && now > bannerUntil + SCREEN_MIN_MS))) return;
    lastDraw = now;
    lastT = t;
    const g = scr.ctx;
    if (ch === 'SHIELD Dashboard') {
      const pps = globalThis.__shield?.state?.ids?.metrics?.pps;
      const v = Number.isFinite(pps) ? pps : 26 + 9 * Math.sin(t * 0.7) + 5 * hash1(Math.floor(t * 4));
      if (!seeded) { series.fill(v); seeded = true; }   // start from a level line, not from zero
      series[head] = v;
      head = (head + 1) % series.length;
      paintDashboard(g, scr.w, scr.h, t, series, head);
    } else {
      (CHANNEL_PAINTERS[ch] || paintCricket)(g, scr.w, scr.h, t);
    }
    if (now < bannerUntil) paintChannelBanner(g, scr.w, String(ch), Math.max(0, channels.indexOf(ch)));
    scr.tex.needsUpdate = true;
  };
}

function buildSpeaker(kit) {
  const b = kit.body;
  kit.add(b, cyl(0.048, 0.046, 0.006, 32), matte(0x16181d, 0.8), 0, 0.003, 0);
  kit.add(b, cyl(0.05, 0.05, 0.13, 32, true), texMat('speaker-fabric', () => ({ map: fabricTexture(), roughness: 0.95 })), 0, 0.071, 0);
  kit.add(b, cyl(0.049, 0.05, 0.014, 32), matte(0x1d2027, 0.5), 0, 0.143, 0);
  kit.add(b, flatCircle(0.043, 32), matte(0x25282f, 0.6), 0, 0.1502, 0);
  const button = matte(0x3a3e47, 0.5);
  for (let i = 0; i < 4; i++) {
    const a = Math.PI / 4 + (i * Math.PI) / 2;
    kit.add(b, cyl(0.0055, 0.0055, 0.002, 12), button, Math.cos(a) * 0.024, 0.151, Math.sin(a) * 0.024);
  }
  const ringMat = kit.mat({ color: 0x0b0d12, emissive: 0x1a4dff, emissiveIntensity: 0, roughness: 0.3 }, true);
  kit.add(b, flatRing(0.0435, 0.0495, 48), ringMat, 0, 0.1506, 0);
  const spinMat = kit.mat({ color: 0x0b0d12, emissive: 0x52e0ff, emissiveIntensity: 0, roughness: 0.3 }, true);
  const spinner = kit.add(b, flatRing(0.043, 0.05, 16, 0, 1.3), spinMat, 0, 0.151, 0);
  const halo = kit.glowSprite(b, 0x3b8bff, 0.32, 0, 0, 0.16, 0);
  kit.port.set(0, 0.18, 0);

  let lis = 0, mus = 0, ringHex = 0x1a4dff;
  return (d, dt, t, rdt) => {
    const on = d.props.power !== false;
    const listening = on && !!d.props.listening;
    const music = on && !!d.props.music && !listening;
    lis += ((listening ? 1 : 0) - lis) * damp(8, rdt);
    mus += ((music ? 1 : 0) - mus) * damp(4, rdt);
    const hex = listening || lis > mus ? 0x1a4dff : 0x00e5ff;
    if (hex !== ringHex) {
      ringHex = hex;
      ringMat.emissive.setHex(hex);
      halo.material.color.setHex(hex === 0x1a4dff ? 0x3b8bff : 0x22e5ff);
    }
    spinner.rotation.y = (spinner.rotation.y - dt * 7) % TAU;
    spinner.visible = lis > 0.02;
    const pulse = 0.55 + 0.45 * Math.sin(t * 4);
    const ringI = lis * 1.4 + mus * (0.5 + 1.1 * pulse);
    kit.glow(ringMat, ringI);
    kit.glow(spinMat, lis * 3.4);
    halo.material.opacity = Math.min(1, ringI) * 0.55 * kit.dim;
  };
}

const GLASS_ON = new THREE.Color(0xf4f1ea);
const GLASS_OFF = new THREE.Color(0x8e9096);

function buildLight(kit) {
  kit.mount = 'ceiling';
  const b = kit.body;
  const brass = metal(0xb08850, 0.35, 0.6);
  kit.add(b, cyl(0.055, 0.06, 0.016, 24), matte(0xeeeae2, 0.5), 0, 0.072, 0);
  kit.add(b, cyl(0.0035, 0.0035, 0.2, 6), matte(0x1a1a1a, 0.6), 0, -0.036, 0);
  kit.add(b, cyl(0.03, 0.04, 0.035, 20), brass, 0, -0.145, 0);
  // Opal glass bell turned on a lathe. It glows in the bulb's colour, so the
  // light reads from above (the usual camera angle) as well as from below.
  const shadeGeo = sharedGeo('pendant-glass', () => {
    // Listed from the rim up so the lathe's faces point outwards.
    const pts = [[0.155, -0.17], [0.157, -0.158], [0.15, -0.12], [0.13, -0.08], [0.1, -0.045], [0.068, -0.02], [0.036, 0]]
      .map(([x, y]) => new THREE.Vector2(x, y));
    return new THREE.LatheGeometry(pts, 32);
  });
  const glass = kit.mat({ color: 0xf4f1ea, emissive: 0xffd8a8, emissiveIntensity: 0, roughness: 0.35, side: THREE.DoubleSide }, true);
  kit.add(b, shadeGeo, glass, 0, -0.16, 0);
  kit.add(b, flatTorus(0.156, 0.004, 6, 40), brass, 0, -0.33, 0);
  const bulb = kit.mat({ color: 0xfff6e8, emissive: 0xffd8a8, emissiveIntensity: 0, roughness: 0.3 }, true);
  kit.add(b, sphere(0.045, 20, 14), bulb, 0, -0.27, 0);
  const halo = kit.glowSprite(b, 0xffd8a8, 0.85, 0, 0, -0.26, 0);
  kit.port.set(0, -0.11, 0);

  let level = 0, color = null;
  return (d, dt, t, rdt) => {
    const on = d.props.power !== false;
    const target = on ? clamp(Number(d.props.brightness ?? 100) / 100, 0, 1) : 0;
    level += (target - level) * damp(6, rdt);
    if (d.props.color !== color) {
      color = d.props.color;
      if (typeof color === 'string' && /^#[0-9a-f]{6}$/i.test(color)) {
        bulb.emissive.set(color);
        glass.emissive.set(color);
        halo.material.color.set(color);
      }
    }
    glass.color.copy(GLASS_OFF).lerp(GLASS_ON, Math.min(1, level * 2));
    kit.glow(bulb, level * 3.2);
    kit.glow(glass, level * 1.1);
    halo.material.opacity = level * 0.85 * kit.dim;
  };
}

function buildAc(kit) {
  kit.mount = 'wall';
  const b = kit.body;
  const W = 0.9, H = 0.29, D = 0.2, zc = -0.1 + D / 2, zf = zc + D / 2;
  const shell = matte(0xf1f2f4, 0.35);
  kit.add(b, rbox(W, H, D, 0.035, 3), shell, 0, 0, zc).castShadow = true;
  kit.add(b, rbox(W - 0.04, H - 0.11, 0.008, 0.004, 1), matte(0xf8f9fa, 0.25), 0, 0.035, zf + 0.001);
  kit.add(b, box(W - 0.1, 0.05, 0.03), matte(0x2b3038, 0.8), 0, -H / 2 + 0.035, zf - 0.012);
  // Louvre hinged along its top edge; swings while the AC runs.
  const louvre = kit.pivot(b, 0, -H / 2 + 0.06, zf + 0.006);
  kit.add(louvre, rbox(W - 0.12, 0.052, 0.006, 0.002, 1), shell, 0, -0.026, 0);
  kit.add(b, rbox(0.085, 0.034, 0.004, 0.003, 1), matte(0x0e1015, 0.3), 0.33, 0, zf + 0.007);
  const scr = kit.screen(128, 48, 1.25);
  kit.add(b, plane(0.075, 0.028), scr.mat, 0.33, 0, zf + 0.0092);
  const ledMat = kit.mat({ color: 0x0d0f12, emissive: COLOR.ledBlue, emissiveIntensity: 0 }, true);
  kit.add(b, cylZ(0.003, 0.003, 0.002, 10), ledMat, 0.39, -0.03, zf + 0.006);
  kit.add(b, box(0.08, 0.006, 0.002), matte(0xb8bdc6, 0.4), -0.36, 0.09, zf + 0.0062);
  // Airflow sheet: soft bands drifting down and out of the outlet.
  const breezeTex = kit.own(new THREE.CanvasTexture(breezeImage()));
  breezeTex.wrapS = breezeTex.wrapT = THREE.RepeatWrapping;
  const breezeMat = kit.own(new THREE.MeshBasicMaterial({
    map: breezeTex, color: 0x9fd8ff, transparent: true, opacity: 0, depthWrite: false, side: THREE.DoubleSide, blending: THREE.AdditiveBlending,
  }));
  const breeze = new THREE.Mesh(plane(W - 0.16, 0.55), breezeMat);
  breeze.position.set(0, -0.37, 0.27);
  breeze.rotation.x = -0.66;
  kit.decor(breeze, b);
  kit.port.set(0.3, H / 2 + 0.03, 0);

  const MODE_HEX = { Cool: 0x9fd8ff, Dry: 0x8ff0d8, Fan: 0xe8eef7 };
  let lv = 0, angle = 0, shownOn = null, shownTemp = null, shownMode = null, mode = null;
  return (d, dt, t, rdt) => {
    const on = d.props.power !== false;
    lv += ((on ? 1 : 0) - lv) * damp(2.5, rdt);
    const swing = on ? 0.55 + 0.32 * Math.sin(t * 0.9) : 0;
    angle += (swing - angle) * damp(3, rdt);
    louvre.rotation.x = -angle;
    if (on !== shownOn || d.props.temp !== shownTemp || d.props.mode !== shownMode) {
      shownOn = on;
      shownTemp = d.props.temp;
      shownMode = d.props.mode;
      paintAc(scr.ctx, scr.w, scr.h, on, Number(d.props.temp) || 24, d.props.mode || 'Cool');
      scr.tex.needsUpdate = true;
    }
    if (d.props.mode !== mode) { mode = d.props.mode; breezeMat.color.setHex(MODE_HEX[mode] ?? 0x9fd8ff); }
    kit.glow(scr.mat, 1.25 * lv);
    kit.glow(ledMat, on ? 1.4 : 0);
    breezeMat.opacity = 0.2 * lv * kit.dim;
    breezeTex.offset.y = (t * 0.5) % 1;
  };
}

function buildClock(kit) {
  kit.mount = 'wall';
  const b = kit.body;
  kit.add(b, cylZ(0.18, 0.18, 0.04, 48), matte(0x1d2026, 0.45), 0, 0, 0);
  kit.add(b, torus(0.168, 0.0045, 8, 48), metal(0xc9a46a, 0.3, 0.7), 0, 0, 0.0205);
  kit.add(b, circle(0.166, 48), texMat('clock-dial', () => ({ map: dialTexture(), roughness: 0.55 })), 0, 0, 0.0202);
  kit.add(b, rbox(0.104, 0.03, 0.003, 0.004, 1), matte(0x0d1014, 0.3), 0, -0.05, 0.0215);
  const scr = kit.screen(192, 48, 1.3);
  kit.add(b, plane(0.096, 0.024), scr.mat, 0, -0.05, 0.0232);
  const hand = (key, w, len, tail, depth) =>
    sharedGeo(key, () => new THREE.BoxGeometry(w, len + tail, depth).translate(0, (len - tail) / 2, 0));
  const dark = matte(0x14161a, 0.5);
  const hourHand = kit.add(b, hand('hand-h', 0.011, 0.085, 0.018, 0.003), dark, 0, 0, 0.0245);
  const minuteHand = kit.add(b, hand('hand-m', 0.007, 0.13, 0.022, 0.0025), dark, 0, 0, 0.0275);
  const secondHand = kit.add(b, hand('hand-s', 0.0025, 0.142, 0.03, 0.002), matte(0xd63a3a, 0.5), 0, 0, 0.0302);
  kit.add(b, cylZ(0.007, 0.007, 0.004, 16), metal(0xc9a46a, 0.3, 0.7), 0, 0, 0.031);
  kit.port.set(0, 0.2, 0.02);

  let lastKey = -1;
  return (d, dt, t) => {
    const s = simDaySeconds(t);
    secondHand.rotation.z = -((Math.floor(s) % 60) / 60) * TAU;
    minuteHand.rotation.z = -(((s / 60) % 60) / 60) * TAU;
    hourHand.rotation.z = -(((s / 3600) % 12) / 12) * TAU;
    const twelve = d.props.format === '12h';
    const key = Math.floor(s / 60) * 2 + (twelve ? 1 : 0);
    if (key !== lastKey) {
      lastKey = key;
      paintClockLine(scr.ctx, scr.w, scr.h, formatTime(s, twelve));
      scr.tex.needsUpdate = true;
    }
    kit.glow(scr.mat, 1.3);
  };
}

function buildPhone(kit) {
  const b = kit.body;
  const slab = kit.pivot(b);   // vibrates while ringing
  kit.add(slab, rbox(0.075, 0.0085, 0.158, 0.0038), metal(0x1e2330, 0.35, 0.4), 0, 0.00425, 0);
  const scr = kit.screen(150, 316, 1.15);
  kit.add(slab, flatPlane(0.069, 0.151), scr.mat, 0, 0.0086, 0);
  const pulseMat = kit.own(new THREE.MeshBasicMaterial({
    color: COLOR.allowed, transparent: true, opacity: 0, depthWrite: false, side: THREE.DoubleSide, blending: THREE.AdditiveBlending,
  }));
  const pulse = kit.decor(new THREE.Mesh(bandGeometry(), pulseMat), b);
  pulse.position.y = 0.004;
  kit.port.set(0, 0.05, 0);

  let ringing = null, wifi = null, lastDraw = -1e9, minute = -1;
  return (d, dt, t, rdt, now) => {
    const r = !!d.props.ringing, w = d.props.wifi !== false;
    const s = simDaySeconds(t), m = Math.floor(s / 60);
    if (r !== ringing || w !== wifi || m !== minute || now - lastDraw > 2000) {
      ringing = r;
      wifi = w;
      minute = m;
      lastDraw = now;
      paintPhone(scr.ctx, scr.w, scr.h, { ringing: r, wifi: w, time: formatTime(s, false), tiles: phoneTiles() });
      scr.tex.needsUpdate = true;
    }
    if (r) {
      slab.rotation.y = Math.sin(now * 0.006) > 0 ? 0.05 * Math.sin(now * 0.09) : 0;
      const p = (now * 0.0012) % 1;
      pulse.scale.setScalar(0.1 + 0.2 * p);
      pulseMat.opacity = 0.8 * (1 - p) * kit.dim;
    } else {
      slab.rotation.y = 0;
      pulseMat.opacity = 0;
    }
    kit.glow(scr.mat, 1.15);
  };
}

const VACUUM_SPEED = 0.32;   // m/s when travelling between the dock and its cleaning path

function buildVacuum(kit, device) {
  const b = kit.body;
  const floorY = -clamp(device.pos?.[1] ?? 0, 0, 0.2);   // the catalog lifts the origin slightly off the floor
  const y0 = floorY + 0.008;
  const mover = kit.pivot(b);
  kit.fxParent = mover;
  kit.add(mover, cyl(0.17, 0.168, 0.07, 40), matte(0xe9ecf0, 0.4), 0, y0 + 0.035, 0);
  kit.add(mover, cyl(0.163, 0.167, 0.008, 40), metal(0xcfd4db, 0.3, 0.25), 0, y0 + 0.074, 0);
  kit.add(mover, cyl(0.174, 0.174, 0.045, 40, true, -Math.PI / 2, Math.PI), matte(0x2a2e35, 0.6), 0, y0 + 0.03, 0);
  const lidar = kit.pivot(mover, 0, y0 + 0.078, -0.045);
  kit.add(lidar, cyl(0.042, 0.044, 0.024, 24), matte(0x1a1d23, 0.4), 0, 0.012, 0);
  kit.add(lidar, box(0.03, 0.009, 0.012), matte(0x07080a, 0.15), 0, 0.012, 0.037);
  kit.add(lidar, flatCircle(0.03, 20), metal(0x4a505c, 0.3, 0.4), 0, 0.0245, 0);
  const ledMat = kit.mat({ color: 0x0d0f12, emissive: COLOR.ledGreen, emissiveIntensity: 0 }, true);
  kit.add(mover, rbox(0.04, 0.003, 0.008, 0.0015, 1), ledMat, 0, y0 + 0.0785, 0.088);
  kit.add(mover, cyl(0.012, 0.012, 0.003, 16), metal(0x5a606b, 0.3, 0.5), 0, y0 + 0.079, 0.06);
  const brushes = [-1, 1].map(side => {
    const p = kit.pivot(mover, side * 0.115, y0 + 0.001, 0.12);
    for (let i = 0; i < 3; i++) {
      const arm = kit.pivot(p);
      arm.rotation.y = (i * TAU) / 3;
      kit.add(arm, box(0.06, 0.002, 0.004), matte(0x30343c, 0.6), 0.03, 0, 0);
    }
    return p;
  });
  // Charging dock (stays at the origin).
  kit.add(b, rbox(0.25, 0.01, 0.12, 0.004, 1), matte(0x1d2026, 0.6), 0, floorY + 0.005, -0.23);
  kit.add(b, rbox(0.22, 0.11, 0.07, 0.015), matte(0x1d2026, 0.5), 0, floorY + 0.055, -0.275);
  const dockLed = kit.mat({ color: 0x0d0f12, emissive: COLOR.ledWhite, emissiveIntensity: 0 }, true);
  kit.add(b, box(0.03, 0.004, 0.002), dockLed, 0, floorY + 0.09, -0.2395);

  // Lissajous cleaning path in group-local metres. It passes through the dock
  // position (the origin) at s = 0, so leaving and returning are seamless.
  const room = HOUSE.rooms.find(r => r.id === (device.room || 'living'));
  let ax0 = VACUUM_AREA.x[0], ax1 = VACUUM_AREA.x[1], az0 = VACUUM_AREA.z[0], az1 = VACUUM_AREA.z[1];
  if (room) {
    ax0 = Math.max(ax0, room.x[0] + 0.4); ax1 = Math.min(ax1, room.x[1] - 0.4);
    az0 = Math.max(az0, room.z[0] + 0.4); az1 = Math.min(az1, room.z[1] - 0.4);
  }
  const rot = device.rotY || 0, cr = Math.cos(rot), sr = Math.sin(rot);
  const wx = (ax0 + ax1) / 2 - (device.pos?.[0] ?? 0), wz = (az0 + az1) / 2 - (device.pos?.[2] ?? 0);
  const cx = wx * cr - wz * sr, cz = wx * sr + wz * cr;
  const ax = Math.max(0.3, (ax1 - ax0) / 2), az = Math.max(0.3, (az1 - az0) / 2);
  const FX = 0.13, FZ = 0.21;
  const phx = Math.asin(clamp(-cx / ax, -1, 1)), phz = Math.asin(clamp(-cz / az, -1, 1));
  const pathX = s => cx + ax * Math.sin(FX * s + phx);
  const pathZ = s => cz + az * Math.sin(FZ * s + phz);
  kit.port.set(0, floorY + 0.12, 0);

  let s = 0, onPath = false, x = 0, z = 0, heading = 0, ledHex = COLOR.ledGreen;
  return (d, dt, t, rdt) => {
    const running = !!d.props.running;
    let vx = 0, vz = 0;
    if (running && onPath) {
      s += dt;
      const nx = pathX(s), nz = pathZ(s);
      vx = nx - x; vz = nz - z;
      x = nx; z = nz;
    } else {
      // Head for the path (resume where it left off) or back to the dock.
      if (!running) onPath = false;
      const tx = running ? pathX(s) : 0, tz = running ? pathZ(s) : 0;
      const ddx = tx - x, ddz = tz - z, dist = Math.hypot(ddx, ddz), stepLen = VACUUM_SPEED * dt;
      if (dist <= stepLen) {
        x = tx; z = tz;
        if (running) onPath = true; else s = 0;
      } else {
        vx = (ddx / dist) * stepLen; vz = (ddz / dist) * stepLen;
        x += vx; z += vz;
      }
    }
    const moving = vx * vx + vz * vz > 1e-10;
    const targetHeading = moving ? Math.atan2(vx, vz) : (!running && x === 0 && z === 0 ? 0 : heading);
    heading = wrapAngle(heading + wrapAngle(targetHeading - heading) * damp(moving ? 6 : 2.5, dt));
    mover.position.set(x, 0, z);
    mover.rotation.y = heading;
    kit.port.set(x, floorY + 0.12, z);
    lidar.rotation.y = (lidar.rotation.y + (running ? 9 : 0) * dt) % TAU;
    if (moving) {
      brushes[0].rotation.y = (brushes[0].rotation.y + 16 * dt) % TAU;
      brushes[1].rotation.y = (brushes[1].rotation.y - 16 * dt) % TAU;
    }
    const battery = Number(d.props.battery ?? 100);
    const hex = battery > 50 ? COLOR.ledGreen : battery > 20 ? COLOR.ledAmber : COLOR.ledRed;
    if (hex !== ledHex) { ledHex = hex; ledMat.emissive.setHex(hex); }
    const docked = !running && x === 0 && z === 0;
    const charging = docked && battery < 100;
    kit.glow(ledMat, charging ? 0.5 + 0.9 * (0.5 + 0.5 * Math.sin(t * 3)) : 1.4);
    kit.glow(dockLed, docked ? (charging ? 0.6 + 0.8 * (0.5 + 0.5 * Math.sin(t * 3)) : 1.2) : 0.25);
  };
}

// ---- kitchen ------------------------------------------------------------------------

function buildFridge(kit) {
  const b = kit.body;
  const W = 0.8, H = 1.8, D = 0.66, zc = -0.15, zf = zc + D / 2, t = 0.025;
  const steel = metal(0xc8cdd4, 0.32, 0.45), liner = matte(0xf2f4f6, 0.55), dark = matte(0x24272d, 0.7);
  const shell = [
    kit.add(b, box(t, H - 0.08, D), steel, -W / 2 + t / 2, 0.08 + (H - 0.08) / 2, zc),
    kit.add(b, box(t, H - 0.08, D), steel, W / 2 - t / 2, 0.08 + (H - 0.08) / 2, zc),
    kit.add(b, box(W, t, D), steel, 0, H - t / 2, zc),
  ];
  for (const m of shell) m.castShadow = true;
  kit.add(b, box(W, 0.08, D - 0.03), dark, 0, 0.04, zc - 0.015);
  kit.add(b, box(W - 0.08, 0.035, 0.004), matte(0x0d0e10, 0.8), 0, 0.045, zf - 0.027);
  kit.add(b, box(W - 2 * t, H - 0.08 - t, 0.02), liner, 0, 0.08 + (H - 0.08 - t) / 2, zc - D / 2 + 0.01);
  kit.add(b, box(0.03, H - 0.08 - t, D - 0.02), liner, 0, 0.08 + (H - 0.08 - t) / 2, zc - 0.01);
  kit.add(b, box(0.004, H - 0.12, D - 0.04), liner, W / 2 - t - 0.002, 0.08 + (H - 0.12) / 2, zc - 0.01);
  kit.add(b, box(W / 2 - t - 0.015, 0.012, D - 0.04), liner, (W / 2 - t + 0.015) / 2, 0.086, zc - 0.01);
  // Shelves and food in the fresh-food compartment (seen when the door opens).
  const shelfX = (W / 2 - t + 0.015) / 2, shelfW = W / 2 - t - 0.02;
  const glassShelf = matte(0xcfe3ee, 0.15);
  for (const y of [0.52, 0.92, 1.32]) kit.add(b, box(shelfW, 0.008, D - 0.14), glassShelf, shelfX, y, zc - 0.04);
  kit.add(b, rbox(0.065, 0.18, 0.065, 0.006), matte(0xfafafa, 0.6), 0.1, 0.92 + 0.094, zc - 0.07);
  kit.add(b, box(0.066, 0.03, 0.066), matte(0x2f6fdb, 0.5), 0.1, 0.92 + 0.199, zc - 0.07);
  kit.add(b, cyl(0.032, 0.032, 0.2, 16), matte(0xff9a2e, 0.35), 0.29, 0.52 + 0.104, zc - 0.05);
  kit.add(b, cyl(0.012, 0.012, 0.03, 10), matte(0x2b8a3e, 0.5), 0.29, 0.52 + 0.219, zc - 0.05);
  for (const [x, zz] of [[0.14, 0.02], [0.2, -0.03], [0.26, 0.03], [0.32, -0.02]]) kit.add(b, sphere(0.032, 12, 10), matte(0xd62f3a, 0.45), x, 1.32 + 0.036, zc + zz);
  kit.add(b, rbox(0.16, 0.08, 0.12, 0.01), matte(0xf2c14e, 0.5), 0.2, 0.09 + 0.046, zc + 0.02);
  const lightMat = kit.mat({ color: 0xf6f8ff, emissive: 0xeaf4ff, emissiveIntensity: 0 }, true);
  kit.add(b, box(0.22, 0.006, 0.05), lightMat, shelfX, H - t - 0.004, zc + 0.08);
  // Left door (closed) with display, dispenser and handle.
  const doorH = H - 0.1, doorY = 0.09 + doorH / 2;
  kit.add(b, rbox(W / 2 - 0.006, doorH, 0.05, 0.012), steel, -W / 4 - 0.0015, doorY, zf + 0.025).castShadow = true;
  const chrome = metal(0xe3e6ea, 0.2, 0.7);
  kit.add(b, cyl(0.011, 0.011, 0.72, 12), chrome, -0.03, 1.0, zf + 0.085);
  for (const y of [0.68, 1.32]) kit.add(b, box(0.016, 0.02, 0.04), chrome, -0.03, y, zf + 0.064);
  kit.add(b, rbox(0.13, 0.09, 0.004, 0.006, 1), matte(0x0a0c10, 0.2), -W / 4, 1.42, zf + 0.0515);
  const scr = kit.screen(128, 88, 1.25);
  kit.add(b, plane(0.12, 0.082), scr.mat, -W / 4, 1.42, zf + 0.0538);
  kit.add(b, rbox(0.15, 0.22, 0.008, 0.01, 1), matte(0x1a1c21, 0.5), -W / 4, 1.08, zf + 0.052);
  kit.add(b, box(0.03, 0.006, 0.003), led(COLOR.ledBlue, 1.2), -W / 4, 1.17, zf + 0.0575);
  // Right door on its hinge, with door bins on the inside.
  const hinge = kit.pivot(b, W / 2, doorY, zf);
  kit.add(hinge, rbox(W / 2 - 0.006, doorH, 0.05, 0.012), steel, -W / 4 + 0.0015, 0, 0.025).castShadow = true;
  kit.add(hinge, cyl(0.011, 0.011, 0.72, 12), chrome, -W / 2 + 0.03, 1.0 - doorY, 0.085);
  for (const y of [0.68, 1.32]) kit.add(hinge, box(0.016, 0.02, 0.04), chrome, -W / 2 + 0.03, y - doorY, 0.064);
  const bin = matte(0xe8eef3, 0.4);
  for (const y of [-0.45, 0.05, 0.5]) kit.add(hinge, box(0.32, 0.06, 0.045), bin, -W / 4, y, -0.024);
  kit.port.set(0, H + 0.05, zc);

  let angle = 0, shownTemp = null, shownOpen = null;
  return (d, dt, t, rdt) => {
    const open = !!d.props.doorOpen;
    angle += ((open ? 1.75 : 0) - angle) * damp(open ? 3 : 4, rdt);
    hinge.rotation.y = angle;
    kit.glow(lightMat, angle > 0.05 ? 1.6 : 0);
    if (d.props.temp !== shownTemp || open !== shownOpen) {
      shownTemp = d.props.temp;
      shownOpen = open;
      paintFridge(scr.ctx, scr.w, scr.h, d.props.temp ?? 4, open);
      scr.tex.needsUpdate = true;
    }
    kit.glow(scr.mat, 1.25);
  };
}

function buildPlug(kit) {
  const b = kit.body;
  const WALL = -0.365;   // the kitchen backsplash, behind the counter
  const white = matte(0xf4f3ef, 0.45);
  kit.add(b, rbox(0.082, 0.082, 0.008, 0.006, 1), white, 0, 0.13, WALL + 0.004);
  kit.add(b, rbox(0.058, 0.058, 0.04, 0.01), white, 0, 0.13, WALL + 0.028);
  const plugLed = kit.mat({ color: 0x0d0f12, emissive: COLOR.ledGreen, emissiveIntensity: 0 }, true);
  kit.add(b, ring(0.011, 0.0145, 28), plugLed, 0, 0.13, WALL + 0.0485);
  kit.add(b, cylZ(0.009, 0.009, 0.003, 20), matte(0xe4e2dc, 0.5), 0, 0.13, WALL + 0.0485);
  const plugGlow = kit.glowSprite(b, COLOR.ledGreen, 0.09, 0, 0, 0.13, WALL + 0.052);

  // Espresso machine on the counter, wired to the plug.
  const mx = -0.535, mz = -0.15, front = mz + 0.18, gx = mx - 0.03;
  const steel = metal(0xb5bac2, 0.3, 0.55), black = matte(0x17191d, 0.5);
  kit.add(b, rbox(0.42, 0.39, 0.36, 0.02), steel, mx, 0.195, mz).castShadow = true;
  kit.add(b, box(0.38, 0.006, 0.3), black, mx, 0.393, mz);
  for (const x of [-0.08, 0.02]) {
    kit.add(b, cyl(0.024, 0.03, 0.045, 16), matte(0xf5f3ee, 0.4), mx + x, 0.4185, mz + 0.03);
  }
  kit.add(b, box(0.25, 0.16, 0.004), black, gx, 0.13, front + 0.002);
  kit.add(b, rbox(0.09, 0.05, 0.06, 0.008), steel, gx, 0.235, front + 0.025);
  kit.add(b, cyl(0.036, 0.036, 0.028, 20), metal(0x8e949d, 0.3, 0.6), gx, 0.198, front + 0.03);
  kit.add(b, cyl(0.032, 0.028, 0.02, 20), metal(0x9aa0a8, 0.3, 0.6), gx, 0.176, front + 0.03);
  kit.add(b, rbox(0.022, 0.018, 0.12, 0.008), black, gx, 0.174, front + 0.1);
  kit.add(b, rbox(0.27, 0.025, 0.09, 0.006, 1), steel, gx, 0.0125, front + 0.04);
  kit.add(b, box(0.25, 0.002, 0.07), black, gx, 0.026, front + 0.04);
  const cupGeo = sharedGeo('espresso-cup', () => new THREE.LatheGeometry(
    [[0, 0], [0.022, 0], [0.026, 0.004], [0.03, 0.046], [0.027, 0.046], [0.024, 0.008], [0, 0.008]].map(([x, y]) => new THREE.Vector2(x, y)), 24));
  kit.add(b, cupGeo, matte(0xf7f5f0, 0.35), gx, 0.027, front + 0.035);
  const coffee = kit.add(b, flatCircle(0.026, 20), matte(0x5a3218, 0.25), gx, 0.036, front + 0.035);
  coffee.visible = false;
  // Pressure gauge, buttons, power LED and steam wand.
  kit.add(b, cylZ(0.03, 0.03, 0.01, 24), steel, mx + 0.13, 0.3, front + 0.005);
  kit.add(b, circle(0.025, 24), matte(0xfaf8f2, 0.4), mx + 0.13, 0.3, front + 0.0102);
  const needle = kit.add(b, sharedGeo('gauge-needle', () => new THREE.BoxGeometry(0.0022, 0.02, 0.001).translate(0, 0.009, 0)),
    matte(0xd63a3a, 0.5), mx + 0.13, 0.3, front + 0.011);
  for (let i = 0; i < 3; i++) kit.add(b, cylZ(0.009, 0.009, 0.006, 16), black, mx - 0.15 + i * 0.035, 0.31, front + 0.003);
  const powerLed = kit.mat({ color: 0x110808, emissive: COLOR.ledRed, emissiveIntensity: 0 }, true);
  kit.add(b, circle(0.004, 12), powerLed, mx + 0.13, 0.245, front + 0.0012);
  kit.add(b, cyl(0.005, 0.005, 0.13, 8), steel, mx + 0.175, 0.17, front + 0.04, 0, 0, 0.15);
  kit.add(b, cylZ(0.012, 0.012, 0.02, 12), black, mx + 0.165, 0.245, front + 0.01);
  // Power cord along the counter to the smart plug.
  const cord = sharedGeo('plug-cord', () => new THREE.TubeGeometry(new THREE.CatmullRomCurve3([
    new THREE.Vector3(mx + 0.2, 0.04, mz - 0.14), new THREE.Vector3(-0.22, 0.006, -0.33), new THREE.Vector3(-0.06, 0.006, -0.335),
    new THREE.Vector3(-0.012, 0.05, -0.335), new THREE.Vector3(0, 0.1, -0.33),
  ]), 24, 0.004, 6, false));
  kit.add(b, cord, matte(0xeceae4, 0.6));
  // Steam over the cup.
  const steamMats = [];
  const steam = [0, 1, 2].map(i => {
    const m = kit.own(new THREE.SpriteMaterial({ map: glowTexture(), color: 0xffffff, transparent: true, opacity: 0, depthWrite: false }));
    steamMats.push(m);
    const sp = new THREE.Sprite(m);
    sp.position.set(gx, 0.09, front + 0.035);
    sp.scale.set(0.06, 0.06, 1);
    sp.userData.phase = i / 3;
    return kit.decor(sp, b);
  });
  kit.port.set(0, 0.2, -0.33);

  let level = 0, wasOn = false, steamLv = 0, gauge = -0.9;
  return (d, dt, t, rdt) => {
    const on = !!d.props.power;
    const brewing = on && Number(d.props.watts || 0) > 500;
    if (on && !wasOn) level = 0;
    wasOn = on;
    if (brewing) level = Math.min(1, level + dt / 40);
    coffee.visible = level > 0.02;
    coffee.position.y = 0.036 + level * 0.032;
    coffee.scale.setScalar(0.86 + 0.14 * level);
    kit.glow(plugLed, on ? 1.6 : 0);
    plugGlow.material.opacity = on ? 0.55 * kit.dim : 0;
    kit.glow(powerLed, on ? 1.8 : 0);
    gauge += ((brewing ? 0.9 + 0.08 * Math.sin(t * 7) : on ? 0.25 : -0.9) - gauge) * damp(3, rdt);
    needle.rotation.z = -gauge;
    steamLv += ((on ? (brewing ? 1 : 0.4) : 0) - steamLv) * damp(1.5, rdt);
    for (let i = 0; i < 3; i++) {
      const p = (t * 0.35 + steam[i].userData.phase) % 1;
      steam[i].position.y = 0.08 + p * 0.2;
      steam[i].position.x = gx + 0.012 * Math.sin(t * 2 + i * 2);
      steam[i].scale.setScalar(0.04 + p * 0.08);
      steamMats[i].opacity = Math.sin(p * Math.PI) * 0.22 * steamLv * kit.dim;
    }
  };
}

// ---- bedroom ------------------------------------------------------------------------

function buildFan(kit) {
  kit.mount = 'ceiling';
  const b = kit.body;
  const bronze = metal(0x4a3b2e, 0.4, 0.45);
  kit.add(b, cyl(0.06, 0.075, 0.03, 24), bronze, 0, 0.005, 0);
  kit.add(b, cyl(0.011, 0.011, 0.22, 10), bronze, 0, -0.115, 0);
  kit.add(b, cyl(0.1, 0.115, 0.075, 28), bronze, 0, -0.26, 0);
  kit.add(b, cyl(0.06, 0.045, 0.035, 20), matte(0xf4f1ea, 0.4), 0, -0.333, 0);
  const rotor = kit.pivot(b, 0, -0.3, 0);
  kit.add(rotor, cyl(0.105, 0.105, 0.02, 28), bronze, 0, 0, 0);
  const iron = metal(0x2a2a2e, 0.4, 0.5), wood = matte(0x8b5a36, 0.55);
  for (let i = 0; i < 3; i++) {
    const arm = kit.pivot(rotor);
    arm.rotation.y = (i * TAU) / 3;
    kit.add(arm, box(0.13, 0.008, 0.035), iron, 0.15, -0.004, 0);
    const blade = kit.add(arm, rbox(0.52, 0.008, 0.13, 0.004, 1), wood, 0.44, -0.008, 0);
    blade.rotation.x = 0.12;
    blade.castShadow = true;
  }
  // Motion-blur disc that fades in at high speed (the blades alias otherwise).
  const blurMat = kit.own(new THREE.MeshBasicMaterial({ color: 0x6b4a30, transparent: true, opacity: 0, depthWrite: false, side: THREE.DoubleSide }));
  const blur = new THREE.Mesh(flatRing(0.12, 0.7, 48), blurMat);
  blur.position.y = -0.308;
  kit.decor(blur, b);
  kit.port.set(0, -0.15, 0);

  let omega = 0;
  return (d, dt) => {
    const speed = clamp(Number(d.props.speed) || 0, 0, 5);
    const target = d.props.power !== false ? (0.55 + 0.6 * speed) * TAU : 0;
    omega += (target - omega) * damp(0.8, dt);
    rotor.rotation.y = (rotor.rotation.y + Math.min(omega * dt, 0.5)) % TAU;
    blurMat.opacity = clamp((omega - 9) / 14, 0, 1) * 0.28 * kit.dim;
  };
}

function buildWatch(kit) {
  const b = kit.body;
  const strap = matte(0x23262d, 0.8), steel = metal(0x9aa1ab, 0.25, 0.7);
  kit.add(b, rbox(0.022, 0.0035, 0.072, 0.0015, 1), strap, 0, 0.00175, 0.055);
  kit.add(b, rbox(0.022, 0.0035, 0.072, 0.0015, 1), strap, 0, 0.00175, -0.055);
  kit.add(b, box(0.025, 0.004, 0.006), steel, 0, 0.002, 0.088);
  kit.add(b, cyl(0.0215, 0.0225, 0.011, 32), metal(0x2c3038, 0.3, 0.5), 0, 0.0065, 0);
  kit.add(b, flatTorus(0.0205, 0.0013, 6, 32), steel, 0, 0.012, 0);
  kit.add(b, cyl(0.0026, 0.0026, 0.004, 10), steel, 0.0235, 0.0068, 0, 0, 0, Math.PI / 2);
  const scr = kit.screen(128, 128, 1.25);
  kit.add(b, flatCircle(0.0195, 32), scr.mat, 0, 0.0122, 0);
  kit.port.set(0, 0.04, 0);

  let shownHr = -1, shownBig = null, shownSteps = null, shownMinute = -1;
  return (d, dt, t) => {
    const hr = Math.round(Number(d.props.heartRate) || 0);
    const big = ((t * hr) / 60) % 1 < 0.18;
    const s = simDaySeconds(t), minute = Math.floor(s / 60);
    if (hr !== shownHr || big !== shownBig || d.props.steps !== shownSteps || minute !== shownMinute) {
      shownHr = hr;
      shownBig = big;
      shownSteps = d.props.steps;
      shownMinute = minute;
      paintWatch(scr.ctx, scr.w, hr, Number(d.props.steps) || 0, big, formatTime(s, false));
      scr.tex.needsUpdate = true;
    }
    kit.glow(scr.mat, 1.25);
  };
}

// ---- hallway ------------------------------------------------------------------------

function buildLock(kit) {
  kit.mount = 'wall';
  const b = kit.body;
  const satin = metal(0x2a2d34, 0.35, 0.45), nickel = metal(0xd9dde2, 0.2, 0.8);
  // Strike plate on the frame, bolt rail and the sliding bolt (towards -x, the door edge).
  kit.add(b, rbox(0.03, 0.06, 0.004, 0.002, 1), metal(0xb7bcc4, 0.3, 0.6), -0.125, -0.022, 0.002);
  kit.add(b, box(0.075, 0.024, 0.012), satin, -0.065, -0.022, 0.006);
  const bolt = kit.add(b, rbox(0.05, 0.016, 0.01, 0.003, 1), nickel, -0.07, -0.022, 0.0065);
  kit.add(b, rbox(0.068, 0.155, 0.036, 0.012), satin, 0, 0, 0.018);
  kit.add(b, rbox(0.058, 0.142, 0.004, 0.01), metal(0x3a3f48, 0.3, 0.4), 0, 0, 0.0365);
  const turn = kit.pivot(b, 0, -0.022, 0.0385);
  kit.add(turn, cylZ(0.016, 0.017, 0.006, 24), metal(0x9aa0aa, 0.3, 0.6), 0, 0, 0.003);
  kit.add(turn, rbox(0.012, 0.044, 0.016, 0.004, 1), nickel, 0, 0, 0.012);
  const ledMat = kit.mat({ color: 0x0d0f12, emissive: COLOR.ledGreen, emissiveIntensity: 0 }, true);
  kit.add(b, cylZ(0.0045, 0.0045, 0.002, 16), ledMat, 0, 0.052, 0.0395);
  const ledGlow = kit.glowSprite(b, COLOR.ledGreen, 0.07, 0.6, 0, 0.052, 0.043);
  kit.port.set(0, 0.11, 0.02);

  let k = -1, hex = 0;
  return (d, dt, t, rdt) => {
    const locked = !!d.props.locked;
    k = k < 0 ? (locked ? 1 : 0) : k + ((locked ? 1 : 0) - k) * damp(7, rdt);
    bolt.position.x = -0.07 - 0.045 * k;
    turn.rotation.z = (Math.PI / 2) * k;
    const h = locked ? COLOR.ledGreen : COLOR.ledRed;
    if (h !== hex) { hex = h; ledMat.emissive.setHex(h); ledGlow.material.color.setHex(h); }
    const blink = locked ? 1 : (t % 1 < 0.6 ? 1 : 0.25);
    kit.glow(ledMat, 1.8 * blink);
    ledGlow.material.opacity = 0.6 * blink * kit.dim;
  };
}

function buildThermostat(kit) {
  kit.mount = 'wall';
  const b = kit.body;
  kit.add(b, cylZ(0.046, 0.046, 0.004, 40), matte(0xf2f2f0, 0.5), 0, 0, 0.002);
  kit.add(b, cylZ(0.041, 0.041, 0.026, 40), metal(0xc9cdd3, 0.22, 0.75), 0, 0, 0.017);
  const scr = kit.screen(160, 160, 1.2);
  kit.add(b, circle(0.0365, 40), scr.mat, 0, 0, 0.0302);
  kit.port.set(0, 0.07, 0.02);

  let shownCurrent = null, shownTarget = null;
  return (d) => {
    const current = Number(d.props.current ?? 24), target = Number(d.props.target ?? 23.5);
    if (current !== shownCurrent || target !== shownTarget) {
      shownCurrent = current;
      shownTarget = target;
      paintThermostat(scr.ctx, scr.w, current, target);
      scr.tex.needsUpdate = true;
    }
    kit.glow(scr.mat, 1.2);
  };
}

function buildCamera(kit) {
  kit.mount = 'wall';
  const b = kit.body;
  const white = matte(0xf1f2f4, 0.35);
  kit.add(b, rbox(0.075, 0.075, 0.012, 0.006, 1), white, 0, 0, 0.006);
  kit.add(b, cylZ(0.011, 0.013, 0.05, 12), white, 0, 0, 0.037);
  kit.add(b, sphere(0.017, 14, 10), white, 0, 0, 0.065);
  const head = kit.pivot(b, 0, 0, 0.065);
  head.rotation.x = 0.32;   // tilted down towards the porch
  kit.add(head, cylZ(0.038, 0.038, 0.13, 28), white, 0, 0, 0.06);
  const hood = sharedGeo('camera-hood', () => new THREE.CylinderGeometry(0.044, 0.044, 0.15, 24, 1, true, Math.PI / 2, Math.PI).rotateX(Math.PI / 2));
  kit.add(head, hood, sharedMat('camera-hood', () => new THREE.MeshStandardMaterial({ color: 0xf1f2f4, roughness: 0.35, side: THREE.DoubleSide })), 0, 0, 0.068);
  kit.add(head, circle(0.034, 28), matte(0x0b0d11, 0.15), 0, 0, 0.1255);
  kit.add(head, cylZ(0.016, 0.018, 0.01, 20), matte(0x111318, 0.3), 0, -0.004, 0.128);
  kit.add(head, circle(0.011, 20), metal(0x2a4c7a, 0.08, 0.6), 0, -0.004, 0.1335);
  const recMat = kit.mat({ color: 0x140606, emissive: 0xff2a2a, emissiveIntensity: 0 }, true);
  kit.add(head, circle(0.0045, 12), recMat, 0.02, 0.017, 0.1258);
  const recGlow = kit.glowSprite(head, 0xff2a2a, 0.06, 0, 0.02, 0.017, 0.13);
  kit.port.set(0, 0.07, 0.05);

  return (d, dt, t) => {
    const rec = d.props.power !== false && !!d.props.recording;
    const on = rec && t % 1.2 < 0.8;
    kit.glow(recMat, rec ? (on ? 2.6 : 0.35) : 0);
    recGlow.material.opacity = on ? 0.75 * kit.dim : 0;
  };
}

// ---- utility + study ------------------------------------------------------------------

function buildWasher(kit) {
  const b = kit.body;
  const W = 0.6, H = 0.85, D = 0.58, zc = -0.21, zf = zc + D / 2;
  const shake = kit.pivot(b);   // vibrates while running
  const white = matte(0xf2f3f5, 0.35);
  for (const [x, z] of FEET(0.25, 0.22)) kit.add(b, cyl(0.02, 0.02, 0.012, 10), matte(0x2a2d33, 0.7), x, 0.006, zc + z);
  kit.add(shake, rbox(W, H, D, 0.02), white, 0, 0.012 + H / 2, zc).castShadow = true;
  kit.add(shake, rbox(W - 0.02, 0.11, 0.012, 0.004, 1), matte(0xe3e6ea, 0.3), 0, 0.79, zf + 0.004);
  kit.add(shake, rbox(0.17, 0.07, 0.01, 0.004, 1), white, -0.19, 0.79, zf + 0.012);
  kit.add(shake, box(0.06, 0.008, 0.003), matte(0x2a2d33, 0.6), -0.19, 0.768, zf + 0.0175);
  kit.add(shake, cylZ(0.032, 0.034, 0.022, 28), metal(0xc3c8cf, 0.25, 0.6), 0.02, 0.79, zf + 0.018);
  kit.add(shake, box(0.004, 0.018, 0.003), matte(0x2a2d33, 0.6), 0.02, 0.805, zf + 0.0295);
  kit.add(shake, rbox(0.13, 0.05, 0.004, 0.004, 1), matte(0x0a0c10, 0.25), 0.19, 0.79, zf + 0.0115);
  const scr = kit.screen(128, 48, 1.25);
  kit.add(shake, plane(0.12, 0.042), scr.mat, 0.19, 0.79, zf + 0.0137);
  // Porthole: chrome ring, drum with paddles and laundry, tinted glass.
  const doorY = 0.4;
  kit.add(shake, torus(0.165, 0.028, 12, 48), metal(0xd0d4da, 0.25, 0.45), 0, doorY, zf + 0.03);
  kit.add(shake, torus(0.142, 0.01, 8, 40), matte(0x2b2f36, 0.5), 0, doorY, zf + 0.022);
  const tube = sharedGeo('washer-tube', () => new THREE.CylinderGeometry(0.15, 0.15, 0.07, 32, 1, true).rotateX(Math.PI / 2));
  kit.add(shake, tube, sharedMat('washer-tube', () => new THREE.MeshStandardMaterial({ color: 0x9ea4ad, roughness: 0.4, metalness: 0.4, side: THREE.DoubleSide })), 0, doorY, zf - 0.01);
  const drum = kit.pivot(shake, 0, doorY, zf - 0.03);
  kit.add(drum, circle(0.15, 32), texMat('washer-drum', () => ({ map: drumTexture(), roughness: 0.4, metalness: 0.4 })), 0, 0, -0.012);
  for (let i = 0; i < 3; i++) {
    const a = (i * TAU) / 3;
    kit.add(drum, box(0.018, 0.05, 0.05), metal(0xb7bcc4, 0.3, 0.5), Math.sin(a) * 0.128, Math.cos(a) * 0.128, 0.012, 0, 0, -a);
  }
  const laundry = [[0x2f5d9e, -0.05, -0.07], [0xd94b4b, 0.04, -0.08], [0xf2c14e, 0.0, -0.03], [0xffffff, -0.08, -0.01], [0x3aa17e, 0.07, -0.03]];
  for (const [hex, x, y] of laundry) {
    const lump = kit.add(drum, sphere(0.045, 10, 8), matte(hex, 0.9), x, y, 0.01);
    lump.scale.set(1, 0.6, 0.55);
  }
  kit.add(shake, circle(0.15, 40), sharedMat('washer-glass', () => new THREE.MeshStandardMaterial({
    color: 0xa9cbe8, roughness: 0.05, metalness: 0.1, transparent: true, opacity: 0.22, depthWrite: false,
  })), 0, doorY, zf + 0.034);
  kit.add(shake, rbox(0.03, 0.09, 0.025, 0.008, 1), metal(0xd0d4da, 0.25, 0.45), 0.172, doorY, zf + 0.038);
  kit.port.set(0, H + 0.06, -0.2);

  let omega = 0, shownRunning = null, shownMin = null, shownProgram = null;
  return (d, dt, t) => {
    const running = !!d.props.running;
    const target = running ? (Math.sin(t * 0.45) > -0.3 ? 6.5 : -5) : 0;
    omega += (target - omega) * damp(1.5, dt);
    drum.rotation.z = (drum.rotation.z + clamp(omega * dt, -0.5, 0.5)) % TAU;
    shake.position.x = running ? 0.0012 * Math.sin(t * 61) : 0;
    if (running !== shownRunning || d.props.remainingMin !== shownMin || d.props.program !== shownProgram) {
      shownRunning = running;
      shownMin = d.props.remainingMin;
      shownProgram = d.props.program;
      paintWasher(scr.ctx, scr.w, scr.h, running, Number(d.props.remainingMin) || 0, d.props.program);
      scr.tex.needsUpdate = true;
    }
    kit.glow(scr.mat, 1.25);
  };
}

function buildMeter(kit) {
  kit.mount = 'wall';
  const b = kit.body;
  kit.add(b, rbox(0.26, 0.34, 0.12, 0.012), matte(0xb9bec5, 0.55), 0, 0, 0.01);
  kit.add(b, rbox(0.24, 0.32, 0.004, 0.008, 1), matte(0xc7cbd1, 0.5), 0, 0, 0.0715);
  kit.add(b, rbox(0.18, 0.1, 0.004, 0.006, 1), matte(0x0d1014, 0.25), 0, 0.075, 0.0745);
  const scr = kit.screen(256, 96, 1.25);
  kit.add(b, plane(0.165, 0.062), scr.mat, 0, 0.082, 0.0768);
  const pulseMat = kit.mat({ color: 0x140606, emissive: COLOR.ledRed, emissiveIntensity: 0 }, true);
  kit.add(b, circle(0.004, 12), pulseMat, 0.07, 0.04, 0.0768);
  kit.add(b, rbox(0.18, 0.05, 0.02, 0.004, 1), matte(0x2a2d33, 0.6), 0, -0.055, 0.08);
  for (let i = 0; i < 5; i++) kit.add(b, box(0.012, 0.022, 0.012), matte(0xf0f0ee, 0.5), -0.064 + i * 0.032, -0.05, 0.093);
  kit.add(b, plane(0.045, 0.04), texMat('hazard', () => ({ map: hazardTexture(), alphaTest: 0.5, roughness: 0.6 })), 0, -0.125, 0.0738);
  kit.add(b, cyl(0.012, 0.012, 0.3, 10), matte(0x8f949c, 0.6), 0.08, -0.32, 0);
  kit.add(b, box(0.01, 0.03, 0.008), matte(0x2a2d33, 0.6), 0.115, 0, 0.076);
  kit.port.set(0, 0.2, 0.02);

  let kw = null, phase = 0;
  return (d, dt) => {
    const v = Number(d.props.kw) || 0;
    if (v !== kw) {
      kw = v;
      paintMeter(scr.ctx, scr.w, scr.h, v);
      scr.tex.needsUpdate = true;
    }
    // Impulse LED: flashes faster as the load grows, like a real meter's imp/kWh light.
    phase = (phase + dt * v * 1.6) % 1;
    kit.glow(pulseMat, phase < 0.15 ? 2.2 : 0);
    kit.glow(scr.mat, 1.25);
  };
}

function buildLaptop(kit) {
  const b = kit.body;
  const alu = metal(0xa7adb7, 0.35, 0.55);
  kit.add(b, rbox(0.31, 0.015, 0.215, 0.006), alu, 0, 0.0075, 0);
  kit.add(b, flatPlane(0.27, 0.105), texMat('keyboard', () => ({ map: keyboardTexture(), roughness: 0.6 })), 0, 0.0152, -0.025);
  kit.add(b, flatPlane(0.1, 0.06), matte(0x9aa1ab, 0.3), 0, 0.0152, 0.068);
  const lid = kit.pivot(b, 0, 0.015, -0.104);
  lid.rotation.x = -0.28;
  kit.add(lid, rbox(0.31, 0.205, 0.006, 0.005), alu, 0, 0.1025, -0.003);
  kit.add(lid, plane(0.3, 0.196), matte(0x0b0c0f, 0.3), 0, 0.1025, 0.0002);
  const scr = kit.screen(512, 320, 1.15);
  kit.add(lid, plane(0.28, 0.175), scr.mat, 0, 0.106, 0.0006);
  kit.add(lid, plane(0.045, 0.056), texMat('sticker', () => ({ map: stickerTexture(), alphaTest: 0.5, roughness: 0.5 })), 0.07, 0.12, -0.0062, 0, Math.PI, 0);
  kit.port.set(0, 0.25, -0.12);

  let typed = -1, wifi = null, blink = null, lastDraw = -1e9;
  return (d, dt, t, rdt, now) => {
    const w = d.props.wifi !== false;
    const n = w ? terminalCursor(t) : typed;
    const bl = t % 1 < 0.55;
    if (w !== wifi || ((n !== typed || bl !== blink) && now - lastDraw >= 300)) {
      wifi = w;
      typed = Math.max(0, n);
      blink = bl;
      lastDraw = now;
      paintTerminal(scr.ctx, scr.w, scr.h, typed, w, bl);
      scr.tex.needsUpdate = true;
    }
    kit.glow(scr.mat, 1.15);
  };
}

// ---- outside the LAN -------------------------------------------------------------------

const CLOUD_PUFFS = [
  [0, 0.05, 0, 0.95], [-1.0, -0.15, 0.1, 0.72], [1.05, -0.12, 0.05, 0.75], [-0.45, 0.48, -0.1, 0.68],
  [0.5, 0.42, 0.05, 0.66], [-1.75, -0.35, 0, 0.45], [1.8, -0.32, 0, 0.48], [0.05, -0.35, 0.45, 0.62],
  [-0.6, -0.35, -0.4, 0.55], [0.7, -0.38, -0.35, 0.55],
];

function buildCloud(kit) {
  const b = kit.body;
  const float = kit.pivot(b);
  kit.fxParent = float;
  const geo = sharedGeo('cloud', () => {
    const parts = CLOUD_PUFFS.map(([x, y, z, r]) => new THREE.IcosahedronGeometry(r, 2).translate(x, y, z));
    const merged = mergeGeometries(parts);
    for (const p of parts) p.dispose();
    return merged;
  });
  const mat = kit.mat({ color: 0xe4ecff, emissive: 0x7f9fe0, emissiveIntensity: 0.35, roughness: 0.95, flatShading: true }, true);
  kit.add(float, geo, mat);
  kit.glowSprite(float, 0x7fa8ff, 6.2, 0.3, 0, 0, -0.7);
  // Three small service beacons orbiting the cloud: streaming, voice, time.
  const orbit = kit.decor(new THREE.Group(), float);
  orbit.rotation.x = 0.25;
  [COLOR.allowed, COLOR.saffron, 0xb48cff].forEach((hex, i) => {
    const beacon = new THREE.Mesh(sphere(0.08, 12, 8), kit.own(new THREE.MeshBasicMaterial({ color: hex })));
    beacon.position.set(Math.cos((i * TAU) / 3) * 2.4, 0, Math.sin((i * TAU) / 3) * 1.3);
    orbit.add(beacon);
  });
  kit.port.set(0, -0.85, 0.35);

  return (d, dt, t) => {
    const bob = 0.14 * Math.sin(t * 0.55);
    float.position.y = bob;
    float.rotation.y = 0.05 * Math.sin(t * 0.23);
    orbit.rotation.y = t * 0.35;
    kit.glow(mat, 0.33 + 0.07 * Math.sin(t * 0.8));
    kit.port.y = -0.85 + bob;
  };
}

function buildRogue(kit, device) {
  const b = kit.body;
  const floorY = -clamp(device.pos?.[1] ?? 0, 0, 0.2);   // origin sits a little above the floor
  // A bare Raspberry-Pi-sized board leaning against the wall behind the washer.
  const lean = kit.pivot(b, 0, floorY, 0);
  lean.rotation.x = -0.6;
  kit.add(lean, box(0.085, 0.056, 0.0016), matte(0x1f7a3a, 0.55), 0, 0.028, 0);
  const chip = matte(0x15171b, 0.4), silver = metal(0xc3c8cf, 0.3, 0.6);
  kit.add(lean, box(0.014, 0.014, 0.0016), chip, -0.008, 0.03, 0.0016);
  kit.add(lean, box(0.01, 0.008, 0.0012), chip, 0.012, 0.018, 0.0014);
  kit.add(lean, box(0.017, 0.013, 0.014), silver, 0.034, 0.012, 0.0078);
  kit.add(lean, box(0.017, 0.013, 0.014), silver, 0.034, 0.028, 0.0078);
  kit.add(lean, box(0.017, 0.014, 0.014), silver, 0.034, 0.045, 0.0078);
  kit.add(lean, box(0.05, 0.005, 0.0085), matte(0x111214, 0.5), -0.012, 0.051, 0.005);
  kit.add(lean, box(0.05, 0.0012, 0.0087), metal(0xd4b45a, 0.3, 0.8), -0.012, 0.0536, 0.005);
  const ledMat = kit.mat({ color: 0x140c02, emissive: COLOR.ledAmber, emissiveIntensity: 0 }, true);
  kit.add(lean, box(0.004, 0.003, 0.0015), ledMat, -0.035, 0.008, 0.0016);
  const ledGlow = kit.glowSprite(lean, COLOR.ledAmber, 0.07, 0, -0.035, 0.008, 0.004);
  // Power cable across the floor.
  const cable = sharedGeo('rogue-cable', () => new THREE.TubeGeometry(new THREE.CatmullRomCurve3([
    new THREE.Vector3(-0.042, 0.004, 0), new THREE.Vector3(-0.075, 0.003, 0.05), new THREE.Vector3(-0.06, 0.003, 0.11), new THREE.Vector3(-0.01, 0.003, 0.15),
  ]), 20, 0.0025, 6, false));
  kit.add(b, cable, matte(0x101114, 0.6), 0, floorY, 0);
  // "?" tag on a stiff wire: nobody registered this board.
  const corner = new THREE.Vector3(-0.038, 0.056, 0).applyEuler(lean.rotation).add(lean.position);
  const tag = kit.pivot(b, -0.05, floorY + 0.09, -0.035);
  const wireLen = tag.position.distanceTo(corner);
  const wire = kit.add(b, cyl(0.0007, 0.0007, wireLen, 4), metal(0xb7bcc4, 0.4, 0.6),
    (tag.position.x + corner.x) / 2, (tag.position.y + corner.y) / 2, (tag.position.z + corner.z) / 2);
  wire.lookAt(tag.position.x, tag.position.y, tag.position.z);
  wire.rotateX(Math.PI / 2);
  const tagMat = texMat('rogue-tag', () => ({ map: tagTexture(), alphaTest: 0.5, roughness: 0.8 }));
  kit.add(tag, plane(0.036, 0.045), tagMat, 0, 0.02, 0.0004);
  kit.add(tag, plane(0.036, 0.045), tagMat, 0, 0.02, -0.0004, 0, Math.PI, 0);
  kit.port.set(0, floorY + 0.06, -0.02);

  return (d, dt, t) => {
    // Busy double blink: the board is scanning the LAN.
    const p = (t * 1.4) % 1;
    const on = p < 0.12 || (p > 0.24 && p < 0.36);
    kit.glow(ledMat, on ? 2.6 : 0.12);
    ledGlow.material.opacity = on ? 0.85 * kit.dim : 0;
    tag.rotation.set(0, 0.35 + 0.12 * Math.sin(t * 1.1), 0.1 + 0.05 * Math.sin(t * 1.7));
  };
}

/** Anything not in the catalog still gets a tidy box with an LED instead of an error. */
function buildGeneric(kit) {
  const b = kit.body;
  kit.add(b, rbox(0.16, 0.1, 0.12, 0.015), matte(0x8892a6, 0.5), 0, 0.05, 0);
  kit.add(b, cylZ(0.006, 0.006, 0.002, 12), led(COLOR.ledGreen, 1.4), 0.05, 0.07, 0.061);
  kit.port.set(0, 0.13, 0);
  return null;
}

const BUILDERS = {
  router: buildRouter,
  shield: buildShield,
  hub: buildHub,
  tv: buildTv,
  speaker: buildSpeaker,
  light: buildLight,
  ac: buildAc,
  clock: buildClock,
  phone: buildPhone,
  vacuum: buildVacuum,
  fridge: buildFridge,
  plug: buildPlug,
  fan: buildFan,
  watch: buildWatch,
  lock: buildLock,
  thermostat: buildThermostat,
  camera: buildCamera,
  washer: buildWasher,
  meter: buildMeter,
  laptop: buildLaptop,
  cloud: buildCloud,
  rogue: buildRogue,
};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

const NO_STEP = () => {};

/**
 * Build the procedural model for one device (a catalog entry or its runtime
 * copy from state.devices). See the header comment for conventions.
 */
export function createDeviceModel(device) {
  const kit = new Kit(device);
  const build = BUILDERS[device.type] || buildGeneric;
  const step = build(kit, device) || NO_STEP;
  const fx = new StatusFx(kit);
  let last = -1;

  return {
    group: kit.group,
    port: kit.port,

    update(dev, dtSec, tSec) {
      const now = performance.now();
      const realDt = last < 0 ? 0 : Math.min((now - last) / 1000, 0.1);
      last = now;
      if (kit.pending.length) kit.attachPending();
      step(dev || device, dtSec > 0 ? dtSec : 0, tSec || 0, realDt, now);
      if (fx.status !== 'ok') fx.update(now / 1000);
    },

    setStatus(status) {
      fx.set(status);
    },

    dispose() {
      kit.group.removeFromParent();
      kit.dispose();
    },
  };
}
