// The three.js world: a night-time cut-away dollhouse of HOUSE (catalog.js),
// one procedural model per device (models.js), the Wi-Fi links that tie every
// device to the router, packets travelling src -> router -> SHIELD-IoT -> dst,
// the shield hologram, CSS2D labels, picking and the camera.
//
// Public API (CONTRACT.md): createScene({bus, state, container})
//   -> Promise<{ tick(realDtMs, simDtMs), focus(id) }>
//
// Performance notes: geometries and materials are shared, the static house is
// merged into a handful of meshes, packets and sparks are drawn by a single
// pooled THREE.Points object, and the per-frame path (tick) does not allocate.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { HOUSE } from './catalog.js';
import { createDeviceModel } from './models.js';

// ---------------------------------------------------------------------------
// Tunables
// ---------------------------------------------------------------------------

const HEX = {
  allowed: 0x5ad1e6,
  alerted: 0xffb347,
  dropped: 0xff4d5e,
  shield: 0xf2b134,
  threat: 0xff5a2a,
  drift: 0xb48cff,
  link: 0x7c9cc9,
  linkOffline: 0x5d6577,
  fibre: 0x9fd3ff,
  select: 0x8fe9ff,
  routerLed: 0x7dffb0,
};
const STATUS_TEXT = { ok: 'online', alert: 'alert', blocked: 'blocked', offline: 'offline' };

const MAX_PACKETS = 250;           // in flight at once (CONTRACT.md)
const MAX_PACKETS_REDUCED = 70;    // prefers-reduced-motion
const TRAIL = 3;                   // points per packet: head + two fading ghosts
const MAX_SPARKS = 320;            // burst particles
const MAX_SEGS = 5;                // bezier segments per packet path
const CLICK_SLOP_PX = 5;
const FOCUS_SEC = 0.8;
const SHADOW_EVERY_SEC = 0.05;     // shadow map refresh period (real time)
const INTRO_SEC = 2.6;
const HOME_PHI = 0.93;             // polar angle of the home view (from straight down)
const HOME_THETA = 0.6;            // azimuth of the home view (from +z towards +x)
const HOME_TARGET = new THREE.Vector3(0, 0.3, 1.0);
const TARGET_BOUNDS = { minX: -16, maxX: 16, minY: 0, maxY: 8.5, minZ: -17, maxZ: 14 };

const T_EXT = 0.16;                // exterior wall thickness
const T_INT = 0.1;                 // interior wall thickness
const H_INT = 2.0;                 // interior walls (lowered so the rooms stay visible)
const H_HALF = 1.1;                // half-height partitions around the router shelf
const H_LOW = 0.9;                 // cut-away front and right walls
const CAP = 0.025;                 // dark "section cut" strip on top of every wall

// Openings in the interior walls: {axis, at, from, to}. axis 'x' = a wall that
// runs along x at z = at; axis 'z' = a wall that runs along z at x = at.
const INTERIOR_OPENINGS = [
  { axis: 'z', at: 1, from: -3.9, to: -1.0 },   // living <-> kitchen, open plan
  { axis: 'x', at: 1, from: -2.45, to: -1.55 }, // living <-> bedroom
  { axis: 'z', at: -1, from: 1.55, to: 2.45 },  // hall <-> bedroom
  { axis: 'z', at: 1, from: 1.55, to: 2.45 },   // hall <-> utility
  { axis: 'x', at: 0, from: 2.6, to: 3.5 },     // kitchen <-> utility
  { axis: 'z', at: 5, from: 3.55, to: 4.45 },   // utility <-> study
];

// Interior wall lines that are only half height, so the router shelf (and the
// SHIELD-IoT sensor on it) stays visible from the street side.
const HALF_WALLS = [
  { axis: 'x', at: 1 },   // living | bedroom, living | hall
  { axis: 'z', at: 1 },   // living | kitchen, living | utility, hall | utility
];

// Where each room's floor label is painted (x, z, rotated to run along z).
const ROOM_LABEL_SPOTS = {
  living: [-7.2, 0.35, false],
  kitchen: [7.1, -0.9, false],
  bedroom: [-4.1, 5.45, false],
  hall: [-0.3, 3.4, true],
  utility: [3.3, 3.0, false],
  study: [7.0, 2.3, false],
};

// How a device is mounted decides where its selection ring goes and how the
// camera approaches it. Wall devices face their local +z (catalog rotY).
const WALL_TYPES = new Set(['tv', 'clock', 'ac', 'thermostat', 'camera', 'meter', 'lock']);
const CEILING_TYPES = new Set(['light', 'fan']);
// Floor appliances with a front (door, drum) that the camera should face when focusing.
const FRONT_TYPES = new Set(['washer', 'fridge']);

const TV_CHANNEL_HEX = {
  'Cricket Live': 0x7fd18b, News: 0x7fa8ff, Movies: 0xffb27a, Cartoons: 0xff8fd8, 'SHIELD Dashboard': 0xf2b134,
};

const LABEL_CSS = `
.sc-overlay{position:absolute;inset:0;pointer-events:none;overflow:hidden;z-index:1}
.sc-label{pointer-events:auto;cursor:pointer;user-select:none;-webkit-user-select:none}
.sc-label>div{display:flex;align-items:center;gap:6px;margin-bottom:8px;padding:3px 9px 3px 7px;border-radius:999px;
  background:rgba(9,13,24,.84);border:1px solid rgba(160,190,255,.2);color:#e9eef7;
  font:600 11px/1.25 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;letter-spacing:.01em;
  white-space:nowrap;box-shadow:0 4px 14px rgba(0,0,0,.4)}
.sc-label .sc-dot{width:7px;height:7px;border-radius:50%;background:#3ddc97;box-shadow:0 0 6px rgba(61,220,151,.8);flex:none}
.sc-label .sc-state{font-weight:500;opacity:.8;display:none}
.sc-label.is-selected>div{border-color:rgba(143,233,255,.85);box-shadow:0 0 0 1px rgba(143,233,255,.25),0 4px 14px rgba(0,0,0,.4)}
.sc-label.is-alert .sc-dot{background:#ffb347;box-shadow:0 0 8px rgba(255,179,71,.9);animation:sc-blink 1s steps(2,start) infinite}
.sc-label.is-blocked>div{border-color:rgba(255,77,94,.75)}
.sc-label.is-blocked .sc-dot{background:#ff4d5e;box-shadow:0 0 8px rgba(255,77,94,.9)}
.sc-label.is-offline .sc-dot{background:#7d8597;box-shadow:none}
.sc-label.is-offline>div{color:#b7bfcc}
.sc-label:not(.is-ok) .sc-state{display:inline}
@keyframes sc-blink{to{opacity:.35}}
@media (prefers-reduced-motion: reduce){.sc-label.is-alert .sc-dot{animation:none}}
`;

// ---------------------------------------------------------------------------
// Small helpers (module level, used at build time only)
// ---------------------------------------------------------------------------

/** Deterministic PRNG so the procedural textures look the same on every load. */
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

function makeCanvas(w, h) {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  return c;
}

function canvasTexture(canvas, aniso = 1, repeat = true) {
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  if (repeat) tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.anisotropy = aniso;
  return tex;
}

const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
const easeInOutCubic = t => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

/** sRGB components of a hex colour, for shaders that write raw output. */
function hexToRgb(hex, out) {
  out[0] = ((hex >> 16) & 255) / 255;
  out[1] = ((hex >> 8) & 255) / 255;
  out[2] = (hex & 255) / 255;
  return out;
}

/** Axis-aligned box geometry from min/max corners (for merging). */
function boxGeo(x0, y0, z0, x1, y1, z1) {
  const g = new THREE.BoxGeometry(Math.abs(x1 - x0), Math.abs(y1 - y0), Math.abs(z1 - z0));
  g.translate((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2);
  return g;
}

function cylGeo(x, z, r, y0, y1, segs = 12, rTop = r) {
  const g = new THREE.CylinderGeometry(rTop, r, y1 - y0, segs);
  g.translate(x, (y0 + y1) / 2, z);
  return g;
}

/** Horizontal floor plane whose UVs are world metres / tile size (so one texture fits any room). */
function floorGeo(x0, x1, z0, z1, tile, y = 0) {
  const g = new THREE.PlaneGeometry(x1 - x0, z1 - z0);
  g.rotateX(-Math.PI / 2);
  g.translate((x0 + x1) / 2, y, (z0 + z1) / 2);
  const pos = g.attributes.position;
  const uv = g.attributes.uv;
  for (let i = 0; i < pos.count; i++) uv.setXY(i, pos.getX(i) / tile, -pos.getZ(i) / tile);
  return g;
}

/**
 * Quadratic-bezier control point for a link between a and b: an arc that rises
 * with distance so Wi-Fi links clear the interior walls.
 */
function arcControl(a, b, out) {
  const len = a.distanceTo(b);
  const lift = len < 1 ? len * 0.4 : 0.25 + 0.16 * len;
  return out.set((a.x + b.x) / 2, Math.max(a.y, b.y) + lift, (a.z + b.z) / 2);
}

/** The ISP fibre drops in from the cloud, over the back wall, onto the router. */
function fibreControl(cloud, entry, out) {
  return out.set((cloud.x + entry.x) / 2, Math.max(cloud.y - 1, entry.y + 4.5), entry.z - 5.5);
}

function bezierAt(a, c, b, t, out) {
  const u = 1 - t;
  const w0 = u * u, w1 = 2 * u * t, w2 = t * t;
  return out.set(
    w0 * a.x + w1 * c.x + w2 * b.x,
    w0 * a.y + w1 * c.y + w2 * b.y,
    w0 * a.z + w1 * c.z + w2 * b.z,
  );
}

// ---------------------------------------------------------------------------
// Procedural textures
// ---------------------------------------------------------------------------

function woodCanvas() {
  const S = 512, rows = 8, rh = S / rows;
  const c = makeCanvas(S, S);
  const g = c.getContext('2d');
  const rnd = mulberry32(11);
  const plank = (x, y, w, light, sat, grains) => {
    g.fillStyle = `hsl(26, ${sat}%, ${light}%)`;
    g.fillRect(x, y, w, rh);
    for (const gr of grains) {
      g.strokeStyle = `rgba(38, 20, 8, ${gr.a})`;
      g.lineWidth = gr.w;
      g.beginPath();
      g.moveTo(x, y + gr.y);
      g.bezierCurveTo(x + w * 0.33, y + gr.y + gr.d1, x + w * 0.66, y + gr.y + gr.d2, x + w, y + gr.y + gr.d3);
      g.stroke();
    }
    g.fillStyle = 'rgba(18, 9, 3, 0.55)';
    g.fillRect(x, y, 2, rh);
  };
  for (let r = 0; r < rows; r++) {
    // Plank ends at random offsets inside [0, S); the last plank wraps so the texture tiles.
    const n = 2 + Math.floor(rnd() * 2);
    const ends = Array.from({ length: n }, () => rnd() * S).sort((a, b) => a - b);
    for (let i = 0; i < n; i++) {
      const x = ends[i];
      const w = (i + 1 < n ? ends[i + 1] : ends[0] + S) - x;
      const light = 27 + rnd() * 10, sat = 34 + rnd() * 14;
      const grains = Array.from({ length: 8 }, () => ({
        a: 0.06 + rnd() * 0.12, w: 0.6 + rnd() * 1.4, y: 3 + rnd() * (rh - 6),
        d1: (rnd() - 0.5) * 6, d2: (rnd() - 0.5) * 6, d3: (rnd() - 0.5) * 4,
      }));
      plank(x, r * rh, w, light, sat, grains);
      if (x + w > S) plank(x - S, r * rh, w, light, sat, grains);
    }
    g.fillStyle = 'rgba(18, 9, 3, 0.6)';
    g.fillRect(0, r * rh, S, 2);
  }
  return c;
}

function tileCanvas() {
  const S = 512, n = 4, ts = S / n;
  const c = makeCanvas(S, S);
  const g = c.getContext('2d');
  const rnd = mulberry32(23);
  g.fillStyle = '#6d6a66';
  g.fillRect(0, 0, S, S);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const l = 74 + rnd() * 8;
      g.fillStyle = `hsl(36, 9%, ${l}%)`;
      g.fillRect(i * ts + 2, j * ts + 2, ts - 4, ts - 4);
      for (let k = 0; k < 140; k++) {
        g.fillStyle = `rgba(${rnd() < 0.5 ? '255,255,255' : '60,55,50'}, ${0.04 + rnd() * 0.08})`;
        g.fillRect(i * ts + 2 + rnd() * (ts - 6), j * ts + 2 + rnd() * (ts - 6), 2 + rnd() * 3, 2 + rnd() * 3);
      }
    }
  }
  return c;
}

function noiseCanvas(size, base, spread, seed, count) {
  const c = makeCanvas(size, size);
  const g = c.getContext('2d');
  const rnd = mulberry32(seed);
  g.fillStyle = base;
  g.fillRect(0, 0, size, size);
  for (let i = 0; i < count; i++) {
    const v = rnd();
    g.fillStyle = v < 0.5 ? `rgba(255,255,255,${rnd() * spread})` : `rgba(0,0,0,${rnd() * spread * 1.4})`;
    const s = 1 + rnd() * 2.5;
    g.fillRect(rnd() * size, rnd() * size, s, s);
  }
  return c;
}

// Colour where the night sky meets the ground; the fog uses it too so the far
// lawn melts into the horizon.
const HORIZON = '#15223f';

/** Vertical sky gradient, used as an equirectangular background (row 0 = zenith, middle = horizon). */
function skyCanvas() {
  const c = makeCanvas(4, 512);
  const g = c.getContext('2d');
  const grad = g.createLinearGradient(0, 0, 0, 512);
  grad.addColorStop(0, '#010309');
  grad.addColorStop(0.25, '#040918');
  grad.addColorStop(0.42, '#0b1530');
  grad.addColorStop(0.5, HORIZON);
  grad.addColorStop(1, HORIZON);
  g.fillStyle = grad;
  g.fillRect(0, 0, 4, 512);
  return c;
}

function glowCanvas(size = 128, inner = 0.0) {
  const c = makeCanvas(size, size);
  const g = c.getContext('2d');
  const grad = g.createRadialGradient(size / 2, size / 2, size * inner, size / 2, size / 2, size / 2);
  grad.addColorStop(0, 'rgba(255,255,255,1)');
  grad.addColorStop(0.25, 'rgba(255,255,255,0.55)');
  grad.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = grad;
  g.fillRect(0, 0, size, size);
  return c;
}

function textCanvas(text) {
  const h = 96;
  const probe = makeCanvas(8, 8).getContext('2d');
  const font = '600 58px ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif';
  probe.font = font;
  if ('letterSpacing' in probe) probe.letterSpacing = '10px';
  const w = Math.ceil(probe.measureText(text).width + 40);
  const c = makeCanvas(w, h);
  const g = c.getContext('2d');
  g.font = font;
  if ('letterSpacing' in g) g.letterSpacing = '10px';
  g.textBaseline = 'middle';
  g.fillStyle = 'rgba(255, 246, 230, 0.92)';
  g.fillText(text, 20, h / 2 + 2);
  return c;
}

// ---------------------------------------------------------------------------
// Static environment: sky, lawn, street, house shell, furniture, lights
// ---------------------------------------------------------------------------

function buildEnvironment(scene, aniso) {
  const b = HOUSE.bounds;
  const occluders = [];   // solid static meshes the focus camera must see past

  // ---- sky, stars, moon, fog ------------------------------------------------
  // World-space sky: the gradient stays put when the camera tilts.
  const sky = canvasTexture(skyCanvas(), 1, false);
  sky.mapping = THREE.EquirectangularReflectionMapping;
  scene.background = sky;
  scene.fog = new THREE.Fog(HORIZON, 40, 120);

  {
    const rnd = mulberry32(5);
    const n = 520;
    const p = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const theta = rnd() * Math.PI * 2;
      const y = 0.12 + rnd() * 0.88;               // upper hemisphere only
      const r = Math.sqrt(1 - y * y);
      p[i * 3] = Math.cos(theta) * r * 170;
      p[i * 3 + 1] = y * 170;
      p[i * 3 + 2] = Math.sin(theta) * r * 170;
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(p, 3));
    const m = new THREE.PointsMaterial({ color: 0xcfd8ff, size: 1.4, sizeAttenuation: false, fog: false, transparent: true, opacity: 0.8, depthWrite: false });
    const stars = new THREE.Points(g, m);
    stars.frustumCulled = false;
    scene.add(stars);
  }

  const glowTex = canvasTexture(glowCanvas(), 1, false);
  {
    const moonMat = new THREE.SpriteMaterial({ map: glowTex, color: 0xdde6ff, fog: false, depthWrite: false, transparent: true });
    const moon = new THREE.Sprite(moonMat);
    moon.position.set(-40, 34, -78);
    moon.scale.setScalar(9);
    scene.add(moon);
    const discMat = new THREE.MeshBasicMaterial({ color: 0xf3f6ff, fog: false });
    const disc = new THREE.Mesh(new THREE.CircleGeometry(1.5, 32), discMat);
    disc.position.copy(moon.position);
    disc.lookAt(0, 0, 0);
    scene.add(disc);
  }

  // ---- lights ----------------------------------------------------------------
  const hemi = new THREE.HemisphereLight(0x6684c8, 0x1c1712, 1.15);
  scene.add(hemi);

  const moonLight = new THREE.DirectionalLight(0xb2c8ff, 2.1);
  moonLight.position.set(-13, 21, -9);
  moonLight.castShadow = true;
  moonLight.shadow.mapSize.set(2048, 2048);
  Object.assign(moonLight.shadow.camera, { left: -17, right: 17, top: 17, bottom: -17, near: 1, far: 70 });
  moonLight.shadow.bias = -0.0005;
  moonLight.shadow.normalBias = 0.03;
  scene.add(moonLight, moonLight.target);

  // Porch spot over the front door: the second (and last) shadow caster.
  const fd = HOUSE.frontDoor;
  const porch = new THREE.SpotLight(0xffc68a, 22, 11, 0.85, 0.65, 2);
  porch.position.set(fd.x - 0.85, 2.35, fd.z + 0.35);
  porch.target.position.set(fd.x, 0, fd.z + 1.4);
  porch.castShadow = true;
  porch.shadow.mapSize.set(512, 512);
  porch.shadow.bias = -0.001;
  scene.add(porch, porch.target);

  // Desk lamp in the study (not a smart device, always on).
  const deskLamp = new THREE.PointLight(0xffc27a, 3.2, 5.5, 2);
  deskLamp.position.set(6.45, 1.12, 5.25);
  scene.add(deskLamp);

  // ---- materials -------------------------------------------------------------
  const std = (color, roughness = 0.85, extra = {}) => new THREE.MeshStandardMaterial({ color, roughness, metalness: 0, ...extra });
  const M = {
    plaster: std(0xd9d1c4, 0.92),
    cap: std(0x23262e, 0.8),
    plinth: std(0x23272e, 0.95),
    fabric: std(0x46536e, 0.95),
    cushion: std(0x5a6886, 0.95),
    woodDark: std(0x4a3324, 0.65),
    woodLight: std(0x9a7452, 0.6),
    cabinet: std(0x2d3a43, 0.55),
    stone: std(0xe4dfd6, 0.35),
    bedding: std(0xe9e5ef, 0.9),
    duvet: std(0x5d4f86, 0.9),
    headboard: std(0x3b3150, 0.8),
    rug: std(0x6e3b3b, 1),
    rugBlue: std(0x2f4058, 1),
    metal: std(0x8d96a3, 0.35, { metalness: 0.6 }),
    black: std(0x15171c, 0.5),
    door: std(0x3a281d, 0.6),
    frame: std(0xe9e4da, 0.6),
    leaf: std(0x356b45, 0.9, { flatShading: true }),
    pine: std(0x2a543a, 0.9, { flatShading: true }),
    trunk: std(0x3b2a1f, 0.95),
    paving: std(0x3b3e46, 0.95),
    kerb: std(0x5b5f68, 0.9),
    lampPost: std(0x2a2e36, 0.5, { metalness: 0.5 }),
    glass: new THREE.MeshStandardMaterial({ color: 0x1b2a48, emissive: 0x1a2c55, emissiveIntensity: 0.55, roughness: 0.08, transparent: true, opacity: 0.5, depthWrite: false }),
    lampShade: new THREE.MeshStandardMaterial({ color: 0xffe2b0, emissive: 0xffc27a, emissiveIntensity: 1.4, roughness: 0.6, side: THREE.DoubleSide }),
    lampHead: new THREE.MeshBasicMaterial({ color: 0xffe4b8 }),
    lane: new THREE.MeshBasicMaterial({ color: 0xbdb79c, transparent: true, opacity: 0.55 }),
  };
  const parts = {};   // material key -> geometries to merge
  const add = (key, geo) => { (parts[key] ||= []).push(geo); };

  // ---- ground and street -----------------------------------------------------
  const lawnTex = canvasTexture(noiseCanvas(256, '#1f3a29', 0.12, 3, 5000), aniso);
  lawnTex.repeat.set(100, 100);   // ~6 m per tile
  // A wide disc whose rim lies far beyond the fog, so no ground edge is ever visible.
  const lawn = new THREE.Mesh(new THREE.CircleGeometry(300, 48), new THREE.MeshStandardMaterial({ color: 0x8aa792, map: lawnTex, roughness: 1 }));
  lawn.rotation.x = -Math.PI / 2;
  lawn.position.y = -0.02;
  lawn.receiveShadow = true;
  scene.add(lawn);

  const sz = HOUSE.street.z;                        // road centre line
  const roadNear = 7.75, roadFar = 2 * sz - roadNear;
  const asphaltTex = canvasTexture(noiseCanvas(256, '#1c1f26', 0.1, 9, 6000), aniso);
  const road = new THREE.Mesh(
    floorGeo(-110, 110, roadNear, roadFar, 3, -0.005),
    new THREE.MeshStandardMaterial({ color: 0xa9adb6, map: asphaltTex, roughness: 0.92 }),
  );
  road.receiveShadow = true;
  scene.add(road);
  add('kerb', boxGeo(-110, -0.02, roadNear - 0.12, 110, 0.07, roadNear));
  add('kerb', boxGeo(-110, -0.02, roadFar, 110, 0.07, roadFar + 0.12));
  add('paving', boxGeo(-110, -0.02, 7.0, 110, 0.05, roadNear - 0.12));                 // pavement
  add('paving', boxGeo(-110, -0.02, roadFar + 0.12, 110, 0.05, roadFar + 0.8));
  add('paving', boxGeo(fd.x - 0.65, -0.02, b.maxZ + T_EXT / 2, fd.x + 0.65, 0.02, 7.0)); // front path
  {
    const dashes = [];
    for (let x = -60; x <= 60; x += 3.2) dashes.push(boxGeo(x, -0.004, sz - 0.06, x + 1.5, 0.001, sz + 0.06));
    scene.add(new THREE.Mesh(mergeGeometries(dashes), M.lane));
    dashes.forEach(d => d.dispose());
  }

  // Street lamps: emissive heads, glow sprites and soft light pools on the ground.
  const poolTex = canvasTexture(glowCanvas(128, 0.05), 1, false);
  const poolMat = new THREE.MeshBasicMaterial({ map: poolTex, color: 0xffcf8f, transparent: true, opacity: 0.22, depthWrite: false, blending: THREE.AdditiveBlending });
  const lampGlowMat = new THREE.SpriteMaterial({ map: glowTex, color: 0xffcf8f, transparent: true, opacity: 0.85, depthWrite: false, blending: THREE.AdditiveBlending });
  const poolGeo = new THREE.PlaneGeometry(1, 1).rotateX(-Math.PI / 2);
  for (const x of [-9.5, 9.5]) {
    const z = 7.35;
    add('lampPost', cylGeo(x, z, 0.07, 0, 4.1, 8, 0.05));
    add('lampPost', boxGeo(x - 0.03, 4.0, z, x + 0.03, 4.08, z + 0.9));
    const head = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.08, 0.2), M.lampHead);
    head.position.set(x, 3.98, z + 0.88);
    scene.add(head);
    const glow = new THREE.Sprite(lampGlowMat);
    glow.position.copy(head.position);
    glow.scale.setScalar(1.6);
    scene.add(glow);
    const pool = new THREE.Mesh(poolGeo, poolMat);
    pool.position.set(x, 0.02, z + 0.9);
    pool.scale.set(7, 1, 7);
    scene.add(pool);
  }
  {
    const pool = new THREE.Mesh(poolGeo, poolMat);
    pool.position.set(fd.x, 0.03, fd.z + 1.0);
    pool.scale.set(3.4, 1, 2.6);
    scene.add(pool);
    const porchLamp = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.2, 0.1), M.lampHead);
    porchLamp.position.set(fd.x - 0.78, 2.0, fd.z + T_EXT / 2 + 0.06);
    scene.add(porchLamp);
    const glow = new THREE.Sprite(lampGlowMat);
    glow.position.copy(porchLamp.position);
    glow.scale.setScalar(0.9);
    scene.add(glow);
  }

  // Trees and hedges: transformed copies merged into one mesh per material
  // (three draw calls instead of thirty-odd), flat shaded.
  {
    const cone = new THREE.ConeGeometry(1, 1, 7);
    const trunk = new THREE.CylinderGeometry(0.12, 0.16, 1, 6);
    const bush = new THREE.IcosahedronGeometry(1, 0);
    const m4 = new THREE.Matrix4();
    const q = new THREE.Quaternion();
    const up = new THREE.Vector3(0, 1, 0);
    const at = new THREE.Vector3();
    const sc = new THREE.Vector3();
    const placed = (geo, x, y, z, sx, sy, sz, rotY = 0) =>
      geo.clone().applyMatrix4(m4.compose(at.set(x, y, z), q.setFromAxisAngle(up, rotY), sc.set(sx, sy, sz)));
    const trees = [[-12.5, -3.5, 1.2], [-11.8, 3.6, 1.0], [12.4, -3.2, 1.25], [12.2, 4.4, 0.95], [-6.5, -10, 1.3], [6.8, -10.5, 1.15], [13.5, -10.5, 1.4], [-14, -11, 1.2], [-3, -12.5, 1.0]];
    for (const [x, z, s] of trees) {
      add('trunk', placed(trunk, x, 0.5 * s, z, s, s, s));
      for (let k = 0; k < 2; k++) {
        const w = 1.25 * s * (1 - k * 0.25);
        add('pine', placed(cone, x, s * (1.6 + k * 1.0), z, w, 1.9 * s, w, x + k));
      }
    }
    for (const [x, z, s] of [[-3.6, 6.75, 0.5], [-6.4, 6.7, 0.55], [-8.4, 6.8, 0.45], [3.4, 6.75, 0.5], [6.3, 6.7, 0.55], [8.3, 6.8, 0.45], [-10, -1, 0.6], [10, 1.5, 0.55]]) {
      add('leaf', placed(bush, x, s * 0.7, z, s * 1.3, s, s * 1.1, x * 1.7));
    }
    cone.dispose();
    trunk.dispose();
    bush.dispose();
  }


  // ---- house: foundation and floors -----------------------------------------
  add('plinth', boxGeo(b.minX - 0.25, -0.3, b.minZ - 0.25, b.maxX + 0.25, -0.003, b.maxZ + 0.25));

  const floorTex = {
    wood: { tex: canvasTexture(woodCanvas(), aniso), tile: 2.0, rough: 0.62, color: 0xffffff },
    tile: { tex: canvasTexture(tileCanvas(), aniso), tile: 2.4, rough: 0.32, color: 0xffffff },
    carpet: { tex: canvasTexture(noiseCanvas(256, '#4a4160', 0.14, 17, 9000), aniso), tile: 1.6, rough: 1, color: 0xffffff },
  };
  const floorsByKind = {};
  for (const r of HOUSE.rooms) {
    const kind = floorTex[r.floor] ? r.floor : 'wood';
    (floorsByKind[kind] ||= []).push(floorGeo(r.x[0], r.x[1], r.z[0], r.z[1], floorTex[kind].tile, 0));
  }
  for (const [kind, geos] of Object.entries(floorsByKind)) {
    const f = floorTex[kind];
    const mesh = new THREE.Mesh(mergeGeometries(geos), new THREE.MeshStandardMaterial({ color: f.color, map: f.tex, roughness: f.rough }));
    mesh.receiveShadow = true;
    scene.add(mesh);
    geos.forEach(g => g.dispose());
  }

  // ---- walls -----------------------------------------------------------------
  const walls = [];
  const caps = [];
  const windows = [];   // {axis, at, from, to, bottom, top}
  /** A straight wall with openings (world coordinates along the wall). */
  const wall = (axis, at, from, to, h, t, openings = []) => {
    const piece = (a, c, y0, y1) => {
      if (c - a < 0.004 || y1 - y0 < 0.004) return;
      walls.push(axis === 'x' ? boxGeo(a, y0, at - t / 2, c, y1, at + t / 2) : boxGeo(at - t / 2, y0, a, at + t / 2, y1, c));
      if (Math.abs(y1 - h) < 1e-6) {
        caps.push(axis === 'x'
          ? boxGeo(a, h, at - t / 2 - 0.006, c, h + CAP, at + t / 2 + 0.006)
          : boxGeo(at - t / 2 - 0.006, h, a, at + t / 2 + 0.006, h + CAP, c));
      }
    };
    let cursor = from;
    for (const o of [...openings].sort((p, q) => p.from - q.from)) {
      piece(cursor, o.from, 0, h);
      piece(o.from, o.to, 0, o.bottom ?? 0);
      piece(o.from, o.to, o.top ?? h, h);
      cursor = o.to;
      if (o.window) windows.push({ axis, at, t, ...o });
    }
    piece(cursor, to, 0, h);
  };

  const H = HOUSE.wallHeight;
  const half = T_EXT / 2;
  // Back (far) walls at full height with windows; front and right cut away.
  wall('x', b.minZ, b.minX - half, b.maxX + half, H, T_EXT, [
    { from: -2.0, to: -0.4, bottom: 0.9, top: 2.2, window: true },
    { from: 4.2, to: 5.8, bottom: 1.15, top: 2.1, window: true },
  ]);
  wall('z', b.minX, b.minZ + half, b.maxZ - half, H, T_EXT, [
    { from: -1.3, to: 0.2, bottom: 0.9, top: 2.2, window: true },
    { from: 2.5, to: 3.9, bottom: 1.35, top: 2.2, window: true },
  ]);
  wall('x', b.maxZ, b.minX - half, b.maxX + half, H_LOW, T_EXT, [
    { from: fd.x - fd.width / 2, to: fd.x + fd.width / 2, bottom: 0, top: H_LOW },
  ]);
  wall('z', b.maxX, b.minZ + half, b.maxZ - half, H_LOW, T_EXT);

  // Entrance portal: door frame plus a pier that carries the doorbell camera.
  {
    const dl = fd.x - fd.width / 2, dr = fd.x + fd.width / 2, top = 2.45;
    const z0 = fd.z - half, z1 = fd.z + half;
    walls.push(boxGeo(dl - 0.14, H_LOW, z0, dl, top, z1));
    walls.push(boxGeo(dr, H_LOW, z0, dr + 0.92, top, z1));
    walls.push(boxGeo(dl, 2.14, z0, dr, top, z1));
    caps.push(boxGeo(dl - 0.14, top, z0 - 0.006, dr + 0.92, top + CAP, z1 + 0.006));
    add('door', boxGeo(dl + 0.01, 0, fd.z - 0.07, dr - 0.01, 2.13, fd.z + 0.03));
    add('metal', boxGeo(dr - 0.2, 0.98, fd.z + 0.03, dr - 0.12, 1.04, fd.z + 0.07));
  }

  // Interior walls are found from the room plan: every shared room edge.
  const interiorHeight = (axis, at) => (HALF_WALLS.some(w => w.axis === axis && w.at === at) ? H_HALF : H_INT);
  const rooms = HOUSE.rooms;
  for (let i = 0; i < rooms.length; i++) {
    for (let j = i + 1; j < rooms.length; j++) {
      const p = rooms[i], q = rooms[j];
      for (const x of [p.x[0], p.x[1]]) {
        if (x !== q.x[0] && x !== q.x[1]) continue;
        if (x === b.minX || x === b.maxX || (x === p.x[0] && x === q.x[0]) || (x === p.x[1] && x === q.x[1])) continue;
        const lo = Math.max(p.z[0], q.z[0]), hi = Math.min(p.z[1], q.z[1]);
        if (hi - lo > 0.01) wall('z', x, lo, hi, interiorHeight('z', x), T_INT, INTERIOR_OPENINGS.filter(o => o.axis === 'z' && o.at === x && o.from >= lo && o.to <= hi));
      }
      for (const z of [p.z[0], p.z[1]]) {
        if (z !== q.z[0] && z !== q.z[1]) continue;
        if (z === b.minZ || z === b.maxZ || (z === p.z[0] && z === q.z[0]) || (z === p.z[1] && z === q.z[1])) continue;
        const lo = Math.max(p.x[0], q.x[0]), hi = Math.min(p.x[1], q.x[1]);
        if (hi - lo > 0.01) wall('x', z, lo, hi, interiorHeight('x', z), T_INT, INTERIOR_OPENINGS.filter(o => o.axis === 'x' && o.at === z && o.from >= lo && o.to <= hi));
      }
    }
  }
  {
    const wallMesh = new THREE.Mesh(mergeGeometries(walls), M.plaster);
    wallMesh.castShadow = true;
    wallMesh.receiveShadow = true;
    scene.add(wallMesh);
    occluders.push(wallMesh);
    const capMesh = new THREE.Mesh(mergeGeometries(caps), M.cap);
    scene.add(capMesh);
    walls.forEach(g => g.dispose());
    caps.forEach(g => g.dispose());
  }

  // Window glass and frames.
  {
    const glass = [];
    for (const w of windows) {
      const gt = 0.03, ft = w.t + 0.02, fw = 0.05;
      if (w.axis === 'x') {
        glass.push(boxGeo(w.from, w.bottom, w.at - gt / 2, w.to, w.top, w.at + gt / 2));
        add('frame', boxGeo(w.from, w.bottom - fw, w.at - ft / 2, w.to, w.bottom, w.at + ft / 2));
        add('frame', boxGeo((w.from + w.to) / 2 - 0.025, w.bottom, w.at - 0.03, (w.from + w.to) / 2 + 0.025, w.top, w.at + 0.03));
      } else {
        glass.push(boxGeo(w.at - gt / 2, w.bottom, w.from, w.at + gt / 2, w.top, w.to));
        add('frame', boxGeo(w.at - ft / 2, w.bottom - fw, w.from, w.at + ft / 2, w.bottom, w.to));
        add('frame', boxGeo(w.at - 0.03, w.bottom, (w.from + w.to) / 2 - 0.025, w.at + 0.03, w.top, (w.from + w.to) / 2 + 0.025));
      }
    }
    const glassMesh = new THREE.Mesh(mergeGeometries(glass), M.glass);
    glassMesh.renderOrder = 2;
    scene.add(glassMesh);
    glass.forEach(g => g.dispose());
  }

  // ---- furniture (unobtrusive blocks, merged per material) -----------------
  // Living room: TV console, sofa, coffee table, rug, speaker side table, router shelf, plant.
  add('woodDark', boxGeo(-5.2, 0, -5.92, -2.8, 0.45, -5.5));
  add('black', boxGeo(-5.1, 0.12, -5.505, -2.9, 0.34, -5.49));
  add('fabric', boxGeo(-5.15, 0.08, -1.25, -2.85, 0.38, -0.42));
  add('fabric', boxGeo(-5.35, 0.08, -0.42, -2.65, 0.9, -0.2));
  add('fabric', boxGeo(-5.35, 0.08, -1.25, -5.15, 0.64, -0.42));
  add('fabric', boxGeo(-2.85, 0.08, -1.25, -2.65, 0.64, -0.42));
  add('cushion', boxGeo(-5.13, 0.38, -1.23, -4.02, 0.5, -0.44));
  add('cushion', boxGeo(-3.98, 0.38, -1.23, -2.87, 0.5, -0.44));
  for (const [x, z] of [[-5.3, -1.2], [-2.7, -1.2], [-5.3, -0.25], [-2.7, -0.25]]) add('black', boxGeo(x - 0.04, 0, z - 0.04, x + 0.04, 0.08, z + 0.04));
  add('woodLight', boxGeo(-4.6, 0.36, -2.9, -3.4, 0.42, -2.3));
  for (const [x, z] of [[-4.52, -2.82], [-3.48, -2.82], [-4.52, -2.38], [-3.48, -2.38]]) add('woodDark', boxGeo(x - 0.03, 0, z - 0.03, x + 0.03, 0.36, z + 0.03));
  add('rug', boxGeo(-5.7, 0.002, -3.7, -2.3, 0.012, -1.4));
  add('woodDark', cylGeo(-6.6, -2.2, 0.26, 0.58, 0.62, 18));
  add('woodDark', cylGeo(-6.6, -2.2, 0.05, 0.02, 0.58, 8));
  add('woodDark', cylGeo(-6.6, -2.2, 0.2, 0, 0.03, 14));
  add('woodLight', boxGeo(-0.85, 0.12, 0.1, 0.93, 0.92, 0.93));     // router shelf
  add('woodDark', boxGeo(-0.88, 0.92, 0.08, 0.94, 0.95, 0.94));
  add('black', boxGeo(-0.8, 0, 0.16, 0.88, 0.12, 0.9));
  add('black', boxGeo(0.05, 0.3, 0.095, 0.07, 0.75, 0.1));
  add('cabinet', cylGeo(0.4, -5.45, 0.2, 0, 0.42, 14, 0.24));
  {
    const leaves = new THREE.Mesh(new THREE.IcosahedronGeometry(0.42, 0), M.leaf);
    leaves.position.set(0.4, 0.86, -5.45);
    leaves.scale.set(1, 1.15, 1);
    leaves.castShadow = true;
    scene.add(leaves);
    occluders.push(leaves);
  }

  // Kitchen: counter run, upper cabinets, sink, coffee maker, island and stools.
  add('cabinet', boxGeo(1.06, 0, -5.92, 7.5, 0.9, -5.32));
  add('stone', boxGeo(1.06, 0.9, -5.92, 7.55, 0.95, -5.27));
  add('cabinet', boxGeo(1.06, 1.55, -5.92, 3.9, 2.2, -5.6));
  add('cabinet', boxGeo(6.1, 1.55, -5.92, 7.55, 2.2, -5.6));
  add('metal', boxGeo(4.55, 0.95, -5.85, 5.45, 0.956, -5.4));
  add('black', boxGeo(2.68, 0.95, -5.85, 3.05, 1.3, -5.55));
  add('cabinet', boxGeo(3.8, 0, -3.4, 6.2, 0.9, -2.4));
  add('stone', boxGeo(3.72, 0.9, -3.48, 6.28, 0.95, -2.32));
  for (const x of [4.3, 5.0, 5.7]) {
    add('metal', cylGeo(x, -1.95, 0.035, 0, 0.64, 6));
    add('woodDark', cylGeo(x, -1.95, 0.19, 0.64, 0.7, 14));
  }

  // Bedroom: bed, bedside tables, wardrobe, rug.
  add('woodDark', boxGeo(-8.93, 0, 3.0, -6.8, 0.32, 4.75));
  add('bedding', boxGeo(-8.85, 0.32, 3.06, -6.88, 0.52, 4.69));
  add('duvet', boxGeo(-8.05, 0.5, 3.03, -6.84, 0.58, 4.72));
  add('bedding', boxGeo(-8.8, 0.52, 3.15, -8.4, 0.64, 3.85));
  add('bedding', boxGeo(-8.8, 0.52, 3.9, -8.4, 0.64, 4.6));
  add('headboard', boxGeo(-8.92, 0, 2.95, -8.82, 1.15, 4.8));
  add('woodLight', boxGeo(-8.36, 0, 4.95, -7.86, 0.62, 5.45));
  add('woodLight', boxGeo(-8.36, 0, 2.3, -7.86, 0.62, 2.8));
  add('woodLight', boxGeo(-5.5, 0, 1.05, -3.6, 1.75, 1.66));
  add('black', boxGeo(-4.56, 0.1, 1.655, -4.54, 1.65, 1.67));
  add('rugBlue', boxGeo(-6.6, 0.002, 3.0, -4.6, 0.012, 5.2));

  // Hallway: bench, runner, doormat.
  add('woodLight', boxGeo(0.45, 0.38, 3.6, 0.93, 0.45, 4.8));
  add('woodDark', boxGeo(0.5, 0, 3.66, 0.88, 0.38, 3.72));
  add('woodDark', boxGeo(0.5, 0, 4.68, 0.88, 0.38, 4.74));
  add('rug', boxGeo(-0.45, 0.002, 1.4, 0.45, 0.012, 5.45));
  add('black', boxGeo(-0.5, 0.002, 5.5, 0.5, 0.016, 5.88));

  // Utility: washer alcove, storage cabinet, basket, power strip.
  add('cabinet', boxGeo(1.58, 0, 4.95, 1.66, 1.6, 5.92));
  add('cabinet', boxGeo(2.54, 0, 4.95, 2.62, 1.6, 5.92));
  add('woodLight', boxGeo(1.58, 1.55, 4.95, 2.62, 1.6, 5.92));
  add('cabinet', boxGeo(1.06, 0, 3.0, 1.6, 1.9, 4.2));
  add('woodLight', cylGeo(4.35, 4.85, 0.22, 0, 0.45, 12, 0.25));
  add('black', boxGeo(3.45, 0, 5.7, 4.05, 0.05, 5.82));

  // Study: desk, chair, bookshelf, desk lamp.
  add('woodLight', boxGeo(6.1, 0.73, 4.62, 7.9, 0.78, 5.6));
  add('cabinet', boxGeo(6.12, 0, 4.66, 6.18, 0.73, 5.56));
  add('cabinet', boxGeo(7.82, 0, 4.66, 7.88, 0.73, 5.56));
  add('cabinet', boxGeo(6.18, 0.35, 5.5, 7.82, 0.73, 5.56));
  add('fabric', boxGeo(6.75, 0.44, 3.95, 7.25, 0.5, 4.45));
  add('fabric', boxGeo(6.75, 0.5, 3.88, 7.25, 0.98, 3.95));
  add('metal', cylGeo(7.0, 4.2, 0.035, 0.07, 0.44, 6));
  add('black', cylGeo(7.0, 4.2, 0.26, 0, 0.07, 10));
  add('woodDark', boxGeo(5.5, 0, 0.05, 7.3, 1.9, 0.09));
  add('woodDark', boxGeo(5.5, 0, 0.05, 5.55, 1.9, 0.42));
  add('woodDark', boxGeo(7.25, 0, 0.05, 7.3, 1.9, 0.42));
  for (const y of [0.02, 0.48, 0.94, 1.4, 1.86]) add('woodDark', boxGeo(5.55, y, 0.09, 7.25, y + 0.04, 0.42));
  add('black', cylGeo(6.45, 5.3, 0.08, 0.78, 0.8, 12));
  add('metal', cylGeo(6.45, 5.3, 0.012, 0.8, 1.15, 5));
  {
    const shade = new THREE.Mesh(new THREE.ConeGeometry(0.12, 0.14, 14, 1, true), M.lampShade);
    shade.position.set(6.45, 1.2, 5.3);
    scene.add(shade);
  }
  // Books: one instanced mesh with per-instance colours.
  {
    const rnd = mulberry32(31);
    const hues = [0x8c3b3b, 0x2f5d7c, 0xc29b45, 0x3d6b4f, 0x6b4f86, 0xd2c6a8, 0x9a5a2e];
    const spots = [];
    for (const y of [0.06, 0.52, 0.98, 1.44]) {
      let x = 5.6;
      while (x < 7.15) {
        const w = 0.04 + rnd() * 0.05, h = 0.26 + rnd() * 0.14;
        if (rnd() < 0.12) { x += 0.12; continue; }
        spots.push([x + w / 2, y + h / 2, w, h, hues[Math.floor(rnd() * hues.length)]]);
        x += w + 0.005;
      }
    }
    const books = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 0.26), std(0xffffff, 0.8), spots.length);
    const m = new THREE.Matrix4();
    const c = new THREE.Color();
    spots.forEach(([x, y, w, h, hex], i) => {
      m.makeScale(w, h, 1).setPosition(x, y, 0.25);
      books.setMatrixAt(i, m);
      books.setColorAt(i, c.setHex(hex));
    });
    scene.add(books);
  }

  // ---- merge everything collected above, one mesh per material --------------
  for (const [key, geos] of Object.entries(parts)) {
    const mesh = new THREE.Mesh(mergeGeometries(geos), M[key]);
    const flat = key === 'rug' || key === 'rugBlue' || key === 'paving' || key === 'kerb' || key === 'plinth';
    mesh.castShadow = !flat;
    mesh.receiveShadow = true;
    scene.add(mesh);
    if (!flat) occluders.push(mesh);
    geos.forEach(g => g.dispose());
  }

  // ---- room names painted on the floors -------------------------------------
  for (const r of HOUSE.rooms) {
    const spot = ROOM_LABEL_SPOTS[r.id] || [(r.x[0] + r.x[1]) / 2, (r.z[0] + r.z[1]) / 2, false];
    const canvas = textCanvas(r.name.toUpperCase());
    const tex = canvasTexture(canvas, aniso, false);
    const h = 0.24, w = h * canvas.width / canvas.height;
    const mat = new THREE.MeshBasicMaterial({ map: tex, transparent: true, opacity: 0.42, depthWrite: false, toneMapped: false, polygonOffset: true, polygonOffsetFactor: -2 });
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(w, h), mat);
    mesh.rotation.set(-Math.PI / 2, 0, spot[2] ? Math.PI / 2 : 0);
    mesh.position.set(spot[0], 0.016, spot[1]);
    mesh.renderOrder = 1;
    scene.add(mesh);
  }

  return { glowTex, occluders };
}

// ---------------------------------------------------------------------------
// Packets and sparks: one pooled THREE.Points draw call
// ---------------------------------------------------------------------------

const KIND_ALLOW = 0, KIND_ALERT = 1, KIND_DROP = 2;

function createPacketSystem(parent, hooks) {
  const POINTS = MAX_PACKETS * TRAIL + MAX_SPARKS;
  const pos = new Float32Array(POINTS * 3);
  const col = new Float32Array(POINTS * 3);
  const size = new Float32Array(POINTS);
  const alpha = new Float32Array(POINTS);
  const geo = new THREE.BufferGeometry();
  const attr = (arr, n) => new THREE.BufferAttribute(arr, n).setUsage(THREE.DynamicDrawUsage);
  const aPos = attr(pos, 3), aCol = attr(col, 3), aSize = attr(size, 1), aAlpha = attr(alpha, 1);
  geo.setAttribute('position', aPos);
  geo.setAttribute('aColor', aCol);
  geo.setAttribute('aSize', aSize);
  geo.setAttribute('aAlpha', aAlpha);
  geo.setDrawRange(0, 0);

  const material = new THREE.ShaderMaterial({
    // uScale: pixels per world unit at distance 1. uKnee: above this many pixels
    // a sprite grows only gently, so close-ups do not fill the view with blobs.
    uniforms: { uScale: { value: 600 }, uKnee: { value: 16 } },
    vertexShader: /* glsl */`
      attribute vec3 aColor;
      attribute float aSize;
      attribute float aAlpha;
      uniform float uScale;
      uniform float uKnee;
      varying vec3 vColor;
      varying float vAlpha;
      void main() {
        vColor = aColor;
        vAlpha = aAlpha;
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        float px = aSize * uScale / -mv.z;
        if (px > uKnee) px = uKnee + (px - uKnee) * 0.3;
        gl_PointSize = clamp(px, 1.5, 72.0);
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: /* glsl */`
      varying vec3 vColor;
      varying float vAlpha;
      void main() {
        float d = length(gl_PointCoord - 0.5) * 2.0;
        float core = smoothstep(0.42, 0.0, d);
        float halo = smoothstep(1.0, 0.1, d);
        float a = (core + halo * halo * 0.55) * vAlpha;
        if (a < 0.01) discard;
        gl_FragColor = vec4(mix(vColor, vec3(1.0), core * 0.55), a);
      }`,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
  const points = new THREE.Points(geo, material);
  points.frustumCulled = false;
  points.renderOrder = 6;
  parent.add(points);

  // Packet slots (struct of arrays).
  const active = new Uint8Array(MAX_PACKETS);
  const kind = new Uint8Array(MAX_PACKETS);
  const segN = new Uint8Array(MAX_PACKETS);
  const burst = new Uint8Array(MAX_PACKETS);       // 0 none, 1 burst at the end of the path
  const shieldDone = new Uint8Array(MAX_PACKETS);
  const elapsed = new Float32Array(MAX_PACKETS);
  const duration = new Float32Array(MAX_PACKETS);
  const shieldAt = new Float32Array(MAX_PACKETS);  // path fraction where the shield sits, or -1
  const born = new Float64Array(MAX_PACKETS);
  const curve = new Float32Array(MAX_PACKETS * MAX_SEGS * 9);
  const cumLen = new Float32Array(MAX_PACKETS * MAX_SEGS);
  const free = new Int16Array(MAX_PACKETS);
  let freeTop = 0;
  for (let i = MAX_PACKETS - 1; i >= 0; i--) free[freeTop++] = i;
  let activeCount = 0;
  let serial = 0;

  // Sparks (ring buffer).
  const sp = new Float32Array(MAX_SPARKS * 3);
  const sv = new Float32Array(MAX_SPARKS * 3);
  const sLife = new Float32Array(MAX_SPARKS);
  const sMax = new Float32Array(MAX_SPARKS);
  const sSize = new Float32Array(MAX_SPARKS);
  const sKind = new Uint8Array(MAX_SPARKS);
  let sNext = 0;

  const RGB = [hexToRgb(HEX.allowed, [0, 0, 0]), hexToRgb(HEX.alerted, [0, 0, 0]), hexToRgb(HEX.dropped, [0, 0, 0])];
  const HEAD_SIZE = [0.2, 0.24, 0.25];
  const tmpA = new THREE.Vector3(), tmpB = new THREE.Vector3(), tmpC = new THREE.Vector3(), tmpOut = new THREE.Vector3();

  function release(i) {
    active[i] = 0;
    free[freeTop++] = i;
    activeCount--;
  }

  /** Choose a slot. Allowed packets are sampled under load; alerted/dropped always win a slot. */
  function acquire(k, cap) {
    if (k === KIND_ALLOW) {
      const load = activeCount / cap;
      if (load >= 1) return -1;
      if (load > 0.6 && Math.random() > (1 - load) / 0.4) return -1;
    }
    if (activeCount < cap && freeTop > 0) return free[--freeTop];
    // Full: evict the oldest allowed packet, else the oldest packet of any kind.
    let victim = -1, oldest = Infinity, victimAny = -1, oldestAny = Infinity;
    for (let i = 0; i < MAX_PACKETS; i++) {
      if (!active[i]) continue;
      if (kind[i] === KIND_ALLOW && born[i] < oldest) { oldest = born[i]; victim = i; }
      if (born[i] < oldestAny) { oldestAny = born[i]; victimAny = i; }
    }
    const v = victim >= 0 ? victim : victimAny;
    if (v < 0) return -1;
    activeCount--;
    return v;
  }

  /**
   * Start a packet along the waypoints wp[0..n-1] (Vector3) with bezier controls
   * ctrl[0..n-2]. shieldIndex is the waypoint index of the shield (or -1).
   */
  function spawn(k, wp, ctrl, n, shieldIndex, burstAtEnd, cap) {
    if (n < 2) return false;
    const i = acquire(k, cap);
    if (i < 0) return false;
    active[i] = 1;
    activeCount++;
    kind[i] = k;
    born[i] = serial++;
    elapsed[i] = 0;
    burst[i] = burstAtEnd ? 1 : 0;
    shieldDone[i] = 0;
    const segs = Math.min(n - 1, MAX_SEGS);
    segN[i] = segs;
    let total = 0, shieldLen = -1;
    for (let s = 0; s < segs; s++) {
      const a = wp[s], c = ctrl[s], b = wp[s + 1];
      const o = (i * MAX_SEGS + s) * 9;
      curve[o] = a.x; curve[o + 1] = a.y; curve[o + 2] = a.z;
      curve[o + 3] = c.x; curve[o + 4] = c.y; curve[o + 5] = c.z;
      curve[o + 6] = b.x; curve[o + 7] = b.y; curve[o + 8] = b.z;
      // Bezier length estimate: average of chord and control polygon.
      total += (a.distanceTo(b) + a.distanceTo(c) + c.distanceTo(b)) / 2;
      cumLen[i * MAX_SEGS + s] = total;
      if (s + 1 === shieldIndex) shieldLen = total;
    }
    duration[i] = clamp(0.6 + total * 0.022, 0.6, 1.2);
    shieldAt[i] = shieldLen >= 0 && total > 0 ? shieldLen / total : -1;
    return true;
  }

  /** Position along packet i's path at fraction f (0..1). */
  function sample(i, f, out) {
    const segs = segN[i];
    const base = i * MAX_SEGS;
    const target = f * cumLen[base + segs - 1];
    let s = 0;
    while (s < segs - 1 && cumLen[base + s] < target) s++;
    const start = s === 0 ? 0 : cumLen[base + s - 1];
    const len = cumLen[base + s] - start;
    const u = len > 1e-6 ? clamp((target - start) / len, 0, 1) : 1;
    const o = (base + s) * 9;
    tmpA.set(curve[o], curve[o + 1], curve[o + 2]);
    tmpC.set(curve[o + 3], curve[o + 4], curve[o + 5]);
    tmpB.set(curve[o + 6], curve[o + 7], curve[o + 8]);
    return bezierAt(tmpA, tmpC, tmpB, u, out);
  }

  function spawnBurst(p, k, count) {
    for (let n = 0; n <= count; n++) {
      const j = sNext;
      sNext = (sNext + 1) % MAX_SPARKS;
      sp[j * 3] = p.x; sp[j * 3 + 1] = p.y; sp[j * 3 + 2] = p.z;
      if (n === 0) {
        // Central flash.
        sv[j * 3] = sv[j * 3 + 1] = sv[j * 3 + 2] = 0;
        sMax[j] = sLife[j] = 0.45;
        sSize[j] = 0.75;
      } else {
        const th = Math.random() * Math.PI * 2;
        const up = Math.random() * 0.9 + 0.1;
        const sp0 = 0.9 + Math.random() * 1.1;
        sv[j * 3] = Math.cos(th) * sp0;
        sv[j * 3 + 1] = up * sp0;
        sv[j * 3 + 2] = Math.sin(th) * sp0;
        sMax[j] = sLife[j] = 0.4 + Math.random() * 0.35;
        sSize[j] = 0.07 + Math.random() * 0.05;
      }
      sKind[j] = k;
    }
  }

  function writePoint(n, x, y, z, rgb, s, a) {
    const o = n * 3;
    pos[o] = x; pos[o + 1] = y; pos[o + 2] = z;
    col[o] = rgb[0]; col[o + 1] = rgb[1]; col[o + 2] = rgb[2];
    size[n] = s;
    alpha[n] = a;
  }

  function update(dt, trail, burstCount) {
    let n = 0;
    for (let i = 0; i < MAX_PACKETS; i++) {
      if (!active[i]) continue;
      elapsed[i] += dt;
      const f = elapsed[i] / duration[i];
      const k = kind[i];
      if (!shieldDone[i] && shieldAt[i] >= 0 && f >= shieldAt[i]) {
        shieldDone[i] = 1;
        hooks.onShield(k);
      }
      if (f >= 1) {
        if (burst[i]) {
          sample(i, 1, tmpOut);
          spawnBurst(tmpOut, k, burstCount);
          hooks.onBurst(k, tmpOut);
        }
        release(i);
        continue;
      }
      const rgb = RGB[k];
      sample(i, f, tmpOut);
      writePoint(n++, tmpOut.x, tmpOut.y, tmpOut.z, rgb, HEAD_SIZE[k], 1);
      if (trail) {
        for (let g = 1; g < TRAIL; g++) {
          const fg = f - g * 0.035;
          if (fg <= 0) break;
          sample(i, fg, tmpOut);
          writePoint(n++, tmpOut.x, tmpOut.y, tmpOut.z, rgb, HEAD_SIZE[k] * (1 - g * 0.28), 0.55 - g * 0.17);
        }
      }
    }
    for (let j = 0; j < MAX_SPARKS; j++) {
      if (sLife[j] <= 0) continue;
      sLife[j] -= dt;
      if (sLife[j] <= 0) continue;
      const o = j * 3;
      sv[o + 1] -= 3.2 * dt;                       // a little gravity
      sp[o] += sv[o] * dt; sp[o + 1] += sv[o + 1] * dt; sp[o + 2] += sv[o + 2] * dt;
      const life = sLife[j] / sMax[j];
      const flash = sSize[j] > 0.5;
      writePoint(n++, sp[o], sp[o + 1], sp[o + 2], RGB[sKind[j]], flash ? sSize[j] * (1.4 - life * 0.6) : sSize[j], flash ? life * life : life);
    }
    geo.setDrawRange(0, n);
    if (n > 0) {
      // Upload only the live prefix of each buffer.
      for (let a = 0; a < ATTRS.length; a++) {
        const at = ATTRS[a];
        at.clearUpdateRanges();
        at.addUpdateRange(0, n * at.itemSize);
        at.needsUpdate = true;
      }
    }
  }
  const ATTRS = [aPos, aCol, aSize, aAlpha];

  function clear() {
    for (let i = 0; i < MAX_PACKETS; i++) if (active[i]) release(i);
    sLife.fill(0);
    geo.setDrawRange(0, 0);
  }

  function setScale(px, dprNow) {
    material.uniforms.uScale.value = px;
    material.uniforms.uKnee.value = 16 * dprNow;
  }

  function dispose() {
    parent.remove(points);
    geo.dispose();
    material.dispose();
  }

  return { spawn, update, clear, setScale, dispose, get count() { return activeCount; } };
}

// ---------------------------------------------------------------------------
// Fallback model if models.js throws for a device (keeps the rest of the scene alive)
// ---------------------------------------------------------------------------

function fallbackModel() {
  const geo = new THREE.BoxGeometry(0.3, 0.2, 0.3);
  geo.translate(0, 0.1, 0);
  const mat = new THREE.MeshStandardMaterial({ color: 0x8892a6, roughness: 0.6 });
  const group = new THREE.Group();
  group.add(new THREE.Mesh(geo, mat));
  return {
    group,
    port: new THREE.Vector3(0, 0.22, 0),
    update() {},
    setStatus(status) { mat.emissive.setHex(status === 'blocked' ? 0x661018 : status === 'alert' ? 0x553300 : 0x000000); },
    dispose() { geo.dispose(); mat.dispose(); },
  };
}

// ---------------------------------------------------------------------------
// createScene
// ---------------------------------------------------------------------------

export async function createScene({ bus, state, container }) {
  if (!container) throw new Error('3D container element not found');

  // ---- renderer, overlay, camera, controls ----------------------------------
  const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
  const maxDpr = Math.min(window.devicePixelRatio || 1, 2);
  let dpr = maxDpr;
  renderer.setPixelRatio(dpr);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  // The house is static; only a few small parts move (fan blades, doors, the
  // vacuum), so the shadow maps are redrawn at ~20 Hz instead of every frame.
  renderer.shadowMap.autoUpdate = false;
  renderer.shadowMap.needsUpdate = true;
  const canvas = renderer.domElement;
  canvas.style.display = 'block';
  canvas.style.touchAction = 'none';
  canvas.setAttribute('aria-label', '3D view of the smart home. Drag to orbit, scroll to zoom, click a device to select it.');

  if (getComputedStyle(container).position === 'static') container.style.position = 'relative';
  container.appendChild(canvas);

  if (!document.getElementById('sc-scene-style')) {
    const style = document.createElement('style');
    style.id = 'sc-scene-style';
    style.textContent = LABEL_CSS;
    document.head.appendChild(style);
  }
  const labelRenderer = new CSS2DRenderer();
  labelRenderer.domElement.className = 'sc-overlay';
  container.appendChild(labelRenderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 400);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 2.2;
  controls.maxDistance = 55;
  controls.minPolarAngle = 0.12;
  controls.maxPolarAngle = 1.36;            // ~78 deg: never dips below the ground
  controls.screenSpacePanning = false;      // pan across the floor plane
  controls.rotateSpeed = 0.7;
  controls.zoomSpeed = 0.9;
  controls.target.copy(HOME_TARGET);

  const reducedQuery = window.matchMedia ? window.matchMedia('(prefers-reduced-motion: reduce)') : null;
  let reducedMotion = !!reducedQuery?.matches;
  reducedQuery?.addEventListener?.('change', e => { reducedMotion = e.matches; });

  const aniso = Math.min(8, renderer.capabilities.getMaxAnisotropy());
  const env = buildEnvironment(scene, aniso);

  const fxRoot = new THREE.Group();       // links, shield hologram, rings
  const deviceRoot = new THREE.Group();
  const labelRoot = new THREE.Group();
  scene.add(deviceRoot, fxRoot, labelRoot);

  // ---- reusable scratch objects (no allocation per frame) --------------------
  const v1 = new THREE.Vector3(), v2 = new THREE.Vector3(), v3 = new THREE.Vector3();
  const sph = new THREE.Spherical();
  const box = new THREE.Box3();
  const raycaster = new THREE.Raycaster();
  raycaster.params.Line.threshold = 0.02;
  raycaster.params.Points.threshold = 0.05;
  const ndc = new THREE.Vector2();
  const wanEntry = new THREE.Vector3().fromArray(HOUSE.wanEntry);
  const WP = Array.from({ length: MAX_SEGS + 1 }, () => new THREE.Vector3());
  const CTRL = Array.from({ length: MAX_SEGS }, () => new THREE.Vector3());

  // ---- shared FX materials and geometry ----------------------------------------
  const ringGeo = new THREE.RingGeometry(0.9, 1, 64).rotateX(-Math.PI / 2);
  const ringDashGeo = (() => {
    const arcs = [];
    for (let k = 0; k < 10; k++) arcs.push(new THREE.RingGeometry(1.1, 1.18, 6, 1, k * Math.PI / 5, Math.PI / 9));
    const g = mergeGeometries(arcs).rotateX(-Math.PI / 2);
    arcs.forEach(a => a.dispose());
    return g;
  })();
  // depth: hidden behind the device and walls like a real decal (pulled towards
  // the camera so it never fights the surface it lies on); otherwise drawn on top.
  const markerMat = (hex, opacity, depth = false) => new THREE.MeshBasicMaterial({
    color: hex, transparent: true, opacity, depthTest: depth, depthWrite: false, toneMapped: false, side: THREE.DoubleSide,
    polygonOffset: depth, polygonOffsetFactor: -2, polygonOffsetUnits: -4,
  });

  // Selection ring: a crisp depth-tested ring plus a faint copy drawn on top, so
  // the selection still shows through walls without covering the device itself.
  const selectRing = new THREE.Group();
  const selectDash = [];
  for (const [opA, opB, depth] of [[0.9, 0.65, true], [0.22, 0.16, false]]) {
    const dash = new THREE.Mesh(ringDashGeo, markerMat(HEX.select, opB, depth));
    selectRing.add(new THREE.Mesh(ringGeo, markerMat(HEX.select, opA, depth)), dash);
    selectDash.push(dash);
  }
  selectRing.renderOrder = 20;
  selectRing.children.forEach(c => { c.renderOrder = 20; });
  selectRing.visible = false;
  const hoverMat = markerMat(0xffffff, 0.35);
  const hoverRing = new THREE.Mesh(ringGeo, hoverMat);
  hoverRing.renderOrder = 19;
  hoverRing.visible = false;
  fxRoot.add(selectRing, hoverRing);

  // Expanding rings for ids:block (pooled).
  const rippleGeo = new THREE.RingGeometry(0.95, 1, 72).rotateX(-Math.PI / 2);
  const blockRings = Array.from({ length: 6 }, () => {
    const m = new THREE.Mesh(rippleGeo, new THREE.MeshBasicMaterial({
      color: HEX.dropped, transparent: true, opacity: 0, depthTest: false, depthWrite: false, toneMapped: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
    }));
    m.renderOrder = 18;
    m.visible = false;
    m.userData = { t: 0, dur: 1.4, from: 0.3, to: 3 };
    fxRoot.add(m);
    return m;
  });
  let blockRingNext = 0;
  function emitRing(x, y, z, hex, from, to, dur) {
    const r = blockRings[blockRingNext];
    blockRingNext = (blockRingNext + 1) % blockRings.length;
    r.position.set(x, y, z);
    r.material.color.setHex(hex);
    Object.assign(r.userData, { t: 0, dur, from, to });
    r.visible = true;
  }

  // ---- SHIELD hologram ----------------------------------------------------------
  const shieldFx = new THREE.Group();
  const shieldColor = new THREE.Color(HEX.shield);
  const shieldTarget = new THREE.Color(HEX.shield);
  const pulseColor = new THREE.Color(HEX.dropped);
  const shellGeo = new THREE.CylinderGeometry(0.55, 1, 1, 6, 1, true).translate(0, 0.5, 0);
  const edgeSrc = new THREE.CylinderGeometry(0.55, 1, 1, 6, 1, false).translate(0, 0.5, 0);
  const edgeGeo = new THREE.EdgesGeometry(edgeSrc);
  edgeSrc.dispose();
  const shellMat = new THREE.MeshBasicMaterial({ color: HEX.shield, transparent: true, opacity: 0.13, side: THREE.DoubleSide, depthWrite: false, blending: THREE.AdditiveBlending, toneMapped: false });
  const edgeMat = new THREE.LineBasicMaterial({ color: HEX.shield, transparent: true, opacity: 0.85, depthWrite: false, blending: THREE.AdditiveBlending, toneMapped: false });
  const baseGeo = new THREE.RingGeometry(1.05, 1.22, 6, 1).rotateX(-Math.PI / 2);
  const baseMat = new THREE.MeshBasicMaterial({ color: HEX.shield, transparent: true, opacity: 0.5, side: THREE.DoubleSide, depthWrite: false, blending: THREE.AdditiveBlending, toneMapped: false });
  const scanPts = [];
  for (let k = 0; k <= 6; k++) scanPts.push(new THREE.Vector3(Math.cos(k * Math.PI / 3 + Math.PI / 6), 0, Math.sin(k * Math.PI / 3 + Math.PI / 6)));
  const scanGeo = new THREE.BufferGeometry().setFromPoints(scanPts);
  const scanMat = new THREE.LineBasicMaterial({ color: HEX.shield, transparent: true, opacity: 0.9, depthWrite: false, blending: THREE.AdditiveBlending, toneMapped: false });
  const shell = new THREE.Mesh(shellGeo, shellMat);
  const edges = new THREE.LineSegments(edgeGeo, edgeMat);
  const base = new THREE.Mesh(baseGeo, baseMat);
  const scan = new THREE.Line(scanGeo, scanMat);
  base.position.y = 0.004;
  const spin = new THREE.Group();
  spin.add(shell, edges, scan);
  // Soft halo so the sensor still reads as "the shield" from the home view.
  const haloMat = new THREE.SpriteMaterial({ map: env.glowTex, color: HEX.shield, transparent: true, opacity: 0.4, depthTest: false, depthWrite: false, blending: THREE.AdditiveBlending, toneMapped: false });
  const halo = new THREE.Sprite(haloMat);
  halo.renderOrder = 5;
  shieldFx.add(spin, base, halo);
  shieldFx.visible = false;
  fxRoot.add(shieldFx);
  let shieldPulse = 0;
  let threatLevel = 'low';
  let driftState = 'stable';

  // Router activity LED glow (the router model's own LEDs belong to models.js).
  const routerLedMat = new THREE.SpriteMaterial({ map: env.glowTex, color: HEX.routerLed, transparent: true, opacity: 0, depthWrite: false, blending: THREE.AdditiveBlending, toneMapped: false });
  const routerLed = new THREE.Sprite(routerLedMat);
  routerLed.scale.setScalar(0.32);
  routerLed.visible = false;
  fxRoot.add(routerLed);
  let routerActivity = 0;

  // TV glow: soft coloured light in front of the screen while the TV is on.
  const tvGlow = new THREE.PointLight(0x7fa8ff, 0, 5.5, 2);
  scene.add(tvGlow);
  let tvGlowLevel = 0;
  const tvGlowColor = new THREE.Color();

  // ---- packets ----------------------------------------------------------------
  const packets = createPacketSystem(fxRoot, {
    onShield(k) {
      if (k === KIND_ALERT) pulseShield(KIND_ALERT, 0.7);
    },
    onBurst(k) {
      if (k === KIND_DROP) pulseShield(KIND_DROP, 1);
    },
  });
  function pulseShield(k, amount) {
    shieldPulse = Math.max(shieldPulse, amount);
    pulseColor.setHex(k === KIND_ALERT ? HEX.alerted : HEX.dropped);
  }

  // ---- per-device records ---------------------------------------------------------
  /** @type {Map<string, object>} */
  const recs = new Map();
  const recList = [];          // same records, as an array for allocation-free loops
  let pickables = [];          // visible device groups

  const linkGeoPoints = 28;
  const LAN_OPACITY = 0.3;

  function makeLink(kindName) {
    const positions = new Float32Array(linkGeoPoints * 3);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3).setUsage(THREE.DynamicDrawUsage));
    const solid = new THREE.LineBasicMaterial({ color: HEX.link, transparent: true, opacity: LAN_OPACITY, depthWrite: false, toneMapped: false });
    const dashed = new THREE.LineDashedMaterial({ color: HEX.linkOffline, transparent: true, opacity: 0.3, dashSize: 0.14, gapSize: 0.12, depthWrite: false, toneMapped: false });
    const line = new THREE.Line(geo, solid);
    line.frustumCulled = false;
    line.renderOrder = 3;
    fxRoot.add(line);
    return {
      kind: kindName,            // 'lan' | 'inline' | 'fibre'
      line, geo, positions, solid, dashed,
      a: new THREE.Vector3(Infinity, 0, 0), b: new THREE.Vector3(Infinity, 0, 0),
      mode: 'normal',            // 'normal' | 'alert' | 'blocked' | 'offline' | 'hidden'
      pulse: 0,
      pulseColor: new THREE.Color(HEX.allowed),
      baseColor: new THREE.Color(HEX.link),
      baseOpacity: LAN_OPACITY,
      dirty: true,
    };
  }

  function disposeLink(link) {
    fxRoot.remove(link.line);
    link.geo.dispose();
    link.solid.dispose();
    link.dashed.dispose();
  }

  /** Rewrite a link's curve if either end moved. */
  function updateLinkGeometry(link, a, b, fibre) {
    if (link.a.distanceToSquared(a) < 1e-6 && link.b.distanceToSquared(b) < 1e-6) return;
    link.a.copy(a);
    link.b.copy(b);
    if (fibre) fibreControl(a, b, v3); else arcControl(a, b, v3);
    const p = link.positions;
    for (let k = 0; k < linkGeoPoints; k++) {
      bezierAt(a, v3, b, k / (linkGeoPoints - 1), v2);
      p[k * 3] = v2.x; p[k * 3 + 1] = v2.y; p[k * 3 + 2] = v2.z;
    }
    link.geo.attributes.position.needsUpdate = true;
    if (link.line.material === link.dashed) link.line.computeLineDistances();
  }

  function setLinkMode(link, mode) {
    if (link.mode === mode) return;
    link.mode = mode;
    link.dirty = true;
    link.line.visible = mode !== 'hidden';
    const wantDashed = mode === 'offline';
    const mat = wantDashed ? link.dashed : link.solid;
    if (link.line.material !== mat) {
      link.line.material = mat;
      if (wantDashed) link.line.computeLineDistances();
    }
    const kindBase = link.kind === 'inline' ? HEX.shield : link.kind === 'fibre' ? HEX.fibre : HEX.link;
    const kindOpacity = link.kind === 'inline' ? 0.75 : link.kind === 'fibre' ? 0.5 : LAN_OPACITY;
    switch (mode) {
      case 'blocked': link.baseColor.setHex(HEX.dropped); link.baseOpacity = 0.7; break;
      case 'alert': link.baseColor.setHex(HEX.alerted); link.baseOpacity = 0.5; break;
      case 'offline': link.baseColor.setHex(HEX.linkOffline); link.baseOpacity = 0.35; break;
      default: link.baseColor.setHex(kindBase); link.baseOpacity = kindOpacity;
    }
  }

  function pulseLink(link, hex, amount = 1) {
    if (!link || link.mode === 'hidden') return;
    link.pulse = Math.max(link.pulse, amount);
    link.pulseColor.setHex(hex);
  }

  function effectiveStatus(d) {
    if (d.blocked || state.ids.blocked.has(d.ip)) return 'blocked';
    if (d.status === 'blocked') return d.online === false ? 'offline' : 'ok';
    return d.status || 'ok';
  }

  function isPresent(d) {
    return d.role !== 'unknown' || !!d.props?.plugged;
  }

  function makeLabel(rec, device) {
    const el = document.createElement('div');
    el.className = 'sc-label is-ok';
    const inner = document.createElement('div');
    const dot = document.createElement('span');
    dot.className = 'sc-dot';
    const name = document.createElement('span');
    name.textContent = device.name;
    const st = document.createElement('span');
    st.className = 'sc-state';
    inner.append(dot, name, st);
    el.appendChild(inner);
    el.addEventListener('click', e => { e.stopPropagation(); select(rec.id); });
    el.addEventListener('dblclick', e => { e.stopPropagation(); focus(rec.id); });
    el.addEventListener('pointerenter', () => setHover(rec.id));
    el.addEventListener('pointerleave', () => { if (hoverId === rec.id) setHover(null); });
    const obj = new CSS2DObject(el);
    obj.center.set(0.5, 1);
    obj.visible = false;
    labelRoot.add(obj);
    rec.label = obj;
    rec.labelEl = el;
    rec.labelState = st;
  }

  function buildDevice(device) {
    let model;
    try {
      model = createDeviceModel(device);
    } catch (err) {
      console.error(`[scene] createDeviceModel failed for ${device.id}`, err);
      model = fallbackModel();
    }
    const group = model.group;
    group.position.fromArray(device.pos);
    group.rotation.y = device.rotY || 0;
    group.userData.deviceId = device.id;
    deviceRoot.add(group);
    group.updateMatrixWorld(true);
    box.setFromObject(group);
    box.getSize(v1);
    const rotY = device.rotY || 0;
    const mount = device.role === 'cloud' ? 'cloud'
      : WALL_TYPES.has(device.type) ? 'wall'
        : CEILING_TYPES.has(device.type) ? 'ceiling' : 'floor';
    let radius;
    if (mount === 'wall') {
      // Ring stands upright around the device, sized to its face.
      const across = Math.abs(Math.cos(rotY)) * v1.x + Math.abs(Math.sin(rotY)) * v1.z;
      // Circumscribed, so the ring frames the face instead of crossing it.
      radius = clamp(Math.hypot(across, v1.y) / 2 + 0.05, 0.16, 1.0);
    } else {
      radius = clamp(Math.max(v1.x, v1.z) / 2 + 0.14, 0.34, 1.9);
    }
    const rec = {
      id: device.id,
      device,
      model,
      group,
      mount,
      rotY,
      port: new THREE.Vector3(),
      // Wall devices: centre of the upright ring, nudged off the wall.
      center: box.getCenter(new THREE.Vector3()).addScaledVector(v2.set(Math.sin(rotY), 0, Math.cos(rotY)), 0.04),
      radius,
      topOffset: Math.max(0.12, box.max.y - group.position.y),
      // Floor rings: on the surface the device stands on, on the floor below
      // ceiling devices, under the cloud's belly.
      ringY: mount === 'cloud' ? box.min.y : mount === 'floor' ? device.pos[1] + 0.015 : 0.015,
      status: 'ok',
      present: true,
      link: null,
      light: null,
      lightLevel: 0,
      lightTarget: 0,
      label: null,
      labelEl: null,
      labelState: null,
    };
    portWorld(rec);
    if (device.id !== 'router' && device.role !== 'cloud') {
      rec.link = makeLink(device.id === 'shield' ? 'inline' : 'lan');
    }
    if (device.role === 'cloud') rec.link = makeLink('fibre');
    if (device.type === 'light') {
      rec.light = new THREE.PointLight(0xffd8a8, 0, 9.5, 2);
      rec.light.position.set(device.pos[0], device.pos[1] - 0.3, device.pos[2]);
      scene.add(rec.light);
    }
    makeLabel(rec, device);
    recs.set(device.id, rec);
    recList.push(rec);
    return rec;
  }

  function portWorld(rec) {
    const p = rec.model.port;
    if (p) rec.port.copy(p).applyMatrix4(rec.group.matrixWorld);
    else rec.port.setFromMatrixPosition(rec.group.matrixWorld);
    return rec.port;
  }

  function refreshPickables() {
    pickables = recList.filter(r => r.present).map(r => r.group);
  }

  /** Apply a device's current state to its model, label, link and light. */
  function applyDevice(rec) {
    const d = state.devices.get(rec.id);
    if (!d) return;
    rec.device = d;
    const present = isPresent(d);
    if (present !== rec.present) {
      rec.present = present;
      rec.group.visible = present;
      refreshPickables();
      if (!present && hoverId === rec.id) setHover(null);
    }
    const status = effectiveStatus(d);
    if (status !== rec.status) {
      rec.status = status;
      try { rec.model.setStatus(status); } catch (err) { console.error('[scene] setStatus failed', err); }
    }
    refreshLabel(rec);
    refreshLink(rec);
    if (rec.light) {
      const on = d.props.power !== false;
      rec.lightTarget = on ? 26 * clamp(Number(d.props.brightness ?? 100) / 100, 0, 1) : 0;
      if (typeof d.props.color === 'string') rec.light.color.set(d.props.color);
    }
    if (rec.id === 'router') {
      for (const r of recList) { refreshLink(r); refreshLabel(r); }
    }
  }

  function refreshLink(rec) {
    if (!rec.link) return;
    const router = recs.get('router');
    const routerDown = router && router.status === 'offline' && rec.link.kind !== 'fibre';
    let mode = 'normal';
    if (!rec.present) mode = 'hidden';
    else if (rec.status === 'blocked') mode = 'blocked';
    else if (rec.status === 'offline' || routerDown) mode = 'offline';
    else if (rec.status === 'alert') mode = 'alert';
    setLinkMode(rec.link, mode);
  }

  function routerIsDown() {
    const router = recs.get('router');
    return !!router && router.status === 'offline';
  }

  /** Short state text after the name: "off" for a switched-off device, otherwise the status. */
  function statusText(rec) {
    if (rec.status === 'ok') return '';
    const p = rec.device.props || {};
    if (rec.status === 'offline' && !routerIsDown()) {
      if (p.power === false) return '· off';
      if (p.wifi === false) return '· Wi-Fi off';
    }
    if (rec.status === 'offline' && rec.id === 'router') return '· rebooting';
    return `· ${STATUS_TEXT[rec.status] || rec.status}`;
  }

  function refreshLabel(rec) {
    const el = rec.labelEl;
    const selected = state.selectedId === rec.id;
    // While the router reboots every LAN device is offline; only the router's
    // own label (plus hovered/selected ones) is shown, the dashed links say the rest.
    const outage = rec.status === 'offline' && rec.id !== 'router' && routerIsDown();
    const show = rec.present && (selected || hoverId === rec.id || (rec.status !== 'ok' && !outage));
    rec.label.visible = show;
    const cls = `sc-label is-${rec.status}${selected ? ' is-selected' : ''}`;
    if (el.className !== cls) el.className = cls;
    const text = statusText(rec);
    if (rec.labelState.textContent !== text) rec.labelState.textContent = text;
  }

  function buildAll() {
    for (const d of state.devices.values()) buildDevice(d);
    for (const rec of recList) {
      rec.status = '';           // force setStatus on the first apply
      rec.present = true;
      applyDevice(rec);
    }
    refreshPickables();
    placeShield();
  }

  function clearAll() {
    packets.clear();
    for (const rec of recList) {
      deviceRoot.remove(rec.group);
      try { rec.model.dispose(); } catch (err) { console.error('[scene] dispose failed', err); }
      labelRoot.remove(rec.label);
      if (rec.link) disposeLink(rec.link);
      if (rec.light) { scene.remove(rec.light); rec.light.dispose(); }
    }
    recs.clear();
    recList.length = 0;
    pickables = [];
    hoverId = null;
    selectRing.visible = false;
    hoverRing.visible = false;
  }

  /** Size the hologram to the shield model and park it on top of the router shelf. */
  function placeShield() {
    const rec = recs.get('shield');
    shieldFx.visible = !!rec;
    if (!rec) return;
    box.setFromObject(rec.group);
    box.getSize(v1);
    // The dome covers the sensor and the router it guards, so it reads as
    // "the inline shield" from the home view, not as a small ornament.
    let r = Math.max(v1.x, v1.z) / 2 + 0.07;
    let cx = rec.group.position.x, cz = rec.group.position.z;
    const router = recs.get('router');
    if (router) {
      const dx = router.group.position.x - cx, dz = router.group.position.z - cz;
      cx += dx / 2;
      cz += dz / 2;
      r = Math.max(r, Math.hypot(dx, dz) / 2 + 0.24);
    }
    r = clamp(r, 0.3, 0.6);
    const h = clamp(r * 1.9, 0.6, 1.1);
    shieldFx.position.set(cx, box.min.y, cz);
    spin.scale.set(r, h, r);
    base.scale.set(r, 1, r);
    shieldFx.userData.h = h;
    shieldFx.userData.r = r;
    shieldFx.userData.halo = clamp(r * 4.6, 1.6, 2.6);
    halo.position.y = h * 0.45;
    halo.scale.setScalar(shieldFx.userData.halo);
  }

  // ---- packet routing ------------------------------------------------------------
  // The path under construction lives in WP/CTRL; pathN counts its waypoints and
  // pathShield is the waypoint index of the shield (-1 if the path stops before it).
  let pathN = 0;
  let pathShield = -1;

  /** Append a waypoint (skipping near-duplicates). fibre: the segment follows the ISP fibre. */
  function pushWaypoint(p, fibre) {
    if (pathN > 0 && WP[pathN - 1].distanceToSquared(p) < 0.0025) return;
    if (pathN > MAX_SEGS) return;
    WP[pathN].copy(p);
    if (pathN > 0) {
      const prev = WP[pathN - 1];
      if (fibre) {
        // fibreControl expects (cloud end, house end): the cloud is the higher one.
        if (prev.y > p.y) fibreControl(prev, p, CTRL[pathN - 1]);
        else fibreControl(p, prev, CTRL[pathN - 1]);
      } else {
        arcControl(prev, p, CTRL[pathN - 1]);
      }
    }
    pathN++;
  }

  /**
   * Build waypoints src -> router -> shield -> dst into WP/CTRL. stopAt is
   * 'router' (firewall drop), 'shield' (IDS drop) or null (delivered).
   * Ports are recomputed here because some devices move (the vacuum).
   */
  function buildPath(srcId, dstId, stopAt) {
    pathN = 0;
    pathShield = -1;
    const router = recs.get('router');
    const shield = recs.get('shield');
    const src = recs.get(srcId);
    const dst = recs.get(dstId);
    if (!router || !src || !dst) return;
    pushWaypoint(portWorld(src), false);
    if (src.device.role === 'cloud') pushWaypoint(wanEntry, true);
    pushWaypoint(portWorld(router), false);
    if (stopAt === 'router') return;
    if (shield) {
      pushWaypoint(portWorld(shield), false);
      pathShield = pathN - 1;
    }
    if (stopAt === 'shield') return;
    if (dst.device.role === 'cloud') {
      pushWaypoint(wanEntry, false);
      pushWaypoint(portWorld(dst), true);
    } else {
      pushWaypoint(portWorld(dst), false);
    }
  }

  function launch(pkt, k, stopAt) {
    if (!pkt) return;
    routerActivity = Math.min(1, routerActivity + 0.18);
    const cap = reducedMotion ? MAX_PACKETS_REDUCED : MAX_PACKETS;
    buildPath(pkt.src, pkt.dst, stopAt);
    if (pathN < 2) return;
    packets.spawn(k, WP, CTRL, pathN, pathShield, stopAt !== null, cap);
    // Links light up for every packet, including ones the animation sampled out.
    const hex = k === KIND_ALLOW ? HEX.allowed : k === KIND_ALERT ? HEX.alerted : HEX.dropped;
    const src = recs.get(pkt.src);
    const dst = recs.get(pkt.dst);
    pulseLink(src?.link, hex);
    if (stopAt === null) pulseLink(dst?.link, hex, 0.8);
    if (src?.device.role === 'cloud' || dst?.device.role === 'cloud') {
      pulseLink(recs.get('cloud')?.link, hex, 0.9);
    }
    if (stopAt !== 'router') pulseLink(recs.get('shield')?.link, hex, 0.6);
  }

  // ---- picking and hover -----------------------------------------------------------
  let hoverId = null;
  let hoverDirty = false;
  let dragging = false;
  let downX = 0, downY = 0, downOk = false;

  function setNdc(clientX, clientY) {
    const r = canvas.getBoundingClientRect();
    ndc.set(((clientX - r.left) / r.width) * 2 - 1, -((clientY - r.top) / r.height) * 2 + 1);
  }

  /** Device id under the pointer (visible parts only), or null. */
  function pick() {
    if (!pickables.length) return null;
    raycaster.setFromCamera(ndc, camera);
    const hits = raycaster.intersectObjects(pickables, true);
    for (const hit of hits) {
      let o = hit.object, id = null, visible = true;
      while (o) {
        if (!o.visible) { visible = false; break; }
        if (o.userData && o.userData.deviceId) { id = o.userData.deviceId; break; }
        o = o.parent;
      }
      if (visible && id) return id;
    }
    return null;
  }

  function setHover(id) {
    if (hoverId === id) return;
    const prev = hoverId;
    hoverId = id;
    canvas.style.cursor = id ? 'pointer' : '';
    if (prev && recs.has(prev)) refreshLabel(recs.get(prev));
    if (id && recs.has(id)) refreshLabel(recs.get(id));
  }

  function select(id) {
    if (id !== null && !recs.has(id)) return;
    if (state.selectedId === id) return;
    state.selectedId = id;
    bus.emit('device:select', { id });
  }

  function onPointerDown(e) {
    downX = e.clientX;
    downY = e.clientY;
    downOk = e.button === 0;
    dragging = true;
  }
  function onPointerUp(e) {
    dragging = false;
    if (!downOk || e.button !== 0) return;
    downOk = false;
    if (Math.hypot(e.clientX - downX, e.clientY - downY) >= CLICK_SLOP_PX) return;
    setNdc(e.clientX, e.clientY);
    const id = pick();
    if (id) select(id);
    else if (state.selectedId !== null) select(null);
  }
  function onPointerMove(e) {
    if (e.buttons) return;                    // dragging: no hover work
    setNdc(e.clientX, e.clientY);
    hoverDirty = true;
  }
  function onPointerLeave() {
    hoverDirty = false;
    dragging = false;
    setHover(null);
  }
  function onPointerCancel() {
    dragging = false;
    downOk = false;
  }
  function onDblClick(e) {
    setNdc(e.clientX, e.clientY);
    const id = pick();
    if (id) focus(id);
    else goHome();                          // double-click empty space: back to the overview
  }
  canvas.addEventListener('pointerdown', onPointerDown);
  canvas.addEventListener('pointerup', onPointerUp);
  canvas.addEventListener('pointermove', onPointerMove);
  canvas.addEventListener('pointerleave', onPointerLeave);
  canvas.addEventListener('pointercancel', onPointerCancel);
  canvas.addEventListener('dblclick', onDblClick);

  // ---- camera: home view, intro and focus flights --------------------------------------
  const fly = {
    active: false, t: 0, dur: FOCUS_SEC,
    fromPos: new THREE.Vector3(), toPos: new THREE.Vector3(),
    fromTarget: new THREE.Vector3(), toTarget: new THREE.Vector3(),
  };
  let userMoved = false;

  controls.addEventListener('start', () => {
    fly.active = false;
    userMoved = true;
  });

  /**
   * Home view: the fixed viewing direction, at the smallest distance where the
   * whole house (plan, walls and porch) fits inside the part of the canvas the
   * HUD panels leave uncovered.
   */
  function homeView(outPos, outTarget) {
    outTarget.copy(HOME_TARGET);
    sph.set(1, HOME_PHI, HOME_THETA);
    v3.setFromSpherical(sph);                    // unit vector from the target to the camera
    fitCam.copy(camera);
    const m = 0.035;                             // margin, as a fraction of the NDC range
    const x0 = (safe.l / width) * 2 - 1 + m, x1 = (safe.r / width) * 2 - 1 - m;
    const y0 = 1 - (safe.b / height) * 2 + m, y1 = 1 - (safe.t / height) * 2 - m;
    const fits = dist => {
      fitCam.position.copy(v3).multiplyScalar(dist).add(outTarget);
      fitCam.lookAt(outTarget);
      fitCam.updateMatrixWorld(true);
      for (const p of HOME_FIT_POINTS) {
        fitPt.copy(p).project(fitCam);
        if (fitPt.x < x0 || fitPt.x > x1 || fitPt.y < y0 || fitPt.y > y1 || fitPt.z > 1) return false;
      }
      return true;
    };
    let lo = 6, hi = controls.maxDistance;
    if (fits(lo)) hi = lo;
    else {
      for (let i = 0; i < 24; i++) {
        const mid = (lo + hi) / 2;
        if (fits(mid)) hi = mid; else lo = mid;
      }
    }
    outPos.copy(v3).multiplyScalar(hi).add(outTarget);
  }

  // ---- uncovered area: the HUD floats over the full-window canvas on desktops ----------
  // The projection centre is moved to the middle of the area the panels leave
  // free, so the home view and focus flights frame things where they can be seen.
  const fitCam = new THREE.PerspectiveCamera();
  const fitPt = new THREE.Vector3();
  const HOME_FIT_POINTS = [];
  {
    const b = HOUSE.bounds;
    for (const x of [b.minX - 0.2, b.maxX + 0.2]) {
      for (const z of [b.minZ - 0.2, b.maxZ + 1.0]) {       // + the front porch
        for (const y of [0, HOUSE.wallHeight]) HOME_FIT_POINTS.push(new THREE.Vector3(x, y, z));
      }
    }
  }
  const safe = { l: 0, r: 1, t: 0, b: 1 };       // canvas pixels
  const OCCLUDERS = ['topbar', 'panel-home', 'panel-shield', 'drawer'];

  /** Measures the canvas area not covered by the HUD panels (all of it on narrow layouts). */
  function measureSafeArea() {
    safe.l = 0; safe.r = width; safe.t = 0; safe.b = height;
    const c = container.getBoundingClientRect();
    const gap = 8;
    const rectOf = id => {
      const el = document.getElementById(id);
      if (!el) return null;
      const q = el.getBoundingClientRect();
      if (q.width < 1 || q.height < 1) return null;
      if (q.right <= c.left || q.left >= c.right || q.bottom <= c.top || q.top >= c.bottom) return null;
      return q;
    };
    const top = rectOf('topbar');
    if (top && top.bottom - c.top < c.height * 0.3) safe.t = Math.max(safe.t, top.bottom - c.top + gap);
    // Side panels only count while they run down the side (not when collapsed to a header).
    const left = rectOf('panel-home');
    if (left && left.height > c.height * 0.5 && left.left - c.left < c.width * 0.2) safe.l = Math.max(safe.l, left.right - c.left + gap);
    const right = rectOf('panel-shield');
    if (right && right.height > c.height * 0.5 && c.right - right.right < c.width * 0.2) safe.r = Math.min(safe.r, right.left - c.left - gap);
    const bottom = rectOf('drawer');
    if (bottom && c.bottom - bottom.bottom < c.height * 0.2) safe.b = Math.min(safe.b, bottom.top - c.top - gap);
    // Fall back to the whole canvas when the panels leave too little room.
    if (safe.r - safe.l < width * 0.3) { safe.l = 0; safe.r = width; }
    if (safe.b - safe.t < height * 0.3) { safe.t = 0; safe.b = height; }
  }

  function applySafeArea() {
    measureSafeArea();
    const dx = (safe.l + safe.r) / 2 - width / 2;
    const dy = (safe.t + safe.b) / 2 - height / 2;
    if (Math.abs(dx) < 0.5 && Math.abs(dy) < 0.5) camera.clearViewOffset();
    else camera.setViewOffset(width, height, -dx, -dy, width, height);
    camera.updateProjectionMatrix();
  }

  /** Back to the home overview (after a reset, or a double-click on empty space). */
  function goHome() {
    homeView(v1, v2);
    userMoved = false;
    if (reducedMotion) {
      fly.active = false;
      camera.position.copy(v1);
      controls.target.copy(v2);
    } else {
      flyTo(v1, v2, FOCUS_SEC * 1.4);
    }
  }

  function flyTo(pos, target, dur) {
    fly.fromPos.copy(camera.position);
    fly.fromTarget.copy(controls.target);
    fly.toPos.copy(pos);
    fly.toTarget.copy(target);
    fly.t = 0;
    fly.dur = dur;
    fly.active = true;
  }

  // Candidate viewing angles for focus(): [polar, azimuth offset]. The first
  // one keeps the current direction; the rest swing round or look down.
  const FOCUS_CANDIDATES = [[0.9, 0], [0.62, 0], [0.9, 0.9], [0.9, -0.9], [0.62, 1.8], [0.62, -1.8], [0.62, Math.PI], [0.28, 0]];
  // Wall devices: start square-on to the device's face, then swing a little.
  const FOCUS_WALL = [[1.12, 0], [0.9, 0], [1.12, 0.5], [1.12, -0.5], [0.7, 0], [0.9, 0.9], [0.9, -0.9], [0.5, 0]];
  // Front-facing appliances: a little above eye level, in front of the door.
  const FOCUS_FRONT = [[0.95, 0], [0.75, 0], [0.95, 0.5], [0.95, -0.5], [0.6, 0.9], [0.6, -0.9], [0.45, 0], ...FOCUS_CANDIDATES];
  const focusRay = new THREE.Raycaster();

  /**
   * Is the straight line between the target and the camera position free of
   * walls, furniture and trees? Tested in both directions because a ray that
   * starts inside a wall does not hit that wall's (back-facing) faces.
   */
  function clearView(target, camPos) {
    v3.copy(camPos).sub(target);
    const len = v3.length();
    if (len < 1e-3) return false;
    v3.divideScalar(len);
    focusRay.set(target, v3);
    focusRay.near = 0.05;
    focusRay.far = len;
    if (focusRay.intersectObjects(env.occluders, false).length) return false;
    if (focusRay.intersectObjects(focusBlockers, false).length) return false;
    focusRay.set(camPos, v3.negate());
    focusRay.near = 0;
    focusRay.far = Math.max(0, len - 0.03);
    return focusRay.intersectObjects(env.occluders, false).length === 0;
  }

  // Other devices' meshes can hide the focused one too (a pendant lamp in
  // front of the TV), so focus() also steers around them.
  const focusBlockers = [];
  function collectBlockers(except) {
    focusBlockers.length = 0;
    for (const r of recList) {
      if (r === except || !r.present || r.device.role === 'cloud') continue;
      r.group.traverseVisible(o => { if (o.isMesh) focusBlockers.push(o); });
    }
  }

  function focus(id) {
    const rec = recs.get(id);
    if (!rec || !rec.present) return;
    box.setFromObject(rec.group);
    box.getCenter(v1);
    box.getSize(v2);
    const size = Math.max(v2.x, v2.y, v2.z);
    const cloud = rec.device.role === 'cloud';
    const dist = cloud ? 14 : clamp(size * 5 + 2.2, 3.2, 7);
    const target = v1.clone();
    const camPos = new THREE.Vector3();
    if (cloud) {
      // The cloud floats behind the house: look at it from the garden side.
      sph.set(dist, 1.18, HOME_THETA);
      camPos.setFromSpherical(sph).add(target);
    } else {
      // Wall devices are approached from the side they face; everything else
      // from the current viewing direction, swinging round if that is blocked.
      collectBlockers(rec);
      const wall = rec.mount === 'wall';
      const front = !wall && FRONT_TYPES.has(rec.device.type);
      sph.setFromVector3(v3.copy(camera.position).sub(controls.target));
      const baseTheta = wall || front ? rec.rotY : sph.theta;
      // Sight lines from the centre and from points towards the edges of the
      // device, so nothing hides part of it (a lamp across the bottom of the TV).
      const samples = [target];
      for (const [x, y, z] of [[0, 0.35, 0], [0, -0.35, 0], [0.35, 0, 0], [-0.35, 0, 0], [0, 0, 0.35], [0, 0, -0.35]]) {
        samples.push(new THREE.Vector3(x * v2.x, y * v2.y, z * v2.z).add(target));
      }
      const visible = p => samples.every(sp => clearView(sp, p));
      let found = false;
      for (const [phi, dTheta] of wall ? FOCUS_WALL : front ? FOCUS_FRONT : FOCUS_CANDIDATES) {
        sph.set(dist, phi, baseTheta + dTheta);
        camPos.setFromSpherical(sph).add(target);
        if (camPos.y > 0.4 && visible(camPos)) { found = true; break; }
      }
      // Nothing fully clear: settle for a clear line to the centre.
      for (const [phi, dTheta] of found ? [] : wall ? FOCUS_WALL : front ? FOCUS_FRONT : FOCUS_CANDIDATES) {
        sph.set(dist, phi, baseTheta + dTheta);
        camPos.setFromSpherical(sph).add(target);
        if (camPos.y > 0.4 && clearView(target, camPos)) { found = true; break; }
      }
      if (!found) camPos.setFromSpherical(sph.set(dist, 0.2, baseTheta)).add(target);
      focusBlockers.length = 0;
    }
    userMoved = true;
    if (reducedMotion) {
      camera.position.copy(camPos);
      controls.target.copy(target);
      fly.active = false;
    } else {
      flyTo(camPos, target, FOCUS_SEC);
    }
  }

  // ---- resize and adaptive resolution ------------------------------------------------
  let width = 0, height = 0;
  /** smooth: glide to the re-framed home view (panels opening/closing) instead of jumping. */
  function resize(force = false, smooth = false) {
    const w = Math.max(1, Math.floor(container.clientWidth));
    const h = Math.max(1, Math.floor(container.clientHeight));
    if (w === width && h === height && !force) return;
    width = w;
    height = h;
    renderer.setSize(w, h);
    labelRenderer.setSize(w, h);
    camera.aspect = w / h;
    applySafeArea();
    updatePointScale();
    if (!userMoved) {
      // Still on the home view (or its intro): re-frame for the new aspect.
      homeView(v1, v2);
      if (fly.active) {
        fly.toPos.copy(v1);
        fly.toTarget.copy(v2);
      } else if (smooth && !reducedMotion) {
        flyTo(v1, v2, 0.45);
      } else {
        camera.position.copy(v1);
        controls.target.copy(v2);
      }
    }
  }
  function updatePointScale() {
    const px = renderer.getPixelRatio() * height / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2));
    packets.setScale(px, renderer.getPixelRatio());
  }
  const resizeObserver = new ResizeObserver(() => resize());
  resizeObserver.observe(container);
  // Panels collapsing or the drawer opening change the uncovered area, not the canvas.
  const safeKey = () => `${safe.l}|${safe.r}|${safe.t}|${safe.b}`;
  const occluderObserver = new ResizeObserver(() => {
    if (width <= 1 || height <= 1) return;
    const before = safeKey();
    measureSafeArea();
    if (safeKey() !== before) resize(true, true);
  });
  for (const id of OCCLUDERS) {
    const el = document.getElementById(id);
    if (el) occluderObserver.observe(el);
  }

  let frameAcc = 0, frameCount = 0;
  function adaptResolution(realDtMs) {
    frameAcc += realDtMs;
    frameCount++;
    if (frameCount < 120) return;
    const avg = frameAcc / frameCount;
    frameAcc = 0;
    frameCount = 0;
    if (avg > 26 && dpr > 1) {
      dpr = Math.max(1, dpr - 0.25);
      renderer.setPixelRatio(dpr);
      renderer.setSize(width, height);
      updatePointScale();
    }
  }

  let contextLost = false;
  canvas.addEventListener('webglcontextlost', e => { e.preventDefault(); contextLost = true; });
  canvas.addEventListener('webglcontextrestored', () => { contextLost = false; });

  // ---- events ----------------------------------------------------------------------------
  bus.on('packet:verdict', ({ pkt, verdict } = {}) => {
    const action = verdict?.action;
    if (action === 'drop') launch(pkt, KIND_DROP, 'shield');
    else launch(pkt, action === 'alert' ? KIND_ALERT : KIND_ALLOW, null);
  });
  bus.on('packet:dropped', ({ pkt, reason } = {}) => {
    if (reason === 'firewall' || reason === 'router-offline') launch(pkt, KIND_DROP, 'router');
  });
  bus.on('device:update', ({ id } = {}) => {
    const rec = recs.get(id);
    if (rec) applyDevice(rec);
  });
  const onBlockChange = blocked => ({ ip, deviceId } = {}) => {
    const rec = recs.get(deviceId || state.byIp.get(ip));
    if (rec) applyDevice(rec);
    if (!blocked) return;
    // A new firewall rule: a saffron ripple from the sensor, a red one at the device.
    const sh = recs.get('shield');
    if (sh) emitRing(sh.group.position.x, shieldFx.position.y + 0.01, sh.group.position.z, HEX.shield, 0.3, 3.2, 1.3);
    pulseShield(KIND_DROP, 1);
    if (rec && rec.present) {
      emitRing(rec.port.x, rec.ringY + 0.01, rec.port.z, HEX.dropped, rec.radius * 0.6, rec.radius * 3.2, 1.1);
      pulseLink(rec.link, HEX.dropped, 1);
    }
  };
  bus.on('ids:block', onBlockChange(true));
  bus.on('ids:unblock', onBlockChange(false));
  bus.on('ids:alert', a => {
    const rec = recs.get(a?.srcId) || recs.get(state.byIp.get(a?.srcIp));
    if (rec) pulseLink(rec.link, HEX.alerted, 1);
  });
  bus.on('ids:metrics', m => {
    if (!m) return;
    threatLevel = m.threat || 'low';
    shieldTarget.setHex(threatLevel === 'high' || threatLevel === 'critical' ? HEX.threat
      : threatLevel === 'elevated' ? 0xf78f2e : HEX.shield);
  });
  bus.on('ids:drift', d => { driftState = d?.state || 'stable'; });
  bus.on('device:select', () => { for (const rec of recList) refreshLabel(rec); });
  bus.on('scene:focus', p => { if (p?.id) focus(p.id); });
  bus.on('sim:after-reset', () => {
    clearAll();
    buildAll();
    goHome();
    shieldPulse = 0;
    threatLevel = 'low';
    driftState = 'stable';
    shieldTarget.setHex(HEX.shield);
  });

  // ---- build and first frame ------------------------------------------------------------
  buildAll();
  resize();
  homeView(fly.toPos, fly.toTarget);
  if (reducedMotion) {
    camera.position.copy(fly.toPos);
    controls.target.copy(fly.toTarget);
  } else {
    // Intro: swing in from a high, wide angle.
    controls.target.copy(fly.toTarget);
    sph.setFromVector3(v1.copy(fly.toPos).sub(fly.toTarget));
    sph.radius *= 1.7;
    sph.theta -= 0.9;
    sph.phi *= 0.55;
    camera.position.setFromSpherical(sph).add(fly.toTarget);
    flyTo(fly.toPos, fly.toTarget, INTRO_SEC);
  }
  controls.update();

  // ---- per-frame update ---------------------------------------------------------------------
  let realT = 0;
  let simT = 0;

  function updateCamera(dt) {
    if (fly.active) {
      fly.t += dt;
      const k = easeInOutCubic(clamp(fly.t / fly.dur, 0, 1));
      camera.position.lerpVectors(fly.fromPos, fly.toPos, k);
      controls.target.lerpVectors(fly.fromTarget, fly.toTarget, k);
      if (fly.t >= fly.dur) fly.active = false;
    }
    const t = controls.target;
    t.x = clamp(t.x, TARGET_BOUNDS.minX, TARGET_BOUNDS.maxX);
    t.y = clamp(t.y, TARGET_BOUNDS.minY, TARGET_BOUNDS.maxY);
    t.z = clamp(t.z, TARGET_BOUNDS.minZ, TARGET_BOUNDS.maxZ);
    controls.update();
  }

  function updateDevices(realDt, simDt) {
    const router = recs.get('router');
    for (let i = 0; i < recList.length; i++) {
      const rec = recList[i];
      if (!rec.present) continue;
      const d = state.devices.get(rec.id) || rec.device;
      try { rec.model.update(d, simDt, simT); } catch (err) {
        if (!rec.updateFailed) console.error(`[scene] model.update failed for ${rec.id}`, err);
        rec.updateFailed = true;
      }
      portWorld(rec);
      if (rec.light) {
        rec.lightLevel += (rec.lightTarget - rec.lightLevel) * Math.min(1, realDt * 6);
        rec.light.intensity = rec.lightLevel;
        rec.light.visible = rec.lightLevel > 0.02;
      }
      const link = rec.link;
      if (link && link.mode !== 'hidden' && router) {
        if (link.kind === 'fibre') updateLinkGeometry(link, rec.port, wanEntry, true);
        else updateLinkGeometry(link, rec.port, router.port, false);
      }
    }
  }

  function updateLinks(dt) {
    const decay = Math.exp(-dt * 3.2);
    for (let i = 0; i < recList.length; i++) {
      const link = recList[i].link;
      if (!link || link.mode === 'hidden') continue;
      if (link.pulse <= 0.002 && !link.dirty) continue;
      link.pulse = link.pulse > 0.002 ? link.pulse * decay : 0;
      const p = link.pulse;
      const mat = link.line.material;
      mat.color.copy(link.baseColor).lerp(link.pulseColor, Math.min(1, p * 1.2));
      mat.opacity = link.baseOpacity + (0.95 - link.baseOpacity) * p;
      link.dirty = false;
    }
  }

  function updateShield(dt) {
    if (!shieldFx.visible) return;
    shieldColor.lerp(shieldTarget, Math.min(1, dt * 2.5));
    shieldPulse *= Math.exp(-dt * 3.5);
    const p = shieldPulse;
    const hot = threatLevel === 'high' || threatLevel === 'critical';
    spin.rotation.y += dt * (hot ? 1.1 : 0.45);
    const s = 1 + 0.14 * p;
    spin.scale.x = spin.scale.z = base.scale.x * s;
    // Pulse colour flashes over the base colour.
    edgeMat.color.copy(shieldColor).lerp(pulseColor, p * 0.85);
    shellMat.color.copy(edgeMat.color);
    if (driftState === 'stable') scanMat.color.copy(shieldColor);
    else scanMat.color.setHex(HEX.drift);
    baseMat.color.copy(edgeMat.color);
    const breathe = reducedMotion ? 0 : 0.04 * Math.sin(realT * 2.2);
    shellMat.opacity = 0.11 + breathe + 0.32 * p + (hot ? 0.06 : 0);
    edgeMat.opacity = 0.7 + 0.3 * p;
    baseMat.opacity = 0.35 + 0.5 * p;
    haloMat.color.copy(edgeMat.color);
    haloMat.opacity = 0.3 + breathe * 2 + 0.4 * p + (hot ? 0.1 : 0);
    halo.scale.setScalar(shieldFx.userData.halo * (1 + 0.3 * p));
    // Scanning band: rises through the dome; faster while drifting.
    const speed = driftState === 'drift' ? 1.6 : driftState === 'warning' ? 1.0 : 0.55;
    const f = reducedMotion ? 0.5 : (realT * speed) % 1;
    scan.position.y = f;
    const w = 1 - 0.45 * f;                    // the dome tapers from 1 to 0.55
    scan.scale.set(w, 1, w);
    scanMat.opacity = reducedMotion ? 0.6 : 0.9 * Math.sin(f * Math.PI);
  }

  function updateRouterLed(dt) {
    const router = recs.get('router');
    const up = router && router.present && router.status !== 'offline';
    routerLed.visible = !!up;
    if (!up) { routerActivity = 0; return; }
    routerActivity *= Math.exp(-dt * 5);
    routerLed.position.copy(router.port);
    routerLed.position.y += 0.04;
    const flicker = routerActivity > 0.03 && !reducedMotion ? 0.55 + 0.45 * Math.random() : 1;
    routerLedMat.opacity = (0.18 + 0.82 * routerActivity) * flicker;
    routerLed.scale.setScalar(0.22 + 0.2 * routerActivity);
  }

  function updateTvGlow(dt) {
    const tvRec = recs.get('tv');
    const tv = tvRec && state.devices.get('tv');
    const on = !!tv && tv.props.power !== false && tvRec.status !== 'offline';
    tvGlowLevel += ((on ? 1 : 0) - tvGlowLevel) * Math.min(1, dt * 4);
    tvGlow.visible = tvGlowLevel > 0.01;
    if (!tvGlow.visible) return;
    tvGlow.position.set(tvRec.group.position.x, tvRec.group.position.y + 0.35, tvRec.group.position.z + 0.9);
    tvGlowColor.setHex(TV_CHANNEL_HEX[tv.props.channel] ?? 0x7fa8ff);
    tvGlow.color.lerp(tvGlowColor, Math.min(1, dt * 3));
    const flicker = reducedMotion ? 1 : 0.82 + 0.18 * Math.sin(realT * 7.3) * Math.sin(realT * 2.9 + 1.3);
    tvGlow.intensity = 2.6 * tvGlowLevel * flicker;
  }

  /** Put a ring marker on a device: upright around wall devices, flat on the floor otherwise. */
  function placeRing(obj, rec, lift, scale) {
    if (rec.mount === 'wall') {
      obj.position.copy(rec.center);
      obj.rotation.set(Math.PI / 2, rec.rotY, 0, 'YXZ');
    } else {
      const c = rec.device.type === 'vacuum' ? rec.port : rec.center;
      obj.position.set(c.x, rec.ringY + lift, c.z);
      obj.rotation.set(0, 0, 0, 'XYZ');
    }
    obj.scale.setScalar(rec.radius * scale);
  }

  function updateMarkers(dt) {
    const sel = state.selectedId ? recs.get(state.selectedId) : null;
    if (sel && sel.present) {
      selectRing.visible = true;
      placeRing(selectRing, sel, 0.006, reducedMotion ? 1 : 1 + 0.05 * Math.sin(realT * 3));
      for (const dash of selectDash) dash.rotation.y += dt * 0.6;
    } else {
      selectRing.visible = false;
    }
    const hov = hoverId && hoverId !== state.selectedId ? recs.get(hoverId) : null;
    if (hov && hov.present) {
      hoverRing.visible = true;
      placeRing(hoverRing, hov, 0.004, 1);
    } else {
      hoverRing.visible = false;
    }
    for (let i = 0; i < blockRings.length; i++) {
      const r = blockRings[i];
      if (!r.visible) continue;
      const u = r.userData;
      u.t += dt;
      const f = u.t / u.dur;
      if (f >= 1) { r.visible = false; continue; }
      const e = 1 - Math.pow(1 - f, 3);
      r.scale.setScalar(u.from + (u.to - u.from) * e);
      r.material.opacity = 0.9 * (1 - f);
    }
  }

  function updateLabels() {
    for (let i = 0; i < recList.length; i++) {
      const rec = recList[i];
      if (!rec.label.visible) continue;
      rec.label.position.set(rec.port.x, rec.group.position.y + rec.topOffset + 0.06, rec.port.z);
      if (rec.port.y > rec.label.position.y) rec.label.position.y = rec.port.y + 0.06;
    }
  }

  let shadowAcc = 0;
  function tick(realDtMs, simDtMs) {
    const realDt = Math.min(Math.max(realDtMs, 0), 100) / 1000;
    const simDt = Math.max(simDtMs || 0, 0) / 1000;
    realT += realDt;
    simT += simDt;
    if (width <= 1 || height <= 1) resize();

    if (hoverDirty && !dragging) {
      hoverDirty = false;
      setHover(pick());
    }
    updateCamera(realDt);
    updateDevices(realDt, simDt);
    updateLinks(realDt);
    packets.update(realDt, !reducedMotion, reducedMotion ? 5 : 10);
    updateShield(realDt);
    updateRouterLed(realDt);
    updateTvGlow(realDt);
    updateMarkers(realDt);
    updateLabels();
    adaptResolution(realDtMs);

    if (contextLost || width <= 1 || height <= 1) return;
    shadowAcc += realDt;
    if (shadowAcc >= SHADOW_EVERY_SEC) {
      shadowAcc = 0;
      renderer.shadowMap.needsUpdate = true;
    }
    renderer.render(scene, camera);
    labelRenderer.render(scene, camera);
  }

  return { tick, focus };
}
