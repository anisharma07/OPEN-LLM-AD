// Boot order and the single animation loop.
import { bus } from './bus.js';
import { createState } from './state.js';
import { createIDS } from './ids.js';
import { createNetwork } from './network.js';
import { createScene } from './scene.js';
import { createUI } from './ui.js';

const MAX_FRAME_MS = 100; // clamp after tab switches so the sim does not jump

async function boot() {
  const state = createState();
  const ids = createIDS({ bus, state });
  const network = createNetwork({ bus, state, ids });
  const ui = createUI({ bus, state });
  let scene = null;
  try {
    scene = await createScene({ bus, state, container: document.getElementById('scene') });
  } catch (err) {
    console.error('[main] 3D scene failed to start', err);
    bus.emit('scene:error', { message: String(err?.message || err) });
  }

  bus.on('sim:speed', ({ speed }) => { state.time.speed = speed; });
  bus.on('sim:pause', ({ paused }) => { state.time.paused = paused; });

  let last = performance.now();
  function frame(now) {
    const realDt = Math.min(now - last, MAX_FRAME_MS);
    last = now;
    const simDt = state.time.paused ? 0 : realDt * state.time.speed;
    if (simDt > 0) {
      state.time.simMs += simDt;
      network.tick(simDt);
      ids.tick(simDt);
    }
    scene?.tick(realDt, simDt);
    ui.tick(realDt);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);

  // Handy for debugging and for the automated smoke test.
  window.__shield = { bus, state, ids, network, scene, ui };
  bus.emit('sim:ready', {});
}

boot();
