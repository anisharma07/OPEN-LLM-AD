// Single mutable source of truth. Modules read freely; each field has one
// owning module that writes it (see CONTRACT.md, "State ownership").
import { DEVICES } from './catalog.js';

const SIM_START = Date.UTC(2026, 9, 2, 14, 12, 0); // 19:42 IST, a weekday evening

function freshDevice(d) {
  return {
    ...d,
    props: structuredClone(d.props),
    online: true,
    status: 'ok',          // 'ok' | 'alert' | 'blocked' | 'offline'
    blocked: false,
    stats: { tx: 0, rx: 0, dropped: 0 },
  };
}

export function createState() {
  const devices = new Map(DEVICES.map(d => [d.id, freshDevice(d)]));
  const byIp = new Map([...devices.values()].map(d => [d.ip, d.id]));
  return {
    time: { simMs: 0, epochMs: SIM_START, speed: 1, paused: false },
    devices,
    byIp,
    selectedId: null,
    ids: {
      mode: 'prevent',                 // 'prevent' (inline IPS) | 'detect' (alert only)
      layers: { ml: true, tcp: true, arp: true, mqtt: false, drift: true },
      threshold: 0.8,                  // ML attack-probability threshold
      autoBlockAfter: 5,               // alerts from one source within 10 s before a block
      blocked: new Map(),              // ip -> { ip, deviceId, reason, ruleId, at, auto }
      alerts: [],                      // newest first, capped at 300
      metrics: null,                   // latest 'ids:metrics' payload
    },
  };
}

export function resetState(state) {
  const fresh = createState();
  state.time.simMs = 0;
  state.time.epochMs = fresh.time.epochMs;
  state.devices = fresh.devices;
  state.byIp = fresh.byIp;
  state.ids.blocked = fresh.ids.blocked;
  state.ids.alerts = fresh.ids.alerts;
  state.ids.metrics = null;
}

export function deviceByIp(state, ip) {
  const id = state.byIp.get(ip);
  return id ? state.devices.get(id) : null;
}

export function simNow(state) {
  return new Date(state.time.epochMs + state.time.simMs);
}
