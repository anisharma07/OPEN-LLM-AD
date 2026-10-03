// SHIELD-IoT inline sensor: a browser simulation of the architecture in
// Anirudh Sharma's research book (NIT Hamirpur). It fuses
//   1. preprocessing: a conservative 37-feature projection (features.js),
//   2. ML evidence: a hand-weighted LightGBM surrogate (features.js),
//   3. deterministic protocol invariant engines (TCP Step 5A, ARP Step 5B,
//      MQTT planned Step 5C, experimental and off by default),
//   4. a home device-inventory check,
//   5. score fusion and an allow / alert / drop decision,
//   6. the response: alerts, aggregation and blocking at the router firewall,
//   7. a PSI drift monitor with a PIWS sample queue (a research hypothesis),
//   8. live metrics for the dashboard.
// Ground truth (pkt.label) is read only after the verdict, to count false
// alarms, which a real sensor could not do.

import { DEVICES } from './catalog.js';
import { deviceByIp } from './state.js';
import {
  Deque, PROTOS, PROTO_OTHER, bit, createFeatureExtractor, createMLSurrogate, isLanIp, mulberry32, num,
} from './features.js';

// ---------------------------------------------------------------------------
// Tunables

const SEED = 42;                       // the frozen baseline's seed
const INVARIANT_SCORE = 0.99;
const UNKNOWN_INVENTORY_SCORE = 0.9;
const ALERT_AGGREGATE_MS = 3000;       // same source + class within 3 s -> one row, count++
const AUTO_BLOCK_WINDOW_MS = 10000;
const MAX_ALERTS = 300;
const PIWS_CAP = 5000;
const METRICS_EVERY_REAL_MS = 500;
const PPS_WINDOW_MS = 1000;
const PROTO_MIX_MS = 10000;
const THREAT_WINDOW_MS = 30000;
const THREAT_RECENT_MS = 15000;
const SWEEP_EVERY_MS = 5000;

const DRIFT = {
  refMs: 30000,          // learn the reference over the first ~30 s of sim time
  refMinSamples: 300,    // ...but never from fewer packets than this
  refMaxMs: 90000,       // give up waiting for refMinSamples after this long
  recentMs: 15000,       // sliding comparison window
  recentMin: 400,        // keep at least this many recent packets (up to recentMaxMs)
  recentMaxMs: 60000,
  evalEveryMs: 1000,
  warn: 0.1,
  drift: 0.25,
  hysteresis: 0.03,      // step back down only once PSI is this far below a boundary
  smoothing: 0.35,       // EMA weight of each new PSI evaluation (damps flapping)
  uniformMix: 0.1,       // share of the uniform distribution mixed into both histograms
};
const LEN_EDGES = [64, 128, 256, 512, 1024];
const SCORE_EDGES = [0.008, 0.012, 0.017, 0.025, 0.05];

const SEVERITY_RANK = { low: 0, medium: 1, high: 2, critical: 3 };
const PROTECTED_ROLES = new Set(['infra', 'cloud']);       // never auto-blocked
const NEVER_BLOCK_IDS = new Set(['router', 'shield']);     // not even manually

export const LAYER_HIT_KEYS = [
  'ml', 'inventory', 'INV_TCP_01', 'INV_TCP_02', 'INV_TCP_03', 'INV_ARP_01', 'INV_MQTT_01', 'INV_MQTT_02', 'INV_MQTT_03',
];

// ---------------------------------------------------------------------------
// Layer 3: protocol invariant engines. Pure functions of the raw packet,
// kept separate from preprocessing (book, Appendix B). Each returns the list
// of violated rules; a missing layer or field means "not applicable", never
// "invalid".

/** Step 5A, TCP. */
export function tcpInvariants(tcp) {
  if (!tcp || !tcp.flags) return [];
  const f = tcp.flags;
  const SYN = bit(f.SYN), ACK = bit(f.ACK), FIN = bit(f.FIN), RST = bit(f.RST), PSH = bit(f.PSH), URG = bit(f.URG);
  const hits = [];
  if (SYN && FIN) hits.push({ ruleId: 'INV_TCP_01', layer: 'tcp', msg: 'SYN and FIN both set' });
  const payload = num(tcp.len);
  if (!SYN && !ACK && !FIN && !RST && !PSH && !URG && payload !== null && payload > 0) {
    hits.push({ ruleId: 'INV_TCP_02', layer: 'tcp', msg: `no flags set on a ${payload}-byte segment` });
  }
  // A SYN-ACK reply is valid: only the initiator's first segment (connInit) must not carry ACK.
  if (tcp.connInit !== undefined && tcp.connInit !== null && bit(tcp.connInit) && SYN && ACK) {
    hits.push({ ruleId: 'INV_TCP_03', layer: 'tcp', msg: 'initial SYN carries ACK' });
  }
  return hits;
}

/** Step 5B, ARP: opcode must be 1 (request) or 2 (reply) and the hardware size 6 (Ethernet). */
export function arpInvariants(arp) {
  if (!arp) return [];
  const opcode = num(arp.opcode);
  const hwSize = num(arp.hwSize);
  const problems = [];
  if (opcode !== null && opcode !== 1 && opcode !== 2) problems.push(`opcode ${opcode} not in {1,2}`);
  if (hwSize !== null && hwSize !== 6) problems.push(`hardware size ${hwSize} != 6`);
  return problems.length ? [{ ruleId: 'INV_ARP_01', layer: 'arp', msg: problems.join(', ') }] : [];
}

/** Parses MQTT connect flags, which the audit found in several encodings ("0x02", "2", 2). */
function parseFlags(v) {
  if (typeof v === 'number') return Number.isInteger(v) ? v : null;
  if (typeof v !== 'string' || !v.trim()) return null;
  const s = v.trim().toLowerCase();
  const n = s.startsWith('0x') ? parseInt(s.slice(2), 16) : s.startsWith('0b') ? parseInt(s.slice(2), 2) : parseInt(s, 10);
  return Number.isFinite(n) ? n : null;
}

/** Planned Step 5C, MQTT (experimental): structural rules from the MQTT specification. */
export function mqttInvariants(mqtt) {
  if (!mqtt) return [];
  const type = num(mqtt.msgtype);
  const hits = [];
  if (type === 0 || type === 15) hits.push({ ruleId: 'INV_MQTT_01', layer: 'mqtt', msg: `reserved packet type ${type}` });
  if (type === 1) {
    const flags = parseFlags(mqtt.conflags);
    if (flags !== null && (flags & 1)) hits.push({ ruleId: 'INV_MQTT_02', layer: 'mqtt', msg: 'CONNECT reserved flag bit 0 set' });
  }
  if (type === 3 && num(mqtt.qos) === 3) hits.push({ ruleId: 'INV_MQTT_03', layer: 'mqtt', msg: 'PUBLISH with QoS 3' });
  return hits;
}

// ---------------------------------------------------------------------------
// Layer 4: the home inventory (every catalog device except the unregistered board).

const INVENTORY = (() => {
  const macs = new Set();
  const ips = new Set();
  for (const d of DEVICES) {
    if (d.role === 'unknown') continue;
    if (d.mac) macs.add(d.mac.toLowerCase());
    if (d.ip) ips.add(d.ip);
  }
  return { macs, ips };
})();

/** Empty string when the source is registered or not on the LAN, else the reason it is unknown. */
export function inventoryCheck(pkt) {
  const ip = pkt.srcIp;
  const lan = isLanIp(ip);
  if (ip && !lan && ip !== '0.0.0.0') return '';           // WAN source: not applicable
  const mac = typeof pkt.srcMac === 'string' ? pkt.srcMac.toLowerCase() : null;
  if (mac && !INVENTORY.macs.has(mac)) return `Inventory: MAC ${mac} not registered`;
  if (lan && !INVENTORY.ips.has(ip)) return `Inventory: IP ${ip} not registered`;
  return '';
}

// ---------------------------------------------------------------------------
// Layer 7: PSI drift monitor

function binOf(v, edges) {
  let i = 0;
  while (i < edges.length && v >= edges[i]) i++;
  return i;
}

function histogram() {
  return {
    len: new Float64Array(LEN_EDGES.length + 1),
    proto: new Float64Array(PROTO_OTHER + 1),
    score: new Float64Array(SCORE_EDGES.length + 1),
    total: 0, n: 0, scoreTotal: 0, scoreN: 0,
  };
}

function addTo(h, e, sign) {
  const w = sign * e.w;
  h.len[e.l] += w;
  h.proto[e.p] += w;
  h.total += w;
  h.n += sign;
  if (e.s >= 0) { h.score[e.s] += w; h.scoreTotal += w; h.scoreN += sign; }
}

/**
 * Population Stability Index between a reference and a current histogram.
 * Both are mixed with a little of the uniform distribution (lambda) so a bin
 * that empties out, say when one device goes quiet, gives a finite, bounded
 * contribution instead of exploding.
 */
function psiOf(ref, refTotal, cur, curTotal) {
  if (refTotal <= 0 || curTotal <= 0) return null;
  const k = ref.length;
  const lam = DRIFT.uniformMix;
  let s = 0;
  for (let i = 0; i < k; i++) {
    const p = (1 - lam) * Math.max(0, cur[i]) / curTotal + lam / k;
    const q = (1 - lam) * Math.max(0, ref[i]) / refTotal + lam / k;
    s += (p - q) * Math.log(p / q);
  }
  return s;
}

function createDriftMonitor() {
  let ref, cur, recent, phase, refStart, psi, level, parts, evaluated;

  function reset() {
    ref = histogram();
    cur = histogram();
    recent = new Deque();
    phase = 'learning';
    refStart = null;
    psi = 0;
    level = 'stable';
    parts = { len: null, proto: null, score: null };
    evaluated = false;
  }
  reset();

  /** Records one inspected packet (bins precomputed; s = -1 when the ML layer is off). */
  function observe(t, l, p, s, w) {
    const e = { t, l, p, s, w };
    if (phase === 'learning') {
      if (refStart === null) refStart = t;
      addTo(ref, e, 1);
    }
    addTo(cur, e, 1);
    recent.push(e);
  }

  function prune(now) {
    while (recent.size) {
      const e = recent.peek();
      const age = now - e.t;
      if (age > DRIFT.recentMaxMs || (age > DRIFT.recentMs && recent.size > DRIFT.recentMin)) addTo(cur, recent.shift(), -1);
      else break;
    }
  }

  /** Re-evaluates PSI; returns true when the drift state changed. */
  function evaluate(now) {
    prune(now);
    if (phase === 'learning') {
      if (refStart === null) return false;
      const elapsed = now - refStart;
      const ready = (elapsed >= DRIFT.refMs && ref.n >= DRIFT.refMinSamples) || (elapsed >= DRIFT.refMaxMs && ref.n >= 50);
      if (!ready) return false;
      phase = 'monitoring';
    }
    if (cur.n < 50) return false;
    parts = {
      len: psiOf(ref.len, ref.total, cur.len, cur.total),
      proto: psiOf(ref.proto, ref.total, cur.proto, cur.total),
      score: cur.scoreN >= 50 && ref.scoreN >= 50 ? psiOf(ref.score, ref.scoreTotal, cur.score, cur.scoreTotal) : null,
    };
    const vals = Object.values(parts).filter(v => v !== null);
    const raw = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
    psi = evaluated ? psi + DRIFT.smoothing * (raw - psi) : raw;
    evaluated = true;

    let next = level;
    const h = DRIFT.hysteresis;
    if (psi > DRIFT.drift) next = 'drift';
    else if (psi >= DRIFT.warn) next = level === 'drift' && psi > DRIFT.drift - h ? 'drift' : 'warning';
    else next = level !== 'stable' && psi > DRIFT.warn - h ? 'warning' : 'stable';
    if (next !== level) { level = next; return true; }
    return false;
  }

  return {
    observe, evaluate, reset,
    get psi() { return psi; },
    get state() { return level; },
    get phase() { return phase; },
    get parts() { return parts; },
  };
}

// ---------------------------------------------------------------------------

const severityOf = score => (score >= 0.95 ? 'critical' : score >= 0.85 ? 'high' : score >= 0.7 ? 'medium' : 'low');
const fx = (v, d = 2) => v.toFixed(d);
const r3 = v => Math.round(v * 1000) / 1000;
const roundParts = p => Object.fromEntries(Object.entries(p).map(([k, v]) => [k, v === null ? null : r3(v)]));

export function createIDS({ bus, state }) {
  const extractor = createFeatureExtractor();
  const drift = createDriftMonitor();
  let rng, ml, cfg, c, layerHits;
  let allowedWin, droppedWin, protoWin, protoCounts, alertEvents;
  let srcAlerts;         // ip -> Deque of alert times (auto-block counter)
  let overrides;         // ips the operator unblocked: no further auto-blocks
  let piws;
  let latencyAvg, latencyN;
  let lastNow, lastDriftEval, lastSweep, lastMetricsReal;
  let alertSeq = 0;      // never reset, so alert ids stay unique across sim resets

  function readConfig() {
    const s = state.ids;
    const L = s.layers || {};
    const thr = Number(s.threshold);
    const after = Math.round(Number(s.autoBlockAfter));
    return {
      mode: s.mode === 'detect' ? 'detect' : 'prevent',
      layers: { ml: L.ml !== false, tcp: L.tcp !== false, arp: L.arp !== false, mqtt: L.mqtt === true, drift: L.drift !== false },
      threshold: Number.isFinite(thr) ? Math.min(0.99, Math.max(0.05, thr)) : 0.8,
      autoBlockAfter: Number.isFinite(after) && after >= 1 ? after : 5,
    };
  }

  function init() {
    rng = mulberry32(SEED);
    ml = createMLSurrogate(rng);
    cfg = readConfig();
    c = { inspected: 0, allowed: 0, alerted: 0, dropped: 0, firewallDrops: 0, normalSeen: 0, falseAlarms: 0 };
    layerHits = Object.fromEntries(LAYER_HIT_KEYS.map(k => [k, 0]));
    allowedWin = new Deque();
    droppedWin = new Deque();
    protoWin = new Deque();
    protoCounts = new Map();
    alertEvents = new Deque();
    srcAlerts = new Map();
    overrides = new Set();
    piws = [];
    latencyAvg = 0;
    latencyN = 0;
    lastNow = 0;
    lastDriftEval = 0;
    lastSweep = 0;
    lastMetricsReal = -Infinity;
    extractor.reset();
    drift.reset();
  }

  const nameOf = (id, ip) => (id && state.devices.get(id)?.name) || id || ip || 'unknown host';
  const alertThreshold = () => Math.max(0.5, cfg.threshold - 0.25);

  // ---- inspect: the per-packet pipeline -----------------------------------

  function inspect(pkt) {
    const t0 = performance.now();
    const now = Number.isFinite(pkt.t) ? pkt.t : state.time.simMs;
    if (now > lastNow) lastNow = now;
    const reasons = [];

    // 1. Preprocessing (windows are always maintained so layers can be toggled live).
    const x = extractor.extract(pkt, now);

    // 2. ML evidence.
    let mlRes = null, mlSusp = 0;
    if (cfg.layers.ml) {
      mlRes = ml.score(x);
      mlSusp = mlRes.suspicion;
      if (mlRes.top === 'Normal') {
        reasons.push(mlRes.attackProb >= 0.1
          ? `ML: Normal p=${fx(mlRes.pNormal)} (${mlRes.attack} ${fx(mlRes.attackProb)})`
          : `ML: Normal p=${fx(mlRes.pNormal)}`);
      } else {
        reasons.push(`ML: ${mlRes.top} p=${fx(mlRes.topProb)} (Normal ${fx(mlRes.pNormal)})`);
      }
    } else {
      reasons.push('ML: layer off');
    }

    // 3. Protocol invariants, only where the layer is present.
    let hits = [];
    if (cfg.layers.tcp && pkt.tcp) hits = hits.concat(tcpInvariants(pkt.tcp));
    if (cfg.layers.arp && pkt.arp) hits = hits.concat(arpInvariants(pkt.arp));
    if (cfg.layers.mqtt && pkt.mqtt) hits = hits.concat(mqttInvariants(pkt.mqtt));
    for (const h of hits) reasons.push(`${h.ruleId}: ${h.msg}`);

    // 4. Inventory.
    const invReason = inventoryCheck(pkt);
    const unknown = invReason !== '';
    if (unknown) reasons.push(invReason);

    // 5. Fusion and decision.
    let score = mlSusp;
    if (hits.length) score = Math.max(score, INVARIANT_SCORE);
    if (unknown) score = Math.max(score, UNKNOWN_INVENTORY_SCORE);
    const alertAt = alertThreshold();
    const strong = hits.length > 0 || score >= cfg.threshold;
    let action;
    if (strong) action = cfg.mode === 'prevent' ? 'drop' : 'alert';
    else action = score >= alertAt ? 'alert' : 'allow';

    const inDrift = cfg.layers.drift && drift.state === 'drift';
    if (cfg.layers.drift && drift.state !== 'stable') reasons.push(`Drift: PSI ${fx(drift.psi)} (${drift.state})`);
    if (hits.length) reasons.push(`Fusion: invariant evidence ${fx(score)} -> ${action}`);
    else if (strong) reasons.push(`Fusion: score ${fx(score)} >= ${fx(cfg.threshold)} -> ${action}`);
    else if (action === 'alert') reasons.push(`Fusion: score ${fx(score)} >= ${fx(alertAt)} -> alert`);
    else reasons.push(`Fusion: score ${fx(score)} -> allow`);

    const latencyUs = (performance.now() - t0) * 1000;
    const verdict = {
      action,
      score,
      mlClass: mlRes ? mlRes.top : null,
      mlProb: mlRes ? mlRes.topProb : null,
      invariantHits: hits,
      inventory: unknown ? 'unknown' : 'known',
      drift: inDrift,
      reasons,
      latencyUs,
    };

    // Bookkeeping and response (outside the measured detection path).
    account(pkt, verdict, now, x, mlRes, mlSusp >= alertAt, unknown);
    if (action !== 'allow') respond(pkt, verdict, now, hits, unknown, invReason, mlRes);
    return verdict;
  }

  function account(pkt, v, now, x, mlRes, mlHit, unknown) {
    c.inspected++;
    if (v.action === 'allow') c.allowed++;
    else if (v.action === 'alert') c.alerted++;
    else c.dropped++;
    (v.action === 'drop' ? droppedWin : allowedWin).push(now);

    const p = x[1];
    protoWin.push({ t: now, p });
    protoCounts.set(p, (protoCounts.get(p) || 0) + 1);

    if (mlHit) layerHits.ml++;
    if (unknown) layerHits.inventory++;
    for (const h of v.invariantHits) layerHits[h.ruleId] = (layerHits[h.ruleId] || 0) + 1;

    // Ground truth, read only after the verdict.
    if (pkt.label === 'Normal') {
      c.normalSeen++;
      if (v.action !== 'allow') c.falseAlarms++;
    }

    latencyN++;
    latencyAvg += (v.latencyUs - latencyAvg) / Math.min(latencyN, 50);   // mean, then EMA over ~50

    if (cfg.layers.drift) {
      // Each packet is weighted by 1/sqrt(its source's rate) so one chatty stream
      // cannot swamp the reference distribution.
      const w = 1 / Math.sqrt(Math.max(1, extractor.sourceRate(pkt.srcIp || pkt.srcMac || '?')));
      const s = mlRes ? binOf(mlRes.suspicion, SCORE_EDGES) : -1;
      drift.observe(now, binOf(x[0] ?? 0, LEN_EDGES), p, s, w);
      if (v.drift && (v.invariantHits.length || unknown)) {
        if (piws.length >= PIWS_CAP) piws.shift();
        piws.push({ t: now, srcIp: pkt.srcIp, label: v.invariantHits[0]?.ruleId ?? 'INV_INVENTORY', features: x });
        v.reasons.push('PIWS: queued as a weakly labelled drift-repair sample (research hypothesis)');
      }
    }
  }

  // ---- 6. response ----------------------------------------------------------

  function raiseAlert({ now, srcIp, srcId, dstId, cls, severity, ruleId, score, msg, action }) {
    alertEvents.push({ t: now, rank: SEVERITY_RANK[severity] });
    const list = state.ids.alerts;
    const scan = Math.min(list.length, 64);
    for (let i = 0; i < scan; i++) {
      const a = list[i];
      if (a.srcIp === srcIp && a.cls === cls) {
        if (now - a.lastT <= ALERT_AGGREGATE_MS) {
          a.count++;
          a.lastT = now;
          a.score = Math.max(a.score, score);
          if (SEVERITY_RANK[severity] > SEVERITY_RANK[a.severity]) a.severity = severity;
          if (action === 'drop') a.action = 'drop';
          a.msg = msg;
          bus.emit('ids:alert', a);
          return a;
        }
        break;   // the newest alert for this pair is too old: start a new row
      }
    }
    const alert = {
      id: ++alertSeq, t: now, srcIp, srcId, dstId, cls, severity, ruleId, score, msg, count: 1,
      lastT: now, action,
    };
    list.unshift(alert);
    if (list.length > MAX_ALERTS) list.length = MAX_ALERTS;
    bus.emit('ids:alert', alert);
    return alert;
  }

  function respond(pkt, v, now, hits, unknown, invReason, mlRes) {
    const srcIp = pkt.srcIp;
    const srcId = pkt.src ?? state.byIp.get(srcIp) ?? null;
    const dstId = pkt.dst ?? state.byIp.get(pkt.dstIp) ?? null;
    const route = `${nameOf(srcId, srcIp)} -> ${nameOf(dstId, pkt.dstIp)}`;
    let cls, ruleId, severity, msg;
    if (hits.length) {
      const h = hits[0];
      cls = `${h.layer.toUpperCase()} invariant`;
      ruleId = h.ruleId;
      severity = severityOf(v.score);
      msg = `${h.ruleId}: ${h.msg} (${route})`;
    } else if (unknown) {
      cls = 'Unregistered device';
      ruleId = 'INV_INVENTORY';
      severity = 'high';
      msg = `${invReason.replace('Inventory: ', '')} in the home inventory (${srcIp})`;
    } else {
      cls = mlRes ? mlRes.attack : 'Anomaly';
      ruleId = 'ML';
      severity = severityOf(v.score);
      msg = mlRes
        ? `ML surrogate leans ${mlRes.attack} p=${fx(mlRes.attackProb)}, Normal ${fx(mlRes.pNormal)} (${route})`
        : `Suspicion ${fx(v.score)} (${route})`;
    }
    raiseAlert({ now, srcIp, srcId, dstId, cls, severity, ruleId, score: v.score, msg, action: v.action });

    if (cfg.mode !== 'prevent' || !srcIp) return;
    if (unknown || (hits.length && isLanIp(srcIp))) {
      tryAutoBlock(srcIp, v, now, unknown ? `Unregistered device: ${invReason.replace('Inventory: ', '')}` : `${ruleId}: ${hits[0].msg}`, ruleId);
      return;
    }
    let q = srcAlerts.get(srcIp);
    if (!q) { q = new Deque(); srcAlerts.set(srcIp, q); }
    while (q.size && q.peek() <= now - AUTO_BLOCK_WINDOW_MS) q.shift();
    q.push(now);
    if (q.size >= cfg.autoBlockAfter) {
      tryAutoBlock(srcIp, v, now, `${q.size} alerts in 10 s (latest: ${cls})`, ruleId);
    }
  }

  function tryAutoBlock(ip, v, now, reason, ruleId) {
    if (state.ids.blocked.has(ip)) return;
    const dev = deviceByIp(state, ip);
    if (dev && PROTECTED_ROLES.has(dev.role)) {
      protectedAlert(ip, dev, reason, v.score, now);
      v.reasons.push('Response: auto-block withheld (protected infrastructure)');
    } else if (overrides.has(ip)) {
      v.reasons.push('Response: auto-block suppressed (operator unblocked this source)');
    } else if (block(ip, { reason, ruleId, auto: true })) {
      v.reasons.push(`Response: ${ip} blocked at the router firewall`);
    }
    srcAlerts.delete(ip);
  }

  function protectedAlert(ip, dev, reason, score, now = state.time.simMs) {
    raiseAlert({
      now, srcIp: ip, srcId: dev.id, dstId: null, cls: 'Protected infrastructure', severity: 'medium',
      ruleId: 'PROTECTED_INFRA', score: score ?? 0, action: 'alert',
      msg: `Auto-block withheld: ${dev.name} is protected infrastructure (${reason})`,
    });
  }

  /** Adds a firewall block. Manual blocks work on any device except the router and the sensor. */
  function block(ip, { reason = 'Blocked by the operator', ruleId = null, auto = false } = {}) {
    if (!ip) return false;
    const dev = deviceByIp(state, ip);
    if (dev && NEVER_BLOCK_IDS.has(dev.id)) {
      if (auto) protectedAlert(ip, dev, reason, null);
      else bus.emit('toast', { kind: 'warn', text: `${dev.name} cannot be blocked: every packet in the home crosses it.` });
      return false;
    }
    if (auto && dev && PROTECTED_ROLES.has(dev.role)) {
      protectedAlert(ip, dev, reason, null);
      return false;
    }
    if (state.ids.blocked.has(ip)) return false;
    const deviceId = dev?.id ?? null;
    state.ids.blocked.set(ip, { ip, deviceId, reason, ruleId, at: state.time.simMs, auto: !!auto });
    if (dev) dev.blocked = true;
    if (!auto) overrides.delete(ip);
    srcAlerts.delete(ip);
    bus.emit('ids:block', { ip, deviceId, reason, ruleId, auto: !!auto });
    return true;
  }

  /** Lifts a block. The operator's decision sticks: that source is not auto-blocked again until reset. */
  function unblock(ip) {
    const entry = state.ids.blocked.get(ip);
    if (!entry) return false;
    state.ids.blocked.delete(ip);
    const dev = deviceByIp(state, ip);
    if (dev) dev.blocked = false;
    overrides.add(ip);
    srcAlerts.delete(ip);
    bus.emit('ids:unblock', { ip, deviceId: entry.deviceId ?? dev?.id ?? null });
    return true;
  }

  const isBlocked = ip => state.ids.blocked.has(ip);

  // ---- 8. metrics -----------------------------------------------------------

  function pruneWindows(now) {
    while (allowedWin.size && allowedWin.peek() <= now - PPS_WINDOW_MS) allowedWin.shift();
    while (droppedWin.size && droppedWin.peek() <= now - PPS_WINDOW_MS) droppedWin.shift();
    while (protoWin.size && protoWin.peek().t <= now - PROTO_MIX_MS) {
      const { p } = protoWin.shift();
      const n = protoCounts.get(p) - 1;
      if (n > 0) protoCounts.set(p, n); else protoCounts.delete(p);
    }
    while (alertEvents.size && alertEvents.peek().t <= now - THREAT_WINDOW_MS) alertEvents.shift();
  }

  function threatLevel(now) {
    const WEIGHT = [1, 3, 6, 10];
    let weight = 0, recentMax = -1;
    for (let i = 0; i < alertEvents.size; i++) {
      const e = alertEvents.at(i);
      weight += WEIGHT[e.rank];
      if (now - e.t <= THREAT_RECENT_MS && e.rank > recentMax) recentMax = e.rank;
    }
    if (recentMax >= 3 || weight >= 40) return 'critical';
    if (recentMax >= 2 || weight >= 15) return 'high';
    if (recentMax >= 1 || weight >= 4) return 'elevated';
    return 'low';
  }

  function buildMetrics(now = Math.max(state.time.simMs, lastNow)) {
    pruneWindows(now);
    const protoMix = {};
    for (const [p, n] of protoCounts) protoMix[p === PROTO_OTHER ? 'OTHER' : PROTOS[p]] = n;
    const allowed1 = allowedWin.size, dropped1 = droppedWin.size;
    return {
      inspected: c.inspected,
      allowed: c.allowed,
      alerted: c.alerted,
      dropped: c.dropped,
      firewallDrops: c.firewallDrops,
      pps: allowed1 + dropped1,
      ppsAllowed: allowed1,
      ppsDropped: dropped1,
      falseAlarms: c.falseAlarms,
      falseAlarmRate: c.normalSeen ? c.falseAlarms / c.normalSeen : 0,
      avgLatencyUs: Math.round(latencyAvg * 100) / 100,
      layerHits: { ...layerHits },
      drift: cfg.layers.drift
        ? { psi: r3(drift.psi), state: drift.state, phase: drift.phase, parts: roundParts(drift.parts) }
        : { psi: 0, state: 'stable', phase: 'off' },
      piwsBuffer: piws.length,
      threat: threatLevel(now),
      protoMix,
    };
  }

  function emitMetrics(now) {
    const m = buildMetrics(now);
    state.ids.metrics = m;
    bus.emit('ids:metrics', m);
    return m;
  }

  // ---- tick, config, reset ----------------------------------------------------

  // simDtMs is not needed: windows key off the sim clock, metrics off real time.
  function tick(simDtMs) { // eslint-disable-line no-unused-vars
    const now = Math.max(state.time.simMs, lastNow);
    if (now < lastDriftEval) lastDriftEval = now;     // sim clock went backwards (reset)
    if (now < lastSweep) lastSweep = now;

    if (cfg.layers.drift && now - lastDriftEval >= DRIFT.evalEveryMs) {
      lastDriftEval = now;
      if (drift.evaluate(now)) {
        bus.emit('ids:drift', {
          psi: drift.psi,
          state: drift.state,
          msg: drift.state === 'drift'
            ? `Drift: PSI ${fx(drift.psi)} > ${DRIFT.drift}. PIWS (research hypothesis) now queues invariant- and inventory-labelled samples for drift repair.`
            : `Drift: PSI ${fx(drift.psi)} (${drift.state})`,
        });
      }
    }
    if (now - lastSweep >= SWEEP_EVERY_MS) {
      lastSweep = now;
      extractor.sweep(now);
      for (const [ip, q] of srcAlerts) {
        while (q.size && q.peek() <= now - AUTO_BLOCK_WINDOW_MS) q.shift();
        if (!q.size) srcAlerts.delete(ip);
      }
    }
    const real = performance.now();
    if (real - lastMetricsReal >= METRICS_EVERY_REAL_MS) {
      lastMetricsReal = real;
      emitMetrics(now);
    }
  }

  function onConfig() {
    const prev = cfg;
    cfg = readConfig();
    if (prev.layers.drift !== cfg.layers.drift) {
      const was = drift.state;
      drift.reset();                                    // re-enabling relearns the reference
      lastDriftEval = Math.max(state.time.simMs, lastNow);
      if (was !== 'stable') bus.emit('ids:drift', { psi: 0, state: 'stable', msg: 'Drift monitor reset' });
    }
  }

  function reset() {
    const wasDrift = drift.state;
    for (const [ip] of state.ids.blocked) {
      const dev = deviceByIp(state, ip);
      if (dev) dev.blocked = false;
    }
    state.ids.blocked.clear();
    state.ids.alerts.length = 0;
    init();
    if (wasDrift !== 'stable') bus.emit('ids:drift', { psi: 0, state: 'stable', msg: 'Drift monitor reset' });
    emitMetrics(state.time.simMs);
  }

  // ---- wiring -----------------------------------------------------------------

  init();
  bus.on('ids:config', onConfig);
  bus.on('ids:request-block', p => { if (p?.ip) block(p.ip, { reason: 'Blocked by the operator', auto: false }); });
  bus.on('ids:request-unblock', p => { if (p?.ip) unblock(p.ip); });
  bus.on('packet:dropped', p => {
    if (p?.reason !== 'firewall') return;
    c.firewallDrops++;
    droppedWin.push(Number.isFinite(p.pkt?.t) ? p.pkt.t : state.time.simMs);
  });

  return {
    inspect,
    isBlocked,
    block,
    unblock,
    reset,
    tick,
    /** Extra, read-only: a fresh Metrics snapshot without emitting it. */
    getMetrics: () => buildMetrics(),
    /** Extra, read-only: the PIWS queue (weakly labelled samples, research hypothesis). */
    getPiws: () => piws,
  };
}
