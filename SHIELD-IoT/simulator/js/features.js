// SHIELD-IoT preprocessing and ML evidence (layers 1 and 2 of the sensor).
//
// Layer 1, the conservative feature projection: every packet becomes a
// 37-value vector built from its own header fields plus per-source sliding
// windows. Identifiers (IP and MAC addresses, frame time) are deliberately
// left out: the research book's Edge-IIoTset audit found severe IP
// concentration (192.168.0.170 appears only in attack traffic) and a
// class-associated frame.time, so a model that sees them learns leakage.
// Not-applicable layers stay null (never zero-filled), so "this packet has no
// MQTT layer" can never be confused with "MQTT field equals 0".
//
// Layer 2, the "LightGBM surrogate": a deterministic, hand-weighted linear
// scorer with a softmax over the 15 Edge-IIoTset classes. It is a stand-in
// for the book's frozen LightGBM baseline (Macro F1 0.8872), not a trained
// model. Like the real baseline it is imperfect where the book says it is:
// Uploading and DDoS_HTTP are weak classes, so benign uploads, web bursts and
// camera RTP bursts occasionally put noticeable probability on a confusable
// class. All randomness comes from a seeded PRNG so runs are reproducible.

import { CLASSES, SUBNET } from './catalog.js';

// ---------------------------------------------------------------------------
// Feature schema

export const FEATURE_NAMES = [
  'frame.len',              //  0 frame length in bytes
  'proto.id',               //  1 highest-layer protocol (categorical, index into PROTOS)
  'flow.outbound',          //  2 LAN -> WAN
  'flow.inbound',           //  3 WAN -> LAN
  'src.bytes_rate_5s',      //  4 bytes per second sent by this source, 5 s window
  'tcp.flags.syn',          //  5
  'tcp.flags.ack',          //  6
  'tcp.flags.fin',          //  7
  'tcp.flags.rst',          //  8
  'tcp.flags.push',         //  9
  'tcp.len',                // 10 TCP payload length
  'tcp.connection.init',    // 11 first segment of a new connection (initiator side)
  'tcp.seq',                // 12 Step 4A: retain seq, ack, ack_raw and checksum
  'tcp.ack',                // 13
  'tcp.ack_raw',            // 14
  'tcp.checksum',           // 15
  'udp.length',             // 16
  'icmp.type',              // 17
  'arp.opcode',             // 18
  'arp.hw.size',            // 19
  'mqtt.msgtype',           // 20
  'mqtt.len',               // 21
  'mqtt.qos',               // 22
  'http.request.method',    // 23 ordinal (GET 1, POST 2, ...), null for responses
  'http.content_length',    // 24
  'http.uri.len',           // 25
  'dns.qry.name.len',       // 26
  'dns.qry.type',           // 27
  'mbtcp.func_code',        // 28
  'port.known_service',     // 29 1 if either port (or the identified protocol) is a known service
  'src.pps_1s',             // 30 packets from this source in the last 1 s
  'src.pps_5s',             // 31 packets per second from this source over 5 s
  'src.dst_hosts_5s',       // 32 distinct destination hosts in 5 s
  'src.dst_ports_5s',       // 33 distinct destination ports in 5 s
  'src.mean_len_5s',        // 34 mean frame length in 5 s
  'src.syn_ratio_5s',       // 35 SYN-without-ACK share of this source's TCP segments in 5 s
  'dst.pps_1s',             // 36 packets towards this destination in the last 1 s (fan-in)
];
export const N_FEATURES = FEATURE_NAMES.length; // 37, as in the frozen baseline

// Highest-layer protocols the packet model can carry; index = proto.id.
export const PROTOS = ['TCP', 'UDP', 'ICMP', 'ARP', 'MQTT', 'HTTP', 'HTTPS', 'DNS', 'NTP', 'MODBUS', 'RTP'];
const PROTO_INDEX = new Map(PROTOS.map((p, i) => [p, i]));
export const PROTO_OTHER = PROTOS.length;

/** Index of a protocol name in PROTOS, or PROTO_OTHER for anything unknown. */
export function protoIndex(proto) {
  return PROTO_INDEX.get(typeof proto === 'string' ? proto.toUpperCase() : proto) ?? PROTO_OTHER;
}

// Ports of services that ordinary home traffic uses (one side of almost every
// benign flow is one of these).
const KNOWN_PORTS = new Set([
  20, 21, 22, 23, 25, 53, 67, 68, 80, 110, 123, 143, 443, 502, 554, 853,
  1883, 5004, 5005, 5353, 5683, 8080, 8443, 8883,
]);
// Highest-layer protocols the sensor's dissector identified by itself.
const APP_PROTOS = new Set(['MQTT', 'HTTP', 'HTTPS', 'DNS', 'NTP', 'MODBUS', 'RTP']);

const HTTP_METHODS = { GET: 1, POST: 2, PUT: 3, DELETE: 4, HEAD: 5, OPTIONS: 6, PATCH: 7 };
const DNS_QTYPES = { A: 1, NS: 2, CNAME: 5, SOA: 6, PTR: 12, MX: 15, TXT: 16, AAAA: 28, SRV: 33, HTTPS: 65, ANY: 255 };

// ---------------------------------------------------------------------------
// Small utilities shared with ids.js

/** mulberry32: tiny, fast, seedable 32-bit PRNG returning floats in [0, 1). */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function next() {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard normal draws (Box-Muller) from a uniform generator. */
function makeGauss(rng) {
  let spare = null;
  return function gauss() {
    if (spare !== null) { const s = spare; spare = null; return s; }
    let u = 0;
    while (u <= 1e-12) u = rng();
    const v = rng();
    const r = Math.sqrt(-2 * Math.log(u));
    spare = r * Math.sin(2 * Math.PI * v);
    return r * Math.cos(2 * Math.PI * v);
  };
}

/** Array-backed FIFO with O(1) amortised shift; compacts itself occasionally. */
export class Deque {
  constructor() { this.items = []; this.head = 0; }
  get size() { return this.items.length - this.head; }
  push(x) { this.items.push(x); }
  peek() { return this.items[this.head]; }
  at(i) { return this.items[this.head + i]; }
  shift() {
    const x = this.items[this.head];
    this.items[this.head++] = undefined;
    if (this.head >= 1024 && this.head * 2 >= this.items.length) {
      this.items = this.items.slice(this.head);
      this.head = 0;
    }
    return x;
  }
  clear() { this.items = []; this.head = 0; }
}

/** Number or null. Strings such as "0x1a2b" or "42" are parsed (encoding audit). */
export function num(v) {
  if (v === null || v === undefined || v === '') return null;
  const n = typeof v === 'number' ? v : Number(v);
  return Number.isFinite(n) ? n : null;
}

/** 0/1 for boolean-like flag encodings (true, 1, "1", "true"). */
export function bit(v) {
  return v === true || v === 1 || v === '1' || v === 'true' || v === 'True' ? 1 : 0;
}

function ipToInt(ip) {
  if (typeof ip !== 'string') return null;
  const parts = ip.split('.');
  if (parts.length !== 4) return null;
  let n = 0;
  for (const s of parts) {
    const v = Number(s);
    if (!Number.isInteger(v) || v < 0 || v > 255) return null;
    n = n * 256 + v;
  }
  return n;
}

const [SUBNET_BASE, SUBNET_BITS] = SUBNET.split('/');
const SUBNET_SIZE = 2 ** (32 - Number(SUBNET_BITS));
const SUBNET_START = ipToInt(SUBNET_BASE);
const lanCache = new Map();

/** True when the address is inside the home subnet (192.168.0.0/24). */
export function isLanIp(ip) {
  let v = lanCache.get(ip);
  if (v === undefined) {
    const n = ipToInt(ip);
    v = n !== null && n >= SUBNET_START && n < SUBNET_START + SUBNET_SIZE;
    if (lanCache.size > 4096) lanCache.clear();
    lanCache.set(ip, v);
  }
  return v;
}

const clamp01 = v => (v < 0 ? 0 : v > 1 ? 1 : v);
/** Linear ramp: 0 at or below a, 1 at or above b. */
const ramp = (v, a, b) => clamp01((v - a) / (b - a));

// ---------------------------------------------------------------------------
// Per-source and per-destination sliding windows

const LONG_MS = 5000;
const SHORT_MS = 1000;

function inc(map, key) { map.set(key, (map.get(key) || 0) + 1); }
function dec(map, key) {
  const n = (map.get(key) || 0) - 1;
  if (n <= 0) map.delete(key); else map.set(key, n);
}

/** Everything one source sent in the last 5 s, with running aggregates. */
class SourceWindow {
  constructor() {
    this.long = new Deque();   // { t, host, port, len, tcp, syn }
    this.short = new Deque();  // timestamps, 1 s
    this.bytes = 0;
    this.tcpCount = 0;
    this.synCount = 0;
    this.hosts = new Map();    // host -> packets in window
    this.ports = new Map();    // port -> packets in window
    this.lastT = 0;
  }

  add(t, host, port, len, tcp, syn) {
    this.prune(t);
    this.long.push({ t, host, port, len, tcp, syn });
    this.short.push(t);
    this.bytes += len;
    if (tcp) this.tcpCount++;
    if (syn) this.synCount++;
    if (host) inc(this.hosts, host);
    if (port !== null) inc(this.ports, port);
    if (t > this.lastT) this.lastT = t;
  }

  prune(now) {
    const longCut = now - LONG_MS;
    while (this.long.size && this.long.peek().t <= longCut) {
      const e = this.long.shift();
      this.bytes -= e.len;
      if (e.tcp) this.tcpCount--;
      if (e.syn) this.synCount--;
      if (e.host) dec(this.hosts, e.host);
      if (e.port !== null) dec(this.ports, e.port);
    }
    const shortCut = now - SHORT_MS;
    while (this.short.size && this.short.peek() <= shortCut) this.short.shift();
  }
}

/** Timestamps of packets towards one destination over the last 1 s. */
class FanInWindow {
  constructor() { this.q = new Deque(); this.lastT = 0; }
  add(t) {
    const cut = t - SHORT_MS;
    while (this.q.size && this.q.peek() <= cut) this.q.shift();
    this.q.push(t);
    if (t > this.lastT) this.lastT = t;
  }
}

/**
 * Layer 1. extract(pkt, now) updates the windows with this packet and returns
 * the 37-value feature vector (plain array; null = not applicable).
 */
export function createFeatureExtractor() {
  const sources = new Map();   // source key (IP, else MAC) -> SourceWindow
  const dests = new Map();     // destination IP -> FanInWindow

  function extract(pkt, now) {
    const x = new Array(N_FEATURES).fill(null);
    const len = num(pkt.len) ?? 0;
    const proto = typeof pkt.proto === 'string' ? pkt.proto.toUpperCase() : '';
    x[0] = len;
    x[1] = protoIndex(proto);

    const srcLan = isLanIp(pkt.srcIp);
    const dstLan = isLanIp(pkt.dstIp);
    const srcWan = !srcLan && pkt.srcIp && pkt.srcIp !== '0.0.0.0';
    const dstWan = !dstLan && pkt.dstIp && pkt.dstIp !== '255.255.255.255';
    x[2] = srcLan && dstWan ? 1 : 0;
    x[3] = srcWan && dstLan ? 1 : 0;

    const tcp = pkt.tcp || null;
    let synOnly = false;
    if (tcp) {
      const f = tcp.flags || {};
      x[5] = bit(f.SYN); x[6] = bit(f.ACK); x[7] = bit(f.FIN); x[8] = bit(f.RST); x[9] = bit(f.PSH);
      x[10] = num(tcp.len);
      x[11] = tcp.connInit === undefined || tcp.connInit === null ? null : bit(tcp.connInit);
      x[12] = num(tcp.seq); x[13] = num(tcp.ack); x[14] = num(tcp.ackRaw); x[15] = num(tcp.checksum);
      synOnly = x[5] === 1 && x[6] === 0;
    }
    if (pkt.udp) x[16] = num(pkt.udp.len);
    if (pkt.icmp) x[17] = num(pkt.icmp.type);
    if (pkt.arp) { x[18] = num(pkt.arp.opcode); x[19] = num(pkt.arp.hwSize); }
    if (pkt.mqtt) { x[20] = num(pkt.mqtt.msgtype); x[21] = num(pkt.mqtt.len); x[22] = num(pkt.mqtt.qos); }
    if (pkt.http) {
      const m = pkt.http.method;
      x[23] = m ? (HTTP_METHODS[String(m).toUpperCase()] ?? 8) : null;
      x[24] = num(pkt.http.bodyLen);
      x[25] = typeof pkt.http.uri === 'string' ? pkt.http.uri.length : null;
    }
    if (pkt.dns) {
      // The audit found strings in dns.qry.name.len: parse, fall back to the name itself.
      x[26] = num(pkt.dns.qnameLen) ?? (typeof pkt.dns.qname === 'string' ? pkt.dns.qname.length : null);
      const qt = pkt.dns.qtype;
      x[27] = typeof qt === 'string' ? (DNS_QTYPES[qt.toUpperCase()] ?? num(qt)) : num(qt);
    }
    if (pkt.modbus) x[28] = num(pkt.modbus.fn);

    const sport = num(pkt.sport);
    const dport = num(pkt.dport);
    if (sport !== null || dport !== null) {
      x[29] = APP_PROTOS.has(proto) || KNOWN_PORTS.has(sport) || KNOWN_PORTS.has(dport) ? 1 : 0;
    }

    // Per-source windows. ARP who-has sweeps are counted by the address asked about.
    const key = pkt.srcIp || pkt.srcMac || '?';
    let w = sources.get(key);
    if (!w) { w = new SourceWindow(); sources.set(key, w); }
    const host = pkt.arp ? (pkt.arp.targetIp || pkt.dstIp || null) : (pkt.dstIp || null);
    w.add(now, host, dport, len, !!tcp, synOnly);
    const n5 = w.long.size;
    x[4] = w.bytes / (LONG_MS / 1000);
    x[30] = w.short.size;
    x[31] = n5 / (LONG_MS / 1000);
    x[32] = w.hosts.size;
    x[33] = w.ports.size;
    x[34] = n5 ? w.bytes / n5 : null;
    x[35] = w.tcpCount ? w.synCount / w.tcpCount : null;

    if (pkt.dstIp) {
      let d = dests.get(pkt.dstIp);
      if (!d) { d = new FanInWindow(); dests.set(pkt.dstIp, d); }
      d.add(now);
      x[36] = d.q.size;
    }
    return x;
  }

  /** Forget sources and destinations that have been silent for over 10 s. */
  function sweep(now) {
    const cut = now - 2 * LONG_MS;
    for (const [k, w] of sources) if (w.lastT < cut) sources.delete(k);
    for (const [k, d] of dests) if (d.lastT < cut) dests.delete(k);
  }

  /** Current 5 s packet rate of a source (used to balance the drift sample). */
  function sourceRate(key) {
    const w = sources.get(key);
    return w ? w.long.size / (LONG_MS / 1000) : 0;
  }

  function reset() { sources.clear(); dests.clear(); }

  return { extract, sweep, sourceRate, reset };
}

// ---------------------------------------------------------------------------
// Layer 2: the LightGBM surrogate

const N_CLASSES = CLASSES.length;
const C = Object.fromEntries(CLASSES.map((c, i) => [c, i]));

// Class priors (logit biases). Normal dominates everyday traffic.
const BIAS = new Float64Array(N_CLASSES).fill(-2.3);
BIAS[C.Normal] = 4.2;
BIAS[C.Backdoor] = -2.4;
BIAS[C.DDoS_HTTP] = -1.6;       // weak class in the book (F1 0.72): sits closer to Normal
BIAS[C.DDoS_ICMP] = -2.6;
BIAS[C.DDoS_TCP] = -2.2;
BIAS[C.DDoS_UDP] = -2.0;
BIAS[C.MITM] = -2.8;
BIAS[C.Password] = -2.2;
BIAS[C.Ransomware] = -2.7;
BIAS[C.SQL_injection] = -2.1;
BIAS[C.Uploading] = -1.6;       // weak class in the book (F1 0.71)
BIAS[C.XSS] = -2.2;

const JITTER_NORMAL = 0.08;     // small score jitter: tree ensembles are not perfectly smooth
const JITTER_ATTACK = 0.18;

// Confusion episodes: on a benign flow whose aggregate features resemble a
// weak class, the surrogate now and then lands in a confusable region and puts
// noticeable probability on that class. The rate is per second of such a flow
// (not per packet), so a busier flow does not produce proportionally more
// false alarms. About 3 in 8 episodes cross the default alert threshold.
const CONFUSION_PER_SEC = 0.02;
const CONFUSION_DELTA = [-0.8, 0.8];   // logit of the confusable class relative to Normal

/**
 * Layer 2. score(x) returns
 * { probs, top, topProb, pNormal, attack, attackProb, suspicion, episode }
 * where suspicion = 1 - P(Normal) and attack is the most likely non-Normal class.
 */
export function createMLSurrogate(rng) {
  const gauss = makeGauss(rng);

  function score(x) {
    const z = Float64Array.from(BIAS);
    const len = x[0] ?? 0;
    const protoName = PROTOS[x[1]] ?? 'OTHER';

    const tcp = x[5] !== null, udp = x[16] !== null, icmp = x[17] !== null, arp = x[18] !== null;
    const mqtt = x[20] !== null;
    const httpReq = x[23] !== null;
    const http = httpReq || x[24] !== null || x[25] !== null;
    const dns = x[26] !== null || x[27] !== null;
    const modbus = x[28] !== null;
    const app = mqtt || http || dns || modbus;
    const dnsNtp = dns || protoName === 'NTP';
    const synOnly = x[5] === 1 && x[6] === 0;
    const out = x[2] === 1, inbound = x[3] === 1, lanToLan = !out && !inbound;
    const known = x[29] === 1, unknownPort = x[29] === 0;
    const web = tcp && (http || protoName === 'HTTP' || protoName === 'HTTPS');
    const pps1 = x[30] ?? 0, pps5 = x[31] ?? 0, hosts = x[32] ?? 0, ports = x[33] ?? 0;
    const synR = x[35] ?? 0, fanIn = x[36] ?? 0, bps = x[4] ?? 0;
    const big = ramp(len, 900, 1450);
    const small = 1 - ramp(len, 60, 200);
    const uriLen = x[25] ?? 0;

    // Evidence for Normal: well-formed application traffic on known services.
    z[C.Normal] += (app ? 0.5 : 0) + (known ? 0.35 : 0) + (mqtt ? 0.3 : 0);

    if (tcp && unknownPort) {
      z[C.Backdoor] += 2.8 + 2.5 * ramp(pps5, 2, 20);
      z[C.Ransomware] += 3 * big * ramp(bps, 50e3, 500e3);
    }
    if (web) z[C.DDoS_HTTP] += 2.2 * ramp(pps1, 25, 200) * (0.5 + 0.5 * small) + 1.2 * ramp(fanIn, 60, 300);
    if (httpReq) {
      z[C.DDoS_HTTP] += 4.5 * ramp(pps1, 8, 50);
      z[C.SQL_injection] += 4.5 * ramp(uriLen, 90, 260);
      z[C.XSS] += 4.0 * ramp(uriLen, 70, 220);
      z[C.Vulnerability_scanner] += 4 * ramp(ports, 4, 20) + 2.5 * ramp(pps1, 10, 60) + ramp(uriLen, 50, 150);
    }
    if (x[23] === 2) z[C.Uploading] += 3 * ramp(x[24] ?? 0, 2e5, 5e6);   // large POST bodies
    if (icmp) {
      z[C.DDoS_ICMP] += 7 * ramp(pps1, 8, 60) + ramp(fanIn, 20, 100);
      if (x[17] === 8) z[C.Fingerprinting] += ramp(hosts, 3, 15);      // echo-request sweeps
    }
    if (icmp || arp) z[C.Fingerprinting] += 3.2 * ramp(hosts, 8, 40);
    if (arp && x[18] === 2) z[C.MITM] += 6 * ramp(pps1, 3, 25);          // reply storms
    if (tcp) {
      z[C.DDoS_TCP] += 7.5 * ramp(synR, 0.4, 0.9) * ramp(pps1, 30, 150) + (synOnly ? ramp(pps1, 10, 60) : 0);
      if (ports <= 2) z[C.Password] += 4.5 * ramp(synR, 0.25, 0.6) * ramp(pps5, 4, 30);
      if (!app) z[C.Port_Scanning] += 7 * ramp(ports, 10, 50) * ramp(synR, 0.3, 0.8) + (synOnly ? 1.5 * ramp(ports, 5, 25) : 0);
      if (out || lanToLan) z[C.Uploading] += 3 * big * ramp(bps, 20e3, 150e3);
    }
    if (udp) z[C.DDoS_UDP] += (4 * ramp(pps1, 80, 400) + 1.5 * ramp(fanIn, 120, 500)) * (dnsNtp ? 0.5 : 1);

    // Jitter.
    z[0] += JITTER_NORMAL * gauss();
    for (let k = 1; k < N_CLASSES; k++) z[k] += JITTER_ATTACK * gauss();

    // Confusion episodes on the book's weak classes (plus UDP floods for RTP bursts).
    let episode = false;
    const confUp = tcp && (out || lanToLan) ? big * ramp(bps, 10e3, 60e3) : 0;
    const confHttp = web && out ? ramp(pps1, 10, 40) : 0;
    const confUdp = udp && !dnsNtp ? ramp(len, 500, 1100) * ramp(pps1, 10, 40) : 0;
    const confTotal = confUp + confHttp + confUdp;
    if (confTotal > 0) {
      const p = CONFUSION_PER_SEC * Math.min(1, confTotal) / Math.max(1, pps1);
      if (rng() < p) {
        const pick = rng() * confTotal;
        const k = pick < confUp ? C.Uploading : pick < confUp + confHttp ? C.DDoS_HTTP : C.DDoS_UDP;
        z[k] = z[0] + CONFUSION_DELTA[0] + (CONFUSION_DELTA[1] - CONFUSION_DELTA[0]) * rng();
        episode = true;
      }
    }

    // Softmax.
    let max = -Infinity;
    for (let k = 0; k < N_CLASSES; k++) if (z[k] > max) max = z[k];
    let sum = 0;
    for (let k = 0; k < N_CLASSES; k++) { z[k] = Math.exp(z[k] - max); sum += z[k]; }
    let top = 0, attack = 1;
    for (let k = 0; k < N_CLASSES; k++) {
      z[k] /= sum;
      if (z[k] > z[top]) top = k;
      if (k > 0 && z[k] > z[attack]) attack = k;
    }
    return {
      probs: z,
      top: CLASSES[top],
      topProb: z[top],
      pNormal: z[0],
      attack: CLASSES[attack],
      attackProb: z[attack],
      suspicion: 1 - z[0],
      episode,
    };
  }

  return { score };
}
