// Everyday (benign) traffic of the smart home.
//
// This module knows what packets look like and when each device sends them:
// packet builders for every layer in the CONTRACT.md packet shape, TCP
// sessions with real handshakes and sequence numbers, MQTT sessions to the
// hub, per-device schedules, and the packet sequences that device commands
// produce. It never routes anything itself: network.js hands in a small `net`
// interface (clock-driven event queue, router + firewall + IDS pipeline, rate
// budget, power state) and owns device props and status.
//
// Determinism: every random choice comes from the seeded PRNG passed in by
// network.js (mulberry32), never Math.random, so a run replays exactly.

import { GATEWAY_IP } from './catalog.js';

export const SEED = 0x5d1e1d07;

/** mulberry32 PRNG plus the distributions the traffic model needs. */
export function createRng(seed = SEED) {
  let a = seed >>> 0;
  const next = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return {
    next,
    range: (lo, hi) => lo + (hi - lo) * next(),
    int: (lo, hi) => lo + Math.floor(next() * (hi - lo + 1)),
    pick: list => list[Math.floor(next() * list.length)],
    chance: p => next() < p,
    // Exponential gap of a Poisson process, floored so events never collapse.
    exp: (mean, floor = 0.15) => Math.max(mean * floor, -mean * Math.log(1 - next())),
    jitter: (mean, spread) => mean + (next() * 2 - 1) * spread,
    u32: () => (next() * 4294967296) >>> 0,
  };
}

// ---------------------------------------------------------------------------
// Static tables

const BROADCAST_MAC = 'ff:ff:ff:ff:ff:ff';
const ZERO_MAC = '00:00:00:00:00:00';
const ETH_IP = 34;   // Ethernet II (14) + IPv4 (20) header bytes in front of L4
const PORT = { DNS: 53, NTP: 123, HTTP: 80, HTTPS: 443, MQTT: 1883, MODBUS: 502, HASS: 8123, RTP: 5004 };

// Device types that keep an MQTT session with the hub's broker.
const MQTT_TYPES = new Set(['tv', 'speaker', 'light', 'ac', 'clock', 'phone', 'vacuum', 'fridge', 'plug', 'fan',
  'watch', 'lock', 'thermostat', 'camera', 'washer']);

// TCP/IP stack per device type: drives ephemeral ports and window sizes.
const STACK = {
  router: 'linux', shield: 'linux', hub: 'linux', speaker: 'linux', vacuum: 'linux', camera: 'linux',
  laptop: 'linux', rogue: 'linux', tv: 'android', phone: 'android', watch: 'android', cloud: 'cloud',
  light: 'lwip', ac: 'lwip', clock: 'lwip', fridge: 'lwip', plug: 'lwip', fan: 'lwip', lock: 'lwip',
  thermostat: 'lwip', washer: 'lwip', meter: 'lwip',
};
const WINDOWS = {
  linux: { syn: 64240, est: [501, 2052] },
  android: { syn: 65535, est: [1024, 4096] },
  lwip: { syn: 5744, est: [2872, 5744] },
  cloud: { syn: 65160, est: [1040, 8192] },
};

const onOff = v => (v ? 'ON' : 'OFF');

// MQTT telemetry: publish period (ms, from current props) and the JSON body.
const TELEMETRY = {
  light: { every: () => 20000, body: p => ({ state: onOff(p.power), brightness: p.brightness, color: p.color }) },
  fan: { every: () => 15000, body: p => ({ state: onOff(p.power), speed: p.speed }) },
  ac: { every: () => 6000, body: (p, s) => ({ state: onOff(p.power), mode: p.mode, setpoint: p.temp, room: s.devices.get('thermostat')?.props.current ?? null }) },
  plug: { every: () => 4000, body: p => ({ state: onOff(p.power), watts: p.watts }) },
  fridge: { every: p => (p.doorOpen ? 3000 : 12000), body: p => ({ temp: p.temp, door: p.doorOpen ? 'open' : 'closed' }) },
  washer: { every: p => (p.running ? 3000 : 30000), body: p => ({ program: p.program, running: p.running, remainingMin: p.remainingMin }) },
  thermostat: { every: () => 10000, body: p => ({ current: p.current, target: p.target }) },
  lock: { every: () => 30000, body: p => ({ locked: p.locked, battery: 87 }) },
  vacuum: { every: p => (p.running ? 2000 : 15000), body: p => ({ state: p.running ? 'cleaning' : p.battery < 100 ? 'charging' : 'docked', battery: p.battery }) },
  watch: { every: () => 5000, body: p => ({ heartRate: p.heartRate, steps: p.steps }) },
};

const HOSTS = {
  tv: { 'Cricket Live': 'live-sports.akamaized.net', News: 'news-live.akamaized.net', Movies: 'vod-movies.cloudfront.net', Cartoons: 'kids-vod.cloudfront.net' },
  tvDns: ['i.ytimg.com', 'play.googleapis.com', 'time.android.com', 'epg.tvguide-cdn.net'],
  speaker: 'avs-alexa-eu.amazon.com',
  music: 'music-stream.cloudfront.net',
  hub: 'remote.nabu.casa',
  ntp: 'in.pool.ntp.org',
  laptop: ['github.com', 'scholar.google.com', 'ieee-dataport.org', 'www.kaggle.com', 'pypi.org', 'docs.python.org',
    'stackoverflow.com', 'lightgbm.readthedocs.io', 'nith.ac.in'],
  phone: ['www.google.com', 'mail.google.com', 'web.whatsapp.com', 'maps.googleapis.com', 'www.youtube.com', 'news.google.com'],
  vacuum: 'api.robovac-cloud.com',
  watch: 'fit.googleapis.com',
  fridge: 'api.smartfridge-cloud.com',
  thermostat: 'api.openweathermap.org',
  rogue: ['archive.raspberrypi.com', 'pool.ntp.org', 'deb.debian.org', 'api.github.com'],
};

// Occasional large, benign HTTP uploads (the book's weak Uploading class).
const UPLOADS = {
  laptop: {
    host: 'files.nith.ac.in', mb: [2, 24], ua: 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:131.0) Gecko/20100101 Firefox/131.0',
    uri: n => `/upload/thesis/shield-iot-results-${n}.tar.gz`,
  },
  phone: {
    host: 'backup.photos-cloud.app', mb: [1.5, 6], ua: 'okhttp/4.12.0',
    uri: n => `/v1/upload/IMG_20261002_19${String(40 + (n % 20)).padStart(2, '0')}.jpg`,
  },
};

const QTYPE_NAME = { 1: 'A', 28: 'AAAA' };
const FLAG_ORDER = ['SYN', 'FIN', 'RST', 'PSH', 'ACK', 'URG'];
const FLAG_LETTER = { S: 'SYN', F: 'FIN', R: 'RST', P: 'PSH', A: 'ACK', U: 'URG' };

function flagsOf(spec) {
  const f = { SYN: false, ACK: false, FIN: false, RST: false, PSH: false, URG: false };
  for (const ch of spec) f[FLAG_LETTER[ch]] = true;
  return f;
}

const trunc = (s, n = 96) => (s.length > n ? `${s.slice(0, n - 1)}…` : s);
const hex4 = v => v.toString(16).padStart(4, '0');

// ---------------------------------------------------------------------------

/**
 * createTraffic({ state, rng, net }) builds the traffic engine for the current
 * state.devices. `net` is supplied by network.js:
 *   at(t, fn)       run fn at sim time t (ms)
 *   send(pkt)       router firewall → IDS → delivery; false when the packet was dropped or suppressed
 *   budget(n)       true when the rate cap has room for about n more packets
 *   powered(id)     device has power / Wi-Fi / is plugged in (ignores router state and blocks)
 *   blocked(id)     device is blocked at the router firewall
 */
export function createTraffic({ state, rng, net }) {
  const dev = id => state.devices.get(id);
  const has = id => state.devices.has(id);
  const ipPrefix = GATEWAY_IP.slice(0, GATEWAY_IP.lastIndexOf('.') + 1);
  // A device takes part in everyday traffic when it is powered; the unregistered
  // board additionally stops once SHIELD-IoT has blocked it.
  const up = id => has(id) && net.powered(id) && (dev(id).role !== 'unknown' || !net.blocked(id));

  let packetSeq = 0;
  let uploadCount = 0;
  let pingSeq = 0;
  let modbusTxn = rng.int(1, 4000);
  let ntpCount = 0;
  let sweepNext = 1;

  // ---- packets ---------------------------------------------------------------

  const stackOf = id => STACK[dev(id).type] || 'linux';
  const macOf = id => dev(id).mac || dev('router').mac;   // WAN hosts appear behind the gateway's MAC
  const lanDelay = () => rng.int(2, 6);
  const wanDelay = () => rng.int(22, 48);
  const hop = (a, b) => (a === 'cloud' || b === 'cloud' ? wanDelay() : lanDelay());
  const ephemeral = id => (stackOf(id) === 'lwip' ? rng.int(49152, 65535) : rng.int(32768, 60999));

  function windowFor(id, syn) {
    const w = WINDOWS[stackOf(id)];
    return syn ? w.syn : rng.int(w.est[0], w.est[1]);
  }

  /** Base packet: every layer null, label from the source's registration. */
  function packet(t, src, dst, proto, fields) {
    const s = dev(src), d = dev(dst);
    return Object.assign({
      id: ++packetSeq, t, src, dst,
      srcIp: s.ip, dstIp: d.ip, srcMac: macOf(src), dstMac: macOf(dst),
      proto, l4: null, sport: null, dport: null, len: 60,
      tcp: null, udp: null, icmp: null, arp: null, mqtt: null, http: null, dns: null, modbus: null,
      payload: '', label: s.role === 'unknown' ? 'Unregistered' : 'Normal',
    }, fields);
  }

  function udp(t, src, dst, sport, dport, proto, bytes, fields) {
    return packet(t, src, dst, proto, { l4: 'udp', sport, dport, len: ETH_IP + 8 + bytes, udp: { len: 8 + bytes }, ...fields });
  }

  function icmpEcho(t, src, dst, type, seq) {
    return packet(t, src, dst, 'ICMP', {
      l4: 'icmp', len: 98, icmp: { type, code: 0 },
      payload: `Echo (ping) ${type === 8 ? 'request' : 'reply'} id=0x0001 seq=${seq}`,
    });
  }

  /** ARP who-has. dst is the device owning targetIp, or the router for a broadcast nobody answers. */
  function arpRequest(t, sender, targetIp, gratuitous = false) {
    const s = dev(sender);
    const owner = state.byIp.get(targetIp);
    const dst = owner && owner !== sender && owner !== 'cloud' ? owner : 'router';
    return packet(t, sender, dst, 'ARP', {
      dstIp: targetIp, dstMac: BROADCAST_MAC, len: 42,
      arp: { opcode: 1, hwSize: 6, protoSize: 4, senderMac: s.mac, senderIp: s.ip, targetMac: ZERO_MAC, targetIp },
      payload: gratuitous ? `Gratuitous ARP for ${s.ip} (announcement)` : `Who has ${targetIp}? Tell ${s.ip}`,
    });
  }

  function arpReply(t, replier, asker) {
    const r = dev(replier), a = dev(asker);
    return packet(t, replier, asker, 'ARP', {
      len: 42,
      arp: { opcode: 2, hwSize: 6, protoSize: 4, senderMac: r.mac, senderIp: r.ip, targetMac: a.mac, targetIp: a.ip },
      payload: `${r.ip} is at ${r.mac}`,
    });
  }

  // ---- TCP sessions ------------------------------------------------------------

  const live = new Set();   // every TCP session that has not been closed or killed
  const held = new Map();   // long-lived sessions by role: 'mqtt:<id>', 'tls:tv', 'modbus', ...

  class Session {
    constructor(client, server, port, { kind = 'tcp', established = false } = {}) {
      this.client = client;
      this.server = server;
      this.kind = kind;
      this.cport = ephemeral(client);
      this.sport = port;
      this.cIsn = rng.u32();
      this.sIsn = rng.u32();
      // Relative sequence numbers (as Wireshark shows them); a session that was
      // already running when the simulation starts is somewhere mid-stream.
      this.cNext = established ? 1 + rng.int(200, 50000) : 0;
      this.sNext = established ? 1 + rng.int(200, 500000) : 0;
      this.ready = established;                    // handshake (and TLS hello) completed
      this.mqttReady = established && kind === 'mqtt';
      this.pid = rng.int(1, 4000);   // MQTT packet identifier
      this.dead = false;
      live.add(this);
    }
    involves(id) { return this.client === id || this.server === id; }
    nextPid() { this.pid = (this.pid % 65535) + 1; return this.pid; }
    kill() { this.dead = true; live.delete(this); }
  }

  /** One TCP segment on session s; advances the sender's sequence number. */
  function segment(t, s, fromClient, spec, o = {}) {
    const f = flagsOf(spec);
    const { data = 0, proto = 'TCP', payload = null, ...layers } = o;
    const src = fromClient ? s.client : s.server;
    const dst = fromClient ? s.server : s.client;
    const seq = fromClient ? s.cNext : s.sNext;
    const ack = f.ACK ? (fromClient ? s.sNext : s.cNext) : 0;
    const ackRaw = f.ACK ? ((fromClient ? s.sIsn : s.cIsn) + ack) >>> 0 : 0;
    const used = data + (f.SYN ? 1 : 0) + (f.FIN ? 1 : 0);
    if (fromClient) s.cNext += used; else s.sNext += used;
    const sport = fromClient ? s.cport : s.sport;
    const dport = fromClient ? s.sport : s.cport;
    const window = windowFor(src, f.SYN);
    return packet(t, src, dst, proto, {
      l4: 'tcp', sport, dport, len: ETH_IP + (f.SYN ? 40 : 32) + data,
      tcp: { flags: f, len: data, seq, ack, ackRaw, checksum: rng.int(0, 0xffff), window, connInit: f.SYN && !f.ACK && fromClient },
      ...layers,
      payload: payload || `${sport} → ${dport} [${FLAG_ORDER.filter(k => f[k]).join(', ')}] Seq=${seq}${f.ACK ? ` Ack=${ack}` : ''} Win=${window} Len=${data}`,
    });
  }

  // ---- flows -----------------------------------------------------------------

  /**
   * A conversation: packets scheduled at increasing sim times. When a packet
   * is dropped (firewall, IDS, router offline) the rest of the conversation
   * never happens, and the TCP sessions it used are torn down, so the next
   * attempt starts with a fresh handshake.
   */
  class Flow {
    constructor(t, guard) {
      this.t = t;
      this.guard = guard;
      this.dead = false;
      this.owned = new Set();   // sessions opened by this flow: killed on any abort
      this.used = new Set();    // sessions written by this flow: killed when a packet is dropped
      this.hooks = [];
    }
    wait(ms) { this.t += ms; return this; }
    send(delay, build) {
      const at = (this.t += delay);
      net.at(at, () => {
        if (this.dead) return;
        if (this.guard && !this.guard()) { this.abort('guard'); return; }
        const pkt = build(at);
        if (!pkt) { this.abort('stale'); return; }
        if (!net.send(pkt)) this.abort('dropped');
      });
      return this;
    }
    seg(delay, s, fromClient, flags, spec) {
      this.used.add(s);
      return this.send(delay, t => (s.dead ? null : segment(t, s, fromClient, flags, typeof spec === 'function' ? spec() : spec)));
    }
    then(delay, fn) {
      const at = (this.t += delay);
      net.at(at, () => { if (!this.dead) fn(at); });
      return this;
    }
    own(s) { this.owned.add(s); return s; }
    release(s) { this.owned.delete(s); }
    onAbort(fn) { this.hooks.push(fn); return this; }
    abort(reason) {
      if (this.dead) return;
      this.dead = true;
      for (const s of this.owned) s.kill();
      if (reason === 'dropped') for (const s of this.used) s.kill();
      for (const fn of this.hooks) fn(reason);
    }
  }

  /** Everyday flow: stops as soon as one of the endpoints loses power or link. */
  const flow = (t, ...ids) => new Flow(t, () => ids.every(id => up(id)));
  /** Command flow: runs to the end (a device being switched off still says goodbye). */
  const commandFlow = t => new Flow(t, null);

  function handshake(f, s) {
    f.seg(0, s, true, 'S');
    f.seg(hop(s.client, s.server), s, false, 'SA');
    f.seg(1, s, true, 'A');
  }

  function close(f, s, byClient = true) {
    f.seg(1, s, byClient, 'FA');
    f.seg(hop(s.client, s.server), s, !byClient, 'FA');
    f.seg(1, s, byClient, 'A');
    f.then(0, () => s.kill());
  }

  function tlsHello(f, s, sni) {
    f.seg(1, s, true, 'PA', { data: rng.int(512, 583), proto: 'HTTPS', payload: `TLS 1.3 Client Hello (SNI ${sni})` });
    f.seg(hop(s.client, s.server), s, false, 'A', { data: 1448, proto: 'HTTPS', payload: 'TLS 1.3 Server Hello, Change Cipher Spec, Encrypted Extensions' });
    f.seg(1, s, false, 'PA', { data: rng.int(700, 1300), proto: 'HTTPS', payload: 'TLS 1.3 Certificate, Certificate Verify, Finished' });
    f.seg(1, s, true, 'PA', { data: 80, proto: 'HTTPS', payload: 'TLS 1.3 Change Cipher Spec, Finished' });
  }

  function tlsData(f, s, fromClient, bytes, what, delay = 1) {
    f.seg(delay, s, fromClient, 'PA', { data: bytes, proto: 'HTTPS', payload: `TLS 1.3 Application Data (${what}, ${bytes} B)` });
  }

  /** Server streams n segments of application data; the client ACKs every second one. */
  function tlsDownload(f, s, n, what, { min = 1134, max = 1434, first = null } = {}) {
    for (let i = 0; i < n; i++) {
      const delay = i === 0 ? (first ?? hop(s.client, s.server)) : rng.int(2, 7);
      tlsData(f, s, false, rng.int(min, max), what, delay);
      if (i % 2 === 1 || i === n - 1) f.seg(1, s, true, 'A');
    }
  }

  /** A long-lived TLS session held under `key`; opens one (DNS, handshake, hello) inside f if needed. */
  function heldTls(f, key, client, host, server = 'cloud') {
    let s = held.get(key);
    if (s && !s.dead && s.server === server) {
      if (!s.ready) f.wait(250);   // another flow is still opening it
      return s;
    }
    if (s && !s.dead) { held.delete(key); close(f, s, true); }   // switching servers (TV dashboard channel)
    s = f.own(new Session(client, server, PORT.HTTPS, { kind: 'tls' }));
    held.set(key, s);
    if (server === 'cloud') dns(f, client, host);
    f.wait(2);
    handshake(f, s);
    tlsHello(f, s, host);
    f.then(0, () => { s.ready = true; f.release(s); });
    return s;
  }

  // ---- DNS, NTP, ARP, ICMP ---------------------------------------------------------

  /** DNS lookup through the router's resolver; ~30 % are cache misses forwarded upstream. */
  function dns(f, id, qname, qtype = 1) {
    const sport = ephemeral(id);
    const q = 12 + qname.length + 2 + 4;
    const answer = qtype === 28 ? 28 : 16;
    const layer = () => ({ qname, qtype, qnameLen: qname.length });
    const name = QTYPE_NAME[qtype] || String(qtype);
    f.send(0, t => udp(t, id, 'router', sport, PORT.DNS, 'DNS', q, { dns: layer(), payload: `Standard query ${name} ${qname}` }));
    if (rng.chance(0.3)) {
      const up2 = ephemeral('router');
      f.send(1, t => udp(t, 'router', 'cloud', up2, PORT.DNS, 'DNS', q, { dns: layer(), payload: `Standard query ${name} ${qname} (forwarded upstream)` }));
      f.send(wanDelay(), t => udp(t, 'cloud', 'router', PORT.DNS, up2, 'DNS', q + answer, { dns: layer(), payload: `Standard query response ${name} ${qname}` }));
    }
    const ans = qtype === 28 ? 'AAAA 2600:1f18:2148:bc00::1' : `A ${dev('cloud').ip}`;
    f.send(1, t => udp(t, 'router', id, PORT.DNS, sport, 'DNS', q + answer, { dns: layer(), payload: trunc(`Standard query response ${name} ${qname} ${ans}`) }));
  }

  function ntp(f, id) {
    const sport = ephemeral(id);
    f.send(0, t => udp(t, id, 'cloud', sport, PORT.NTP, 'NTP', 48, { payload: 'NTP v4 client request (mode 3)' }));
    f.send(wanDelay(), t => udp(t, 'cloud', id, PORT.NTP, sport, 'NTP', 48, { payload: 'NTP v4 server reply (mode 4, stratum 2)' }));
  }

  function arpGateway(t, id) {
    const f = flow(t, id, 'router');
    f.send(0, at => arpRequest(at, id, GATEWAY_IP));
    f.send(lanDelay(), at => arpReply(at, 'router', id));
  }

  /** The router refreshes its ARP cache for one LAN host. */
  function routerProbe(t) {
    const hosts = [...state.devices.values()].filter(d => d.role !== 'cloud' && d.role !== 'unknown' && d.id !== 'router' && up(d.id));
    if (!hosts.length) return;
    const target = rng.pick(hosts).id;
    const f = flow(t, 'router', target);
    f.send(0, at => arpRequest(at, 'router', dev(target).ip));
    f.send(lanDelay(), at => arpReply(at, target, 'router'));
  }

  /** Home Assistant's ping presence sensor checks the phone, laptop or TV. */
  function hubPing(t) {
    const targets = ['phone', 'laptop', 'tv'].filter(up);
    if (!targets.length) return;
    const target = rng.pick(targets);
    const seq = ++pingSeq;
    const f = flow(t, 'hub', target);
    f.send(0, at => icmpEcho(at, 'hub', target, 8, seq));
    f.send(lanDelay(), at => icmpEcho(at, target, 'hub', 0, seq));
  }

  // ---- MQTT --------------------------------------------------------------------------

  const mqttClients = new Set([...state.devices.values()].filter(d => MQTT_TYPES.has(d.type)).map(d => d.id));
  const mqttKey = id => `mqtt:${id}`;

  function mqttSession(id) {
    const s = held.get(mqttKey(id));
    return s && !s.dead ? s : null;
  }

  function mqttText(type, o) {
    switch (type) {
      case 1: return `CONNECT client=${o.clientId} keepalive=60 flags=0x${o.conflags.toString(16).padStart(2, '0')}`;
      case 2: return 'CONNACK session-present=0 rc=0 (accepted)';
      case 3: return `PUBLISH ${o.topic}${o.qos ? ` q${o.qos} id=${o.pid}` : ''}${o.retain ? ' retain' : ''} ${o.body}`;
      case 4: return `PUBACK id=${o.pid}`;
      case 8: return `SUBSCRIBE ${o.topic} (QoS 1)`;
      case 9: return 'SUBACK granted QoS 1';
      case 12: return 'PINGREQ';
      case 13: return 'PINGRESP';
      case 14: return 'DISCONNECT';
      default: return `MQTT type ${type}`;
    }
  }

  /** Segment options for one MQTT control packet (sizes follow MQTT 3.1.1 encoding). */
  function mqtt(msgtype, o = {}) {
    const topic = o.topic || '';
    const body = o.body || '';
    const qos = o.qos || 0;
    let rem;
    switch (msgtype) {
      case 1: rem = 10 + 2 + o.clientId.length + (o.conflags & 0x80 ? 2 + o.user.length : 0) + (o.conflags & 0x40 ? 2 + 16 : 0); break;
      case 3: rem = 2 + topic.length + (qos ? 2 : 0) + body.length; break;
      case 8: rem = 2 + 2 + topic.length + 1; break;
      case 9: rem = 3; break;
      case 2: case 4: rem = 2; break;
      default: rem = 0;   // PINGREQ, PINGRESP, DISCONNECT
    }
    const bytes = 1 + (rem < 128 ? 1 : rem < 16384 ? 2 : 3) + rem;
    return {
      data: bytes,
      proto: 'MQTT',
      mqtt: {
        msgtype,
        conflags: msgtype === 1 ? o.conflags : 0,
        qos: msgtype === 8 ? 1 : qos,            // SUBSCRIBE's fixed-header flags are 0b0010
        retain: !!o.retain,
        topic: msgtype === 3 || msgtype === 8 ? topic : '',
        len: rem,
      },
      payload: trunc(mqttText(msgtype, { ...o, topic, body, qos })),
    };
  }

  /** TCP handshake, CONNECT/CONNACK and SUBSCRIBE home/<id>/set, appended to f. */
  function connect(f, id) {
    const s = f.own(new Session(id, 'hub', PORT.MQTT, { kind: 'mqtt' }));
    held.set(mqttKey(id), s);
    const d = dev(id);
    const clientId = `${d.type}-${d.mac.slice(-5).replace(':', '')}`;
    const conflags = d.type === 'light' ? 0x02 : 0xc2;   // clean session (+ username and password)
    handshake(f, s);
    f.seg(2, s, true, 'PA', mqtt(1, { clientId, conflags, user: 'mqtt-home' }));
    f.seg(lanDelay(), s, false, 'PA', mqtt(2));
    f.seg(2, s, true, 'PA', mqtt(8, { topic: `home/${id}/set`, pid: s.nextPid() }));
    f.seg(lanDelay(), s, false, 'PA', mqtt(9));
    f.then(0, () => { s.ready = true; s.mqttReady = true; f.release(s); });
    return s;
  }

  /** The device's MQTT session, connecting inside f when there is none. */
  function ensureMqtt(f, id) {
    const s = mqttSession(id);
    if (!s) return connect(f, id);
    if (!s.mqttReady) f.wait(80);   // another flow is connecting it right now
    return s;
  }

  /** PUBLISH on s plus its acknowledgement (PUBACK for QoS 1, else often a delayed TCP ACK). */
  function publish(f, s, fromClient, delay, topic, body, { qos = 0, retain = false } = {}) {
    const pid = qos ? s.nextPid() : 0;
    f.seg(delay, s, fromClient, 'PA', () => mqtt(3, { topic, body: body(), qos, retain, pid }));
    if (qos) f.seg(lanDelay(), s, !fromClient, 'PA', mqtt(4, { pid }));
    else if (rng.chance(0.6)) f.seg(rng.int(25, 45), s, !fromClient, 'A');
  }

  const propsJson = id => JSON.stringify(dev(id).props);

  function telemetry(t, id) {
    const s = mqttSession(id);
    if (!s?.mqttReady || !net.budget(3)) return;
    const f = flow(t, id, 'hub');
    const spec = TELEMETRY[dev(id).type];
    publish(f, s, true, 0, `home/${id}/telemetry`, () => JSON.stringify(spec.body(dev(id).props, state)));
  }

  function keepalive(t, id) {
    const s = mqttSession(id);
    if (!s?.mqttReady) return;
    const f = flow(t, id, 'hub');
    f.seg(0, s, true, 'PA', mqtt(12));
    f.seg(lanDelay(), s, false, 'PA', mqtt(13));
  }

  /** Reconnect loop: a powered MQTT client without a session connects. */
  function presence(t, id) {
    if (mqttSession(id) || !up(id) || !net.budget(10)) return;
    connect(flow(t, id, 'hub'), id);
  }

  /** Graceful shutdown of a session (MQTT DISCONNECT first). */
  function goodbye(f, s) {
    if (s.dead) return;
    if (s.kind === 'mqtt') f.seg(2, s, true, 'PA', mqtt(14));
    close(f, s, true);
  }

  /** Retained state publish, e.g. after a physical action (fridge door) or a finished cycle. */
  function statePublish(t, id) {
    const s = mqttSession(id);
    if (!s?.mqttReady) return;
    publish(flow(t, id, 'hub'), s, true, 0, `home/${id}/state`, () => propsJson(id), { retain: true });
  }

  // ---- per-device everyday behaviour ------------------------------------------------

  const tvServer = () => (dev('tv').props.channel === 'SHIELD Dashboard' ? 'shield' : 'cloud');
  const tvHost = () => (tvServer() === 'shield' ? 'shield.home.arpa' : HOSTS.tv[dev('tv').props.channel] || HOSTS.tv['Cricket Live']);

  /** One adaptive-streaming segment (or a dashboard refresh); `burst` after a channel change. */
  function tvFetch(t, burst = false) {
    if (!net.budget(burst ? 45 : 16)) return;
    const f = flow(t, 'tv');
    const s = heldTls(f, 'tls:tv', 'tv', tvHost(), tvServer());
    const dashboard = s.server === 'shield';
    tlsData(f, s, true, rng.int(300, 460), dashboard ? 'GET /api/metrics' : 'GET video segment', 2);
    const n = dashboard ? rng.int(1, 2) : burst ? rng.int(14, 18) : rng.int(5, 8);
    tlsDownload(f, s, n, dashboard ? 'dashboard JSON' : 'video', dashboard ? { min: 400, max: 1300 } : {});
  }

  /** CDNs rotate edges: the TV closes its streaming connection now and then. */
  function tvRotate(t) {
    const s = held.get('tls:tv');
    if (!s || s.dead) return;
    held.delete('tls:tv');
    close(flow(t, 'tv'), s, true);
  }

  function tvChannel(t) {
    if (!up('tv')) return;
    tvRotate(t);
    tvFetch(t + 40, true);
  }

  function speakerBeat(t) {
    if (!net.budget(6)) return;
    const f = flow(t, 'speaker');
    const s = heldTls(f, 'tls:speaker', 'speaker', HOSTS.speaker);
    tlsData(f, s, true, rng.int(110, 190), 'heartbeat', 2);
    tlsData(f, s, false, rng.int(60, 120), 'heartbeat ack', wanDelay());
    f.seg(1, s, true, 'A');
  }

  function speakerMusic(t) {
    if (!net.budget(8)) return;
    const f = flow(t, 'speaker');
    const s = heldTls(f, 'tls:music', 'speaker', HOSTS.music);
    tlsData(f, s, true, rng.int(90, 140), 'GET audio chunk', 2);
    tlsDownload(f, s, rng.int(2, 4), 'audio', { min: 900, max: 1400 });
  }

  const rtp = {
    hub: { seq: rng.int(1, 60000), ts: rng.u32(), ssrc: rng.u32(), port: 40002 },
    cloud: { seq: rng.int(1, 60000), ts: rng.u32(), ssrc: rng.u32(), port: 5004 },
  };

  /** One video frame as an RTP burst: to the hub's recorder, sometimes to the cloud. */
  function cameraBurst(t) {
    const recording = !!dev('camera').props.recording;
    const dst = rng.chance(recording ? 0.3 : 0.08) ? 'cloud' : 'hub';
    const st = rtp[dst];
    const n = rng.int(3, 5);
    if (!net.budget(n)) return;
    st.ts = (st.ts + 3000) >>> 0;   // 90 kHz clock at 30 fps
    const ts = st.ts;
    const f = flow(t, 'camera', dst);
    for (let i = 0; i < n; i++) {
      const bytes = rng.int(900, 1350);
      const mark = i === n - 1 ? ' Mark' : '';
      f.send(i ? rng.int(4, 9) : 0, at => {
        st.seq = (st.seq + 1) & 0xffff;
        return udp(at, 'camera', dst, PORT.RTP, st.port, 'RTP', bytes, {
          payload: `RTP PT=96 SSRC=0x${st.ssrc.toString(16)} Seq=${st.seq} Time=${ts}${mark} (H.264 FU-A)`,
        });
      });
    }
  }

  function clockNtp(t) {
    const f = flow(t, 'clock');
    if (ntpCount++ % 4 === 0) dns(f, 'clock', HOSTS.ntp);
    ntp(f.wait(2), 'clock');
  }

  // Modbus/TCP holding registers of the energy meter.
  const REGISTERS = [
    { name: 'voltage', addr: 0x0000, qty: 2, value: () => `${(229 + rng.next() * 4).toFixed(1)} V` },
    { name: 'current', addr: 0x0006, qty: 2, value: () => `${((dev('meter').props.kw * 1000) / 230).toFixed(2)} A` },
    { name: 'active power', addr: 0x000c, qty: 2, value: () => `${Number(dev('meter').props.kw).toFixed(2)} kW` },
    { name: 'import energy', addr: 0x0048, qty: 2, value: () => `${(4821.6 + state.time.simMs / 3.6e6 * dev('meter').props.kw).toFixed(2)} kWh` },
  ];
  let registerIdx = 0;

  /** Read Holding Registers (fn 3) over the hub's persistent Modbus/TCP connection. */
  function modbusRead(f, reg) {
    let s = held.get('modbus');
    if (!s || s.dead) {
      s = f.own(new Session('hub', 'meter', PORT.MODBUS, { kind: 'modbus' }));
      held.set('modbus', s);
      handshake(f, s);
      f.then(0, () => { s.ready = true; f.release(s); });
    } else if (!s.ready) {
      f.wait(60);
    }
    modbusTxn = (modbusTxn % 65535) + 1;
    const tid = modbusTxn;
    const layer = () => ({ fn: 3, unit: 1, register: reg.addr });
    f.seg(1, s, true, 'PA', { data: 12, proto: 'MODBUS', modbus: layer(), payload: `Read Holding Registers unit=1 addr=0x${hex4(reg.addr)} qty=${reg.qty} (tid ${tid})` });
    f.seg(lanDelay() + 2, s, false, 'PA', () => ({ data: 9 + 2 * reg.qty, proto: 'MODBUS', modbus: layer(), payload: `Read Holding Registers response (tid ${tid}): ${reg.name} ${reg.value()}` }));
    f.seg(rng.int(25, 40), s, true, 'A');
  }

  function modbusPoll(t) {
    if (!net.budget(4)) return;
    const reg = REGISTERS[registerIdx++ % REGISTERS.length];
    modbusRead(flow(t, 'hub', 'meter'), reg);
  }

  /** A short HTTPS exchange on a fresh connection (browsing, vendor cloud calls). */
  function httpsVisit(t, id, host, { req = [380, 900], chunks = [2, 6], what = 'page', linger = [1500, 6000] } = {}) {
    if (!up(id) || !net.budget(26)) return;
    const f = flow(t, id);
    if (rng.chance(0.7)) dns(f, id, host, rng.chance(0.25) ? 28 : 1);
    const s = f.own(new Session(id, 'cloud', PORT.HTTPS));
    f.wait(2);
    handshake(f, s);
    tlsHello(f, s, host);
    tlsData(f, s, true, rng.int(req[0], req[1]), `request ${what}`, 2);
    tlsDownload(f, s, rng.int(chunks[0], chunks[1]), what);
    f.wait(rng.int(linger[0], linger[1]));
    close(f, s, rng.chance(0.6));
  }

  /** Large benign HTTP POST: a sample of the body segments stands in for the whole upload. */
  function upload(t, id) {
    const spec = UPLOADS[id];
    if (!spec || !net.budget(40)) return;
    const f = flow(t, id);
    dns(f, id, spec.host);
    const s = f.own(new Session(id, 'cloud', PORT.HTTP));
    f.wait(2);
    handshake(f, s);
    const bodyLen = Math.round(rng.range(spec.mb[0], spec.mb[1]) * 1048576);
    const uri = spec.uri(++uploadCount);
    const mb = (bodyLen / 1048576).toFixed(1);
    f.seg(2, s, true, 'PA', {
      data: 1448, proto: 'HTTP', http: { method: 'POST', uri, status: null, ua: spec.ua, bodyLen },
      payload: trunc(`POST ${uri} HTTP/1.1 (${mb} MB, multipart/form-data)`),
    });
    const n = rng.int(10, 16);
    for (let i = 0; i < n; i++) {
      f.seg(rng.int(1, 3), s, true, i === n - 1 ? 'PA' : 'A', { data: 1448, payload: `HTTP request body continuation (${mb} MB upload)` });
      if (i % 2 === 1) f.seg(rng.int(2, 5), s, false, 'A');
    }
    f.seg(wanDelay(), s, false, 'PA', {
      data: rng.int(180, 260), proto: 'HTTP', http: { method: null, uri, status: 201, ua: null, bodyLen: 48 },
      payload: 'HTTP/1.1 201 Created (application/json)',
    });
    f.seg(1, s, true, 'A');
    close(f, s, true);
  }

  /** The phone's home app polls Home Assistant's REST API over a keep-alive connection. */
  function haPoll(t) {
    if (!net.budget(6)) return;
    const f = flow(t, 'phone', 'hub');
    let s = held.get('http:phone');
    if (!s || s.dead) {
      s = f.own(new Session('phone', 'hub', PORT.HASS, { kind: 'http' }));
      held.set('http:phone', s);
      handshake(f, s);
      f.then(0, () => { s.ready = true; f.release(s); });
    } else if (!s.ready) {
      f.wait(60);
    }
    const uri = '/api/states';
    const body = rng.int(2600, 4200);
    f.seg(2, s, true, 'PA', { data: 186, proto: 'HTTP', http: { method: 'GET', uri, status: null, ua: 'Home Assistant/2026.9.1 (Android 15)', bodyLen: 0 }, payload: `GET ${uri} HTTP/1.1` });
    f.seg(lanDelay() + 3, s, false, 'A', { data: 1448, proto: 'HTTP', http: { method: null, uri, status: 200, ua: null, bodyLen: body }, payload: `HTTP/1.1 200 OK (application/json, ${body} B)` });
    f.seg(1, s, false, 'PA', { data: Math.max(320, Math.min(1448, body - 1448)), payload: 'HTTP response body continuation' });
    f.seg(1, s, true, 'A');
  }

  /** The hub's outbound remote-access tunnel keeps itself alive. */
  function hubCloud(t) {
    if (!net.budget(6)) return;
    const f = flow(t, 'hub');
    const s = heldTls(f, 'tls:hub', 'hub', HOSTS.hub);
    tlsData(f, s, true, rng.int(90, 160), 'remote UI keepalive', 2);
    tlsData(f, s, false, rng.int(60, 120), 'keepalive ack', wanDelay());
    f.seg(1, s, true, 'A');
  }

  /** The unregistered board walks the /24 with ARP who-has; live hosts answer. */
  function rogueSweep(t) {
    const ip = ipPrefix + sweepNext;
    sweepNext = sweepNext >= 254 ? 1 : sweepNext + 1;
    if (ip === dev('rogue').ip || !net.budget(2)) return;
    const f = flow(t, 'rogue');
    f.send(0, at => arpRequest(at, 'rogue', ip));
    const owner = state.byIp.get(ip);
    if (owner && owner !== 'cloud' && up(owner)) f.send(lanDelay(), at => arpReply(at, owner, 'rogue'));
  }

  function rogueDns(t) {
    dns(flow(t, 'rogue'), 'rogue', rng.pick(HOSTS.rogue), rng.chance(0.2) ? 28 : 1);
  }

  // ---- commands ----------------------------------------------------------------------

  /**
   * A command from the home app (or Alexa, or the watch).
   *   origin → hub   MQTT PUBLISH home/<target>/set (QoS 1) + PUBACK; from the
   *                  vendor cloud through the hub's tunnel when the phone is off Wi-Fi
   *   hub → target   MQTT PUBLISH (QoS 1) + PUBACK, then the target's retained
   *                  state publish; HTTPS for the router; Modbus/TCP for the meter
   * o: { origin, target, body() → object, deliver: 'mqtt'|'https'|'modbus'|'none',
   *      topic?, retire?: Session[] (sessions of a device that was just switched off),
   *      onTarget?(f), what?, register?, after?(f) }
   * Returns { started, flow }: `started` turns true once the first packet is built,
   * which lets network.js fold rapid slider moves into one exchange.
   */
  function control(t, o) {
    const ctl = { started: false, flow: null };
    const f = (ctl.flow = commandFlow(t));
    for (const s of o.retire || []) f.own(s);
    const topic = o.topic || `home/${o.target}/set`;
    const text = () => JSON.stringify(o.body());

    // 1. The request reaches the hub.
    if (o.origin === 'cloud') {
      const s = heldTls(f, 'tls:hub', 'hub', HOSTS.hub);
      f.seg(2, s, false, 'PA', () => {
        ctl.started = true;
        const bytes = 150 + text().length;
        return { data: bytes, proto: 'HTTPS', payload: `TLS 1.3 Application Data (remote command for ${o.target}, ${bytes} B)` };
      });
      f.seg(lanDelay(), s, true, 'A');
    } else {
      const s = ensureMqtt(f, o.origin);
      const pid = s.nextPid();
      f.seg(2, s, true, 'PA', () => { ctl.started = true; return mqtt(3, { topic, body: text(), qos: 1, pid }); });
      f.seg(lanDelay(), s, false, 'PA', mqtt(4, { pid }));
    }

    // 2. The hub relays it to the device, which reports back.
    if (o.deliver === 'mqtt') {
      const s = o.retire ? o.retire.find(x => x.kind === 'mqtt') : ensureMqtt(f, o.target);
      if (s) {
        const pid = s.nextPid();
        f.seg(lanDelay(), s, false, 'PA', () => mqtt(3, { topic, body: text(), qos: 1, pid }));
        f.seg(lanDelay(), s, true, 'PA', mqtt(4, { pid }));
        o.onTarget?.(f);
        publish(f, s, true, rng.int(4, 12), `home/${o.target}/state`, () => propsJson(o.target), { retain: true });
      }
    } else if (o.deliver === 'https') {
      const s = f.own(new Session('hub', o.target, PORT.HTTPS, { kind: 'tls' }));
      f.wait(lanDelay());
      handshake(f, s);
      tlsHello(f, s, `${o.target}.home.arpa`);
      tlsData(f, s, true, rng.int(260, 420), `POST /api/${o.what || 'settings'}`, 1);
      tlsData(f, s, false, rng.int(90, 200), '200 OK', lanDelay());
      close(f, s, true);
    } else if (o.deliver === 'modbus') {
      modbusRead(f, o.register || REGISTERS[2]);
    }

    // 3. A device that was just switched off says goodbye on each of its sessions.
    for (const s of o.retire || []) goodbye(f, s);
    o.after?.(f);
    return ctl;
  }

  /** Alexa: the voice request goes to the cloud, the directive comes back, then MQTT to the hub. */
  function voice(t, actions) {
    const f = commandFlow(t);
    f.onAbort(() => { for (const a of actions) for (const s of a.retire || []) s.kill(); });
    const s = heldTls(f, 'tls:speaker', 'speaker', HOSTS.speaker);
    const n = rng.int(3, 4);
    for (let i = 0; i < n; i++) tlsData(f, s, true, rng.int(900, 1300), 'voice request, Opus audio', i ? rng.int(30, 60) : 2);
    f.seg(wanDelay(), s, false, 'A');
    tlsData(f, s, false, rng.int(380, 720), 'Alexa directive', rng.int(150, 260));   // speech recognition time
    f.seg(1, s, true, 'A');
    f.then(2, at => { for (const a of actions) control(at, { ...a, origin: 'speaker' }); });
    return f;
  }

  const reboot = (t, origin, onRouterDown) => control(t, {
    origin, target: 'router', body: () => ({ reboot: true }), deliver: 'https', what: 'system/reboot',
    after: f => f.then(5, at => onRouterDown(at)),
  });

  const clockSync = (t, origin) => control(t, {
    origin, target: 'clock', body: () => ({ sync: true }), deliver: 'mqtt', onTarget: f => ntp(f.wait(2), 'clock'),
  });

  const meterRead = (t, origin) => control(t, {
    origin, target: 'meter', body: () => ({ read: 'active_power' }), deliver: 'modbus', register: REGISTERS[2],
  });

  const ringPhone = t => control(t, {
    origin: 'watch', target: 'phone', topic: 'home/phone/ring', body: () => ({ ring: true, from: 'watch' }), deliver: 'mqtt',
  });

  // ---- link changes ------------------------------------------------------------------

  /** A device got link (power on, Wi-Fi on, plugged in, router back): announce and reconnect. */
  function deviceUp(t, id) {
    if (!has(id)) return;
    const f = flow(t + rng.int(30, 300), id);
    f.send(0, at => arpRequest(at, id, dev(id).ip, true));
    if (dev(id).role === 'unknown') {
      dns(f.wait(rng.int(60, 160)), id, HOSTS.rogue[0]);
    } else {
      f.send(rng.int(5, 20), at => arpRequest(at, id, GATEWAY_IP));
      f.send(lanDelay(), at => arpReply(at, 'router', id));
    }
    if (mqttClients.has(id)) f.then(rng.int(20, 200), at => presence(at, id));
  }

  /** Removes a device's long-lived sessions from use and returns the live ones (for a graceful close). */
  function detach(id) {
    const out = [];
    for (const [key, s] of held) {
      if (!s.involves(id)) continue;
      held.delete(key);
      if (!s.dead) out.push(s);
    }
    return out;
  }

  /** Link lost without a goodbye (Wi-Fi off, unplugged). */
  function dropDevice(id) {
    for (const s of [...live]) if (s.involves(id)) s.kill();
    detach(id);
  }

  /** Router reboot: Wi-Fi associations and NAT state are gone, every session with them. */
  function dropAll() {
    for (const s of [...live]) s.kill();
    held.clear();
  }

  /** After the router is back, every powered LAN host rejoins over ~2.5 s. */
  function rejoinAll(t) {
    for (const d of state.devices.values()) {
      if (d.role === 'cloud' || d.id === 'router' || !up(d.id)) continue;
      deviceUp(t + rng.int(100, 2500), d.id);
    }
  }

  // ---- schedules -------------------------------------------------------------------

  const jobs = [];
  /** A recurring job: `interval()` gives the gap to the next run (ms of sim time). */
  function every(owner, interval, run, enabled = () => up(owner)) {
    if (has(owner)) jobs.push({ interval, run, enabled, next: null });
  }

  function install() {
    for (const d of state.devices.values()) {
      const id = d.id;
      if (d.role !== 'cloud' && d.role !== 'unknown' && id !== 'router') {
        every(id, () => rng.exp(80000), t => arpGateway(t, id));
      }
      if (mqttClients.has(id)) {
        every(id, () => rng.jitter(3000, 800), t => presence(t, id));
        every(id, () => rng.jitter(30000, 5000), t => keepalive(t, id));
      }
      const tel = TELEMETRY[d.type];
      if (tel) every(id, () => tel.every(dev(id).props) * rng.range(0.8, 1.2), t => telemetry(t, id));
    }
    every('router', () => rng.jitter(5000, 1500), routerProbe);
    every('hub', () => rng.jitter(30000, 5000), hubCloud);
    every('hub', () => rng.jitter(30000, 5000), hubPing);
    every('hub', () => rng.jitter(2000, 150), modbusPoll, () => up('hub') && up('meter'));
    every('tv', () => rng.exp(2400), t => tvFetch(t));
    every('tv', () => rng.exp(20000), t => dns(flow(t, 'tv'), 'tv', rng.pick(HOSTS.tvDns)));
    every('tv', () => rng.jitter(80000, 20000), tvRotate);
    every('speaker', () => rng.jitter(12000, 3000), speakerBeat);
    every('speaker', () => rng.exp(2800), speakerMusic, () => up('speaker') && !!dev('speaker').props.music);
    every('camera', () => rng.exp(1100), cameraBurst);
    every('clock', () => rng.jitter(15000, 3000), clockNtp);
    every('laptop', () => rng.exp(9000), t => httpsVisit(t, 'laptop', rng.pick(HOSTS.laptop)));
    every('laptop', () => rng.exp(55000), t => upload(t, 'laptop'));
    every('phone', () => rng.exp(15000), t => httpsVisit(t, 'phone', rng.pick(HOSTS.phone)));
    every('phone', () => rng.jitter(20000, 4000), haPoll, () => up('phone') && up('hub'));
    every('phone', () => rng.exp(120000), t => upload(t, 'phone'));
    every('vacuum', () => rng.jitter(30000, 6000), t => httpsVisit(t, 'vacuum', HOSTS.vacuum, { req: [1100, 1448], chunks: [1, 1], what: 'cleaning map upload', linger: [200, 800] }),
      () => up('vacuum') && !!dev('vacuum').props.running);
    every('watch', () => rng.exp(45000), t => httpsVisit(t, 'watch', HOSTS.watch, { req: [300, 700], chunks: [1, 2], what: 'health sync', linger: [300, 1200] }));
    every('fridge', () => rng.jitter(90000, 20000), t => httpsVisit(t, 'fridge', HOSTS.fridge, { req: [200, 400], chunks: [1, 1], what: 'vendor telemetry', linger: [200, 900] }));
    every('thermostat', () => rng.jitter(120000, 30000), t => httpsVisit(t, 'thermostat', HOSTS.thermostat, { req: [200, 320], chunks: [1, 1], what: 'weather forecast', linger: [200, 900] }));
    every('rogue', () => rng.jitter(250, 60), rogueSweep);
    every('rogue', () => rng.exp(4000), rogueDns);
  }

  /** The house has been running all evening: long-lived sessions already exist at t = 0. */
  function preEstablish() {
    const keep = (key, s) => held.set(key, s);
    for (const id of mqttClients) if (up(id)) keep(mqttKey(id), new Session(id, 'hub', PORT.MQTT, { kind: 'mqtt', established: true }));
    if (has('tv') && up('tv')) keep('tls:tv', new Session('tv', tvServer(), PORT.HTTPS, { kind: 'tls', established: true }));
    if (has('speaker') && up('speaker')) keep('tls:speaker', new Session('speaker', 'cloud', PORT.HTTPS, { kind: 'tls', established: true }));
    if (has('hub')) keep('tls:hub', new Session('hub', 'cloud', PORT.HTTPS, { kind: 'tls', established: true }));
    if (has('meter')) keep('modbus', new Session('hub', 'meter', PORT.MODBUS, { kind: 'modbus', established: true }));
    if (has('phone') && up('phone')) keep('http:phone', new Session('phone', 'hub', PORT.HASS, { kind: 'http', established: true }));
  }

  /** Runs every job whose time has come (job times are in the past by at most one tick). */
  function tick(now) {
    for (const j of jobs) {
      if (!j.enabled()) { j.next = null; continue; }
      if (j.next === null) j.next = now + rng.range(0.05, 1) * j.interval();
      let runs = 0;
      while (j.next <= now && runs++ < 20) {
        j.run(j.next);
        j.next += Math.max(20, j.interval());
      }
      if (j.next <= now) j.next = now + j.interval();   // fell far behind: skip ahead
    }
  }

  install();
  preEstablish();

  return {
    tick,
    control,
    voice,
    reboot,
    clockSync,
    meterRead,
    ringPhone,
    tvChannel,
    statePublish,
    deviceUp,
    detach,
    dropDevice,
    dropAll,
    rejoinAll,
    /** Debug counters. */
    info: () => ({ packets: packetSeq, liveSessions: live.size, heldSessions: held.size, jobs: jobs.length }),
  };
}
