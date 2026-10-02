# SHIELD-IoT Home Simulator — module contract

This is the binding interface between the simulator's modules. Each module is
owned by one file and talks to the others only through `state` (shared data)
and `bus` (events). If a module needs something that is not listed here, add
it to this file first.

Plain ES modules, no build step. three.js 0.169.0 comes from an import map in
`index.html`:

```html
<script type="importmap">
{ "imports": {
  "three": "https://cdn.jsdelivr.net/npm/three@0.169.0/build/three.module.js",
  "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.169.0/examples/jsm/"
} }
</script>
```

Only `three` and `three/addons/...` may be imported from outside the project.
No other libraries, no fetches to other hosts, no `alert/confirm/prompt`, no
`localStorage` without try/catch. The page must run when served as static files.

## Files and owners

| File | Exports | Notes |
|---|---|---|
| `js/catalog.js` | `HOUSE, DEVICES, CLASSES, ATTACKS, RESEARCH, RESOURCES, SUBNET, GATEWAY_IP, deviceById, attackById` | Static data. **Read-only, do not edit.** |
| `js/bus.js` | `bus.on(evt, fn) → unsubscribe`, `bus.off`, `bus.emit(evt, payload)` | Synchronous. **Do not edit.** |
| `js/state.js` | `createState(), resetState(state), deviceByIp(state, ip), simNow(state)` | **Do not edit.** |
| `js/main.js` | — | Boot and the animation loop. **Do not edit.** |
| `js/ids.js` (+ optional `js/features.js`) | `createIDS({bus, state})` | SHIELD-IoT detection engine. |
| `js/network.js`, `js/traffic.js`, `js/attacks.js` | `createNetwork({bus, state, ids})` | Packet creation, routing, attacks, device commands. |
| `js/scene.js` | `createScene({bus, state, container}) → Promise<{tick(realDtMs, simDtMs), focus(id)}>` | three.js world, links, packet animation, picking. |
| `js/models.js` | `createDeviceModel(device) → DeviceModel` | Procedural low-poly model per device type. |
| `index.html`, `css/style.css`, `js/ui.js`, `js/charts.js` | `createUI({bus, state}) → {tick(realDtMs)}` | All DOM/HUD. |

Boot order in `main.js`: `createState` → `createIDS` → `createNetwork` →
`createUI` → `await createScene`. Each frame: `network.tick(simDt)`,
`ids.tick(simDt)`, `scene.tick(realDt, simDt)`, `ui.tick(realDt)`.
`simDt` is 0 while paused and is scaled by `state.time.speed`. `window.__shield`
exposes `{bus, state, ids, network, scene, ui}` for debugging and tests.

## State (see `js/state.js`)

```js
state.time      = { simMs, epochMs, speed /* 0.5|1|2|4 */, paused }
state.devices   = Map<id, Device>      // catalog entry + runtime fields below
state.byIp      = Map<ip, id>
state.selectedId = id | null
state.ids       = { mode: 'prevent'|'detect', layers: {ml, tcp, arp, mqtt, drift}, threshold, autoBlockAfter,
                    blocked: Map<ip, Block>, alerts: Alert[] /* newest first, ≤300 */, metrics: Metrics|null }
state.campaigns = Map<campaignId, Campaign>
```

Device runtime fields: `props` (mutable copy of catalog props), `online`,
`status: 'ok'|'alert'|'blocked'|'compromised'`, `blocked`, `compromised`,
`maliciousRx`, `stats: {tx, rx, dropped}`.

**State ownership** (who writes):
- `ids.js`: `state.ids.blocked`, `state.ids.alerts`, `state.ids.metrics`, and `device.blocked`.
- `network.js`: `device.props`, `device.compromised`, `device.maliciousRx`, `device.stats`, `device.online`, `device.status`, `state.campaigns`. It also handles `sim:reset`.
- `ui.js`: `state.ids.mode`, `state.ids.layers`, `state.ids.threshold`, `state.selectedId` (always together with emitting the matching event).
- `main.js`: `state.time`.

`device.status` precedence, computed by network.js after any change:
`blocked` > `compromised` > `alert` (an alert on this device as source within the last 8 s of sim time) > `ok`.

## Packet

Field names follow Edge-IIoTset where one exists.

```js
{
  id: number, t: simMs,
  src: deviceId, dst: deviceId,          // any DEVICES id, including 'cloud' and 'attacker'
  srcIp, dstIp, srcMac, dstMac,          // srcIp may be spoofed (198.51.100.x) for DDoS with spoof on
  proto: 'TCP'|'UDP'|'ICMP'|'ARP'|'MQTT'|'HTTP'|'HTTPS'|'DNS'|'NTP'|'MODBUS'|'RTP',   // highest layer
  l4: 'tcp'|'udp'|'icmp'|null,           // null for ARP
  sport, dport, len,                     // frame length in bytes
  tcp:  { flags: {SYN, ACK, FIN, RST, PSH, URG}, len, seq, ack, ackRaw, checksum, window, connInit } | null,
  udp:  { len } | null,
  icmp: { type, code } | null,
  arp:  { opcode, hwSize, protoSize, senderMac, senderIp, targetMac, targetIp } | null,
  mqtt: { msgtype /* 0–15 */, conflags, qos, retain, topic, len } | null,
  http: { method, uri, status, ua, bodyLen } | null,
  dns:  { qname, qtype, qnameLen } | null,
  modbus: { fn, unit, register } | null,
  payload: string,                       // short human-readable excerpt for the packet log
  label: 'Normal' | ClassName,           // GROUND TRUTH. ids.js must never read it except for the confusion matrix.
  campaignId: string | null, manual: boolean,
}
```

`tcp.connInit` is true only on the first segment of a new connection sent by
the initiator. Not-applicable layers are `null` (never zero-filled), so
invariants can tell "not applicable" from "invalid" (research checklist).

## IDS API (`createIDS`)

```js
{
  inspect(pkt) → Verdict,      // synchronous, called by network.js once per packet that passed the firewall
  isBlocked(ip) → boolean,     // router firewall lookup
  block(ip, { reason, ruleId = null, auto = false }),
  unblock(ip),
  reset(),
  tick(simDtMs),               // windows, drift monitor, emits 'ids:metrics' about every 500 ms of real time
}

Verdict = {
  action: 'allow' | 'alert' | 'drop',   // 'drop' only in prevent mode
  score,                                // fused attack probability 0..1
  mlClass, mlProb,                      // top ML class and its probability (null when the ML layer is off)
  invariantHits: [{ ruleId, layer: 'tcp'|'arp'|'mqtt', msg }],
  drift: boolean,                       // drift state was 'drift' when this packet was scored
  reasons: string[],                    // short human explanations
  latencyUs,                            // measured with performance.now()
}

Alert = { id, t, srcIp, srcId, dstId, cls, severity: 'low'|'medium'|'high'|'critical', ruleId, score, msg, count }
Block = { ip, deviceId, reason, ruleId, at, auto }

Metrics = {
  inspected, allowed, alerted, dropped, firewallDrops, firewallBenignDrops,
  pps, ppsBenign, ppsMalicious,              // ground-truth split, for the chart only
  tp, fp, tn, fn, precision, recall, f1, fpr, // positive = attack; detected = action !== 'allow'
  avgLatencyUs,
  layerHits: { ml, INV_TCP_01, INV_TCP_02, INV_TCP_03, INV_ARP_01, INV_MQTT_01, INV_MQTT_02, INV_MQTT_03 },
  drift: { psi, state: 'stable'|'warning'|'drift' },
  piwsBuffer,                                 // invariant-labelled samples queued for drift repair (hypothesis)
  threat: 'low'|'elevated'|'high'|'critical',
  byClass: { [cls]: count },                  // detections by ML class
}
```

## Network API (`createNetwork`)

```js
{ tick(simDtMs), injectPacket(spec), startAttack(cfg) → Campaign, stopAttack(id), stopAll() }
Campaign = { id, type /* ATTACKS id */, attackerId, targetId, rate, durationSec, spoof, startedAt, sent, dropped, delivered, active }
```

Per packet: build → emit `packet:created` → if `ids.isBlocked(pkt.srcIp)`,
emit `packet:dropped {pkt, reason:'firewall'}` → else `verdict = ids.inspect(pkt)`,
emit `packet:verdict {pkt, verdict}` → `drop` ⇒ `packet:dropped {pkt, reason:'ids', verdict}`,
otherwise `packet:delivered {pkt, verdict}`.

## Events

| Event | Payload | Emitted by → consumed by |
|---|---|---|
| `sim:ready` | `{}` | main → any |
| `sim:speed` | `{speed}` | ui → main |
| `sim:pause` | `{paused}` | ui → main |
| `sim:reset` | `{}` | ui → network (stops campaigns, `resetState`, `ids.reset()`) |
| `sim:after-reset` | `{}` | network → scene, ui (rebuild from the new `state.devices`) |
| `scene:error` | `{message}` | main → ui (show "3D view unavailable" but keep the panels working) |
| `packet:created` | `pkt` | network → ui |
| `packet:verdict` | `{pkt, verdict}` | network → scene, ui |
| `packet:delivered` | `{pkt, verdict}` | network → scene, ui |
| `packet:dropped` | `{pkt, reason:'firewall'|'ids', verdict?}` | network → scene, ui |
| `packet:send` | `PacketSpec` | ui → network (manual composer) |
| `ids:alert` | `Alert` | ids → scene, ui, network (status) |
| `ids:block` / `ids:unblock` | `{ip, deviceId, reason?, auto?}` | ids → scene, ui, network |
| `ids:request-block` / `ids:request-unblock` | `{ip}` | ui → ids |
| `ids:drift` | `{psi, state}` | ids → ui, scene (only on state change) |
| `ids:metrics` | `Metrics` | ids → ui |
| `ids:config` | `{}` | ui → ids, after it changed `state.ids.mode/layers/threshold` |
| `attack:request` | `{type, attackerId, targetId, rate, durationSec, spoof}` | ui → network |
| `attack:stop` | `{id}` or `{all:true}` | ui → network |
| `attack:started` / `attack:ended` | `Campaign` | network → ui, scene |
| `device:select` | `{id}` (id may be null) | ui, scene → ui, scene |
| `device:command` | `{id, key, value}` | ui → network |
| `device:update` | `{id}` | network → ui, scene (re-read `state.devices.get(id)`) |
| `device:compromised` | `{id, by}` | network → ui, scene |
| `scene:focus` | `{id}` | ui → scene (fly camera to device) |
| `toast` | `{kind:'info'|'warn'|'danger'|'ok', text}` | any → ui |

`PacketSpec` (manual composer): `{ srcId, dstId, proto, sport, dport, count,
tcpFlags: {SYN,ACK,FIN,RST,PSH,URG}, tcpLen, connInit, arpOpcode, arpHwSize,
mqttType, mqttConflags, mqttQos, payload, label }`. `label` is the
ground truth the user claims: `'Normal'` or one of `CLASSES`.

## Device commands (network.js)

`device:command {id, key, value}` for a control in the device's `controls`:
- toggle/range/select/color set `props[key] = value`. Buttons have no value.
- A control packet travels phone → hub (MQTT PUBLISH `home/<id>/set`), then
  hub → device, then the device publishes its new state back to the hub.
- Special keys: speaker `speak` acts on `props.say` (lights on/off ⇒ every
  light; fan speed 3; play music ⇒ `props.music`; lock the door ⇒ `lock.locked = true`;
  start the washer ⇒ `washer.running = true`). Also router `reboot` (2 s offline),
  vacuum `dock`, clock `sync` (NTP exchange), meter `read` (Modbus read),
  watch `findPhone`.
- `{id, key:'restore'}` clears `compromised`/`maliciousRx` (UI "Restore device" button).
- A blocked device ignores commands and the UI says why.

Compromise rule: when `maliciousRx` from Backdoor or Ransomware campaigns
reaches 25 on a device that is not `attacker`/`rogue`/`cloud`, it becomes
`compromised`, emits `device:compromised`, and starts beaconing (label
`Backdoor`, TCP to the attacker on port 4444, about 1 pkt/s) until restored or
blocked. A compromised TV shows a ransom screen, a compromised light flickers red.

## Scene (scene.js + models.js)

`createDeviceModel(device)` returns
```js
{ group: THREE.Group,            // local origin = device base; scene.js sets position/rotation from catalog
  port: THREE.Vector3,           // local point where network links attach
  update(device, dtSec, tSec),   // animate from device.props/status (fan spin, TV screen, clock hands…)
  setStatus(status),             // 'ok'|'alert'|'blocked'|'compromised' visual treatment
  dispose() }
```
Packets travel `src → router → shield → dst`. WAN devices (`cloud`,
`attacker`) join at `HOUSE.wanEntry` via a fibre line. IDS drops burst at the
shield, firewall drops burst at the router. Animate at most about 250 packets
at once and sample the rest, so floods stay smooth.

## Visual language

- Benign packet: cyan `#5ad1e6`. Malicious and allowed (missed): amber `#ffb347`. Dropped: red `#ff4d5e`.
- Shield/IDS accent: saffron `#f2b134`. Blocked: red cage or ring. Compromised: magenta `#e04fd8`.
- The scene is a night-time dollhouse: cut-away walls, warm interior light, cool outdoor blue.
