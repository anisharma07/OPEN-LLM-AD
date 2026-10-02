# SHIELD-IoT Smart Home Lab — module contract

This is the binding interface between the simulator's modules. Each module is
owned by one file and talks to the others only through `state` (shared data)
and `bus` (events). If a module needs something that is not listed here, add
it to this file first.

Scope of this version: a 3D smart home whose devices all share one network,
everyday (benign) traffic, device controls, the SHIELD-IoT dashboard, alerts
and blocking. The inline sensor raises alerts from its evidence layers and
from a device-inventory check (an unregistered device plugged into the LAN).
Blocking is automatic in Prevent mode or manual from the device card.

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
| `js/catalog.js` | `HOUSE, DEVICES, CLASSES, RESEARCH, RESOURCES, SUBNET, GATEWAY_IP, deviceById` | Static data. **Read-only.** |
| `js/bus.js` | `bus.on(evt, fn) → unsubscribe`, `bus.off`, `bus.emit(evt, payload)` | Synchronous. **Read-only.** |
| `js/state.js` | `createState(), resetState(state), deviceByIp(state, ip), simNow(state)` | **Read-only.** |
| `js/main.js` | — | Boot and the animation loop. **Read-only.** |
| `js/ids.js` (+ optional `js/features.js`) | `createIDS({bus, state})` | SHIELD-IoT detection engine. |
| `js/network.js`, `js/traffic.js` | `createNetwork({bus, state, ids})` | Benign traffic, routing, device commands and behaviour. |
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
```

Device runtime fields: `props` (mutable copy of catalog props), `online`,
`status: 'ok'|'alert'|'blocked'|'offline'`, `blocked`, `stats: {tx, rx, dropped}`.

Device roles in the catalog: `infra` (router, shield, hub), `device`, `cloud`
(the internet, outside the house, `wan: true`), `unknown` (the rogue board:
not in the home inventory; it only joins the LAN while `props.plugged` is true).

**State ownership** (who writes):
- `ids.js`: `state.ids.blocked`, `state.ids.alerts`, `state.ids.metrics`, and `device.blocked`.
- `network.js`: `device.props`, `device.stats`, `device.online`, `device.status`. It also handles `sim:reset`.
- `ui.js`: `state.ids.mode`, `state.ids.layers`, `state.ids.threshold`, `state.ids.autoBlockAfter`, `state.selectedId` (always together with emitting the matching event).
- `main.js`: `state.time`.

`device.status` precedence, computed by network.js after any change:
`blocked` > `offline` (power/wifi off, router rebooting, or rogue unplugged) >
`alert` (an alert with this device as source within the last 8 s of sim time) > `ok`.

## Packet

Field names follow Edge-IIoTset where one exists.

```js
{
  id: number, t: simMs,
  src: deviceId, dst: deviceId,          // any DEVICES id, including 'cloud'
  srcIp, dstIp, srcMac, dstMac,
  proto: 'TCP'|'UDP'|'ICMP'|'ARP'|'MQTT'|'HTTP'|'HTTPS'|'DNS'|'NTP'|'MODBUS'|'RTP',   // highest layer
  l4: 'tcp'|'udp'|'icmp'|null,           // null for ARP
  sport, dport, len,                     // frame length in bytes
  tcp:  { flags: {SYN, ACK, FIN, RST, PSH, URG}, len, seq, ack, ackRaw, checksum, window, connInit } | null,
  udp:  { len } | null,
  icmp: { type, code } | null,
  arp:  { opcode, hwSize, protoSize, senderMac, senderIp, targetMac, targetIp } | null,
  mqtt: { msgtype /* 1–14 */, conflags, qos, retain, topic, len } | null,
  http: { method, uri, status, ua, bodyLen } | null,
  dns:  { qname, qtype, qnameLen } | null,
  modbus: { fn, unit, register } | null,
  payload: string,                       // short human-readable excerpt for the packet log
  label: 'Normal' | 'Unregistered',      // GROUND TRUTH. ids.js must never read it except for the false-alarm counters.
}
```

`tcp.connInit` is true only on the first segment of a new connection sent by
the initiator. Not-applicable layers are `null` (never zero-filled), so
invariants can tell "not applicable" from "invalid" (research checklist).
Everyday traffic is protocol-valid, so the invariant engines should report
zero hits; that matches the book (sparse, high-precision evidence).

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
  score,                                // fused suspicion 0..1
  mlClass, mlProb,                      // top ML class and its probability (null when the ML layer is off)
  invariantHits: [{ ruleId, layer: 'tcp'|'arp'|'mqtt', msg }],
  inventory: 'known' | 'unknown',       // is the source MAC/IP in the home inventory?
  drift: boolean,                       // drift state was 'drift' when this packet was scored
  reasons: string[],                    // short human explanations
  latencyUs,                            // measured with performance.now()
}

Alert = { id, t, srcIp, srcId, dstId, cls, severity: 'low'|'medium'|'high'|'critical', ruleId, score, msg, count }
Block = { ip, deviceId, reason, ruleId, at, auto }

Metrics = {
  inspected, allowed, alerted, dropped, firewallDrops,
  pps, ppsAllowed, ppsDropped,               // for the live chart
  falseAlarms, falseAlarmRate,                // alerts or drops on label 'Normal' traffic (known only because this is a simulation)
  avgLatencyUs,
  layerHits: { ml, inventory, INV_TCP_01, INV_TCP_02, INV_TCP_03, INV_ARP_01, INV_MQTT_01, INV_MQTT_02, INV_MQTT_03 },
  drift: { psi, state: 'stable'|'warning'|'drift' },
  piwsBuffer,                                 // invariant-labelled samples queued for drift repair (research hypothesis)
  threat: 'low'|'elevated'|'high'|'critical',
  protoMix: { [proto]: count },               // packets per protocol over the last 10 s
}
```

## Network API (`createNetwork`)

```js
{ tick(simDtMs) }
```

Per packet: build → emit `packet:created` → if `ids.isBlocked(pkt.srcIp)`,
emit `packet:dropped {pkt, reason:'firewall'}` → else `verdict = ids.inspect(pkt)`,
emit `packet:verdict {pkt, verdict}` → `drop` ⇒ `packet:dropped {pkt, reason:'ids', verdict}`,
otherwise `packet:delivered {pkt, verdict}`. While the router reboots, LAN
packets are dropped with `reason:'router-offline'` (no IDS involvement).

## Events

| Event | Payload | Emitted by → consumed by |
|---|---|---|
| `sim:ready` | `{}` | main → any |
| `sim:speed` | `{speed}` | ui → main |
| `sim:pause` | `{paused}` | ui → main |
| `sim:reset` | `{}` | ui → network (`resetState`, `ids.reset()`) |
| `sim:after-reset` | `{}` | network → scene, ui (rebuild from the new `state.devices`) |
| `scene:error` | `{message}` | main → ui (show "3D view unavailable" but keep the panels working) |
| `packet:created` | `pkt` | network → ui |
| `packet:verdict` | `{pkt, verdict}` | network → scene, ui |
| `packet:delivered` | `{pkt, verdict}` | network → scene, ui |
| `packet:dropped` | `{pkt, reason:'firewall'|'ids'|'router-offline', verdict?}` | network → scene, ui |
| `ids:alert` | `Alert` | ids → scene, ui, network (status) |
| `ids:block` / `ids:unblock` | `{ip, deviceId, reason?, auto?}` | ids → scene, ui, network |
| `ids:request-block` / `ids:request-unblock` | `{ip}` | ui → ids |
| `ids:drift` | `{psi, state}` | ids → ui, scene (only on state change) |
| `ids:metrics` | `Metrics` | ids → ui, scene |
| `ids:config` | `{}` | ui → ids, after it changed `state.ids.mode/layers/threshold/autoBlockAfter` |
| `device:select` | `{id}` (id may be null) | ui, scene → ui, scene |
| `device:command` | `{id, key, value}` | ui → network |
| `device:update` | `{id}` | network → ui, scene (re-read `state.devices.get(id)`) |
| `scene:focus` | `{id}` | ui → scene (fly camera to device) |
| `toast` | `{kind:'info'|'warn'|'danger'|'ok', text}` | any → ui |

## Device commands (network.js)

`device:command {id, key, value}` for a control in the device's `controls`:
- toggle/range/select/color set `props[key] = value`. Buttons have no value.
- A control packet travels phone → hub (MQTT PUBLISH `home/<id>/set`), then
  hub → device, then the device publishes its new state back to the hub.
- Special keys: speaker `speak` acts on `props.say` (lights on/off ⇒ every
  light; fan speed 3; play music ⇒ `props.music`; lock the door ⇒ `lock.locked = true`;
  start the washer ⇒ `washer.running = true`) with `props.listening` true for ~2 s.
  Router `reboot` (2 s offline), vacuum `dock`, clock `sync` (NTP exchange),
  meter `read` (Modbus read), watch `findPhone` (a ring on the phone),
  rogue `plugged` (the unknown device joins or leaves the LAN).
- A blocked device ignores commands and the UI says why.

The unknown device, while plugged in and not blocked, sends ARP who-has
requests across the subnet and DNS lookups to the router (label
`Unregistered`). It is valid traffic from a MAC that is not in the inventory.

## Scene (scene.js + models.js)

`createDeviceModel(device)` returns
```js
{ group: THREE.Group,            // local origin = device base; scene.js sets position/rotation from catalog
  port: THREE.Vector3,           // local point where network links attach (may change per frame, e.g. vacuum)
  update(device, dtSec, tSec),   // animate from device.props/status (fan spin, TV screen, clock hands…)
  setStatus(status),             // 'ok'|'alert'|'blocked'|'offline' visual treatment
  dispose() }
```
Packets travel `src → router → shield → dst`. The cloud joins at
`HOUSE.wanEntry` via a fibre line. IDS drops burst at the shield, firewall
drops burst at the router. Animate at most about 250 packets at once and
sample the rest.

## Visual language

- Allowed packet: cyan `#5ad1e6`. Alerted (allowed but flagged): amber `#ffb347`. Dropped: red `#ff4d5e`.
- Shield/IDS accent: saffron `#f2b134`. Blocked: red cage or ring. Offline: dimmed, dashed link.
- The scene is a night-time dollhouse: cut-away walls, warm interior light, cool outdoor blue.
