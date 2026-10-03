# SHIELD-IoT

SHIELD-IoT is a research project by Anirudh Sharma (NIT Hamirpur) on IoT/IIoT intrusion detection that stays reliable under distribution shift. It combines probabilistic ML evidence (a frozen LightGBM baseline) with deterministic protocol invariants (TCP, ARP and, planned, MQTT), drift detection and, as a research hypothesis, Protocol-Invariant Weak Supervision (PIWS) for drift repair.

**Architecture, research summary, results and status: [archi.md](archi.md).**

This folder also contains the **SHIELD-IoT Smart Home Lab**, a 3D browser simulator of a smart home with the SHIELD-IoT sensor inline at the router.

## Run the simulator

The simulator is plain static files (ES modules, no build step), but browsers do not load modules from `file://`, so serve the folder over HTTP:

```bash
cd SHIELD-IoT/simulator
python3 -m http.server 8000
# or: npx serve -l 8000
```

Then open <http://localhost:8000>.

It needs an internet connection: three.js 0.169.0 is loaded from the jsDelivr CDN and the fonts from Google Fonts. It also needs a browser with WebGL. If the 3D view cannot start, the panels keep working.

## Features

- A night-time cut-away 3D house with 19 smart devices, the router, the SHIELD-IoT sensor, a home hub (MQTT broker), the internet, and one unregistered board, all on `192.168.0.0/24`.
- Everyday traffic in many protocols (MQTT, HTTPS, HTTP, DNS, NTP, RTP, Modbus/TCP, ARP, ICMP), animated device → router → SHIELD-IoT → destination: cyan allowed, amber flagged, red dropped.
- Device cards with working controls (lights, fan, AC, TV, Alexa voice commands, door lock, washer, thermostat, camera, vacuum, clock, meter, router reboot). Each command is real simulated network traffic.
- A switch to plug in an unknown device whose MAC address is not in the home inventory.
- The SHIELD-IoT dashboard: counters, a live traffic chart, evidence layers with hit counts, response tuning, alerts, blocked devices, a drift gauge (PSI) with the PIWS buffer, the protocol mix and the false-alarm rate.
- Evidence layers you can switch on and off: ML (a hand-weighted LightGBM *surrogate*, not the trained model), TCP invariants (Step 5A), ARP invariant (Step 5B), MQTT invariants (experimental, simulator-only, off by default) and the drift monitor. The inventory check is always on.
- Prevent mode (drop and auto-block at the router) or Detect mode (alerts only); manual block and unblock.
- A packet log of the last 200 packets with a protocol filter and a per-packet explanation of the verdict.
- Research and Resources tabs with the book's numbers and links to the datasets and standards.

There is no attack launcher in this version: everyday traffic is protocol-valid, so the invariant engines stay quiet. The ground-truth label of each packet is used only to count false alarms.

## Controls

| Input | Action |
|---|---|
| Left-drag | Orbit the camera |
| Scroll wheel or pinch | Zoom |
| Right-drag | Pan across the floor |
| Click a device | Select it and open its card |
| Double-click a device | Fly the camera to it |
| <kbd>Space</kbd> | Pause or resume the simulation |
| <kbd>Esc</kbd> | Clear the selection |
| Top bar | Pause, speed (0.5×, 1×, 2×, 4×), Prevent or Detect, reset, help (?) |

## Simulator files

All paths are under `simulator/`. The module interface is documented in [`simulator/CONTRACT.md`](simulator/CONTRACT.md).

| File | Purpose |
|---|---|
| `index.html` | Page markup, HUD panels, help dialog and the import map for three.js |
| `css/style.css` | Styles for the HUD and panels |
| `js/main.js` | Boot order and the animation loop |
| `js/catalog.js` | Static data: floor plan, devices, classes, the book's numbers, outside links |
| `js/state.js` | The single shared state object |
| `js/bus.js` | Synchronous event bus between modules |
| `js/network.js` | Routing through the router firewall and the IDS, device commands and behaviour |
| `js/traffic.js` | Everyday traffic: packet builders, TCP and MQTT sessions, device schedules |
| `js/ids.js` | The SHIELD-IoT engine: invariants, inventory, fusion, response, drift monitor, metrics |
| `js/features.js` | The 37-feature projection and the ML surrogate |
| `js/scene.js` | The three.js world: house, links, packet animation, picking and camera |
| `js/models.js` | Procedural low-poly 3D models of every device |
| `js/ui.js` | The dashboard, device cards, alerts, packet log, research and resources tabs |
| `js/charts.js` | The live traffic chart and the bar charts |
| `CONTRACT.md` | The binding interface between the modules |
