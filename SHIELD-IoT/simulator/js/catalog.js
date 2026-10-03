// Static data shared by every module: the home's floor plan, its devices,
// the research numbers quoted in the UI and outside resources.
// Units: metres. Floor at y = 0, ceiling at y = 2.8. +z points to the street.

export const HOUSE = {
  bounds: { minX: -9, maxX: 9, minZ: -6, maxZ: 6 },
  wallHeight: 2.8,
  frontDoor: { x: 0, z: 6, width: 1.1 },
  rooms: [
    { id: 'living',   name: 'Living room', x: [-9, 1], z: [-6, 1], floor: 'wood'  },
    { id: 'kitchen',  name: 'Kitchen',     x: [1, 9],  z: [-6, 0], floor: 'tile'  },
    { id: 'bedroom',  name: 'Bedroom',     x: [-9, -1], z: [1, 6], floor: 'carpet' },
    { id: 'hall',     name: 'Hallway',     x: [-1, 1], z: [1, 6], floor: 'wood'  },
    { id: 'utility',  name: 'Utility',     x: [1, 5],  z: [0, 6], floor: 'tile'  },
    { id: 'study',    name: 'Study',       x: [5, 9],  z: [0, 6], floor: 'wood'  },
  ],
  // Outdoor anchors (not rooms).
  street: { z: 9.5 },
  wanEntry: [0.2, 0.95, 0.35],   // where the ISP fibre meets the router
};

export const SUBNET = '192.168.0.0/24';
export const GATEWAY_IP = '192.168.0.1';

// Control descriptors drive the generic device panel in ui.js.
// type: 'toggle' | 'range' | 'select' | 'color' | 'button'
const power = { key: 'power', label: 'Power', type: 'toggle' };

export const DEVICES = [
  // ---- infrastructure -------------------------------------------------
  {
    id: 'router', name: 'Wi-Fi Router', type: 'router', role: 'infra', room: 'living',
    ip: '192.168.0.1', mac: 'a4:2b:b0:10:00:01', vendor: 'Gateway (Wi-Fi 6)',
    protocols: ['ARP', 'DHCP', 'DNS', 'NAT'], pos: [0.2, 0.95, 0.35], rotY: Math.PI,
    controls: [{ key: 'guestWifi', label: 'Guest Wi-Fi', type: 'toggle' }, { key: 'reboot', label: 'Reboot', type: 'button' }],
    props: { guestWifi: false, clients: 0 },
    desc: 'Default gateway for the 192.168.0.0/24 home network. Every packet crosses it, so SHIELD-IoT sits inline right here.',
    links: [{ label: 'RFC 826: ARP', url: 'https://www.rfc-editor.org/rfc/rfc826' }],
  },
  {
    id: 'shield', name: 'SHIELD-IoT Sensor', type: 'shield', role: 'infra', room: 'living',
    ip: '192.168.0.2', mac: 'a4:2b:b0:5e:1d:02', vendor: 'SHIELD-IoT edge appliance',
    protocols: ['inline tap'], pos: [0.75, 0.95, 0.35], rotY: Math.PI,
    controls: [], props: {},
    desc: 'Inline IDS/IPS. Fuses ML evidence, TCP/ARP/MQTT protocol invariants, a device inventory check and a drift monitor, then alerts or blocks at the router firewall.',
    links: [
      { label: 'Edge-IIoTset dataset', url: 'https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot-applications' },
      { label: 'ToN-IoT datasets (UNSW)', url: 'https://research.unsw.edu.au/projects/toniot-datasets' },
    ],
  },
  {
    id: 'hub', name: 'Home Hub (MQTT broker)', type: 'hub', role: 'infra', room: 'living',
    ip: '192.168.0.10', mac: 'dc:a6:32:00:00:10', vendor: 'Raspberry Pi running Home Assistant + Mosquitto',
    protocols: ['MQTT', 'Modbus/TCP', 'HTTP'], pos: [-0.35, 0.95, 0.35], rotY: Math.PI,
    controls: [], props: { automations: 6 },
    desc: 'Automation hub and MQTT broker on port 1883. Lights, fan, AC and appliances publish state here.',
    links: [{ label: 'MQTT', url: 'https://mqtt.org/' }, { label: 'Home Assistant', url: 'https://www.home-assistant.io/' }],
  },

  // ---- living room ----------------------------------------------------
  {
    id: 'tv', name: 'Smart TV', type: 'tv', role: 'device', room: 'living',
    ip: '192.168.0.20', mac: '70:2a:d5:00:00:20', vendor: '55" Android TV',
    protocols: ['HTTPS', 'DNS', 'HTTP'], pos: [-4, 1.35, -5.88], rotY: 0,
    controls: [power,
      { key: 'channel', label: 'Channel', type: 'select', options: ['Cricket Live', 'News', 'Movies', 'Cartoons', 'SHIELD Dashboard'] },
      { key: 'volume', label: 'Volume', type: 'range', min: 0, max: 100, step: 1 }],
    props: { power: true, channel: 'Cricket Live', volume: 32 },
    desc: 'Streams video from the internet. High-volume HTTPS traffic that the ML layer must not confuse with DDoS_HTTP.',
    links: [{ label: 'OWASP IoT Top 10', url: 'https://owasp.org/www-project-internet-of-things/' }],
  },
  {
    id: 'speaker', name: 'Alexa Echo', type: 'speaker', role: 'device', room: 'living',
    ip: '192.168.0.21', mac: '0c:47:c9:00:00:21', vendor: 'Voice assistant',
    protocols: ['HTTPS', 'MQTT'], pos: [-6.6, 0.62, -2.2], rotY: 0.6,
    controls: [power,
      { key: 'say', label: 'Voice command', type: 'select', options: ['Alexa, lights on', 'Alexa, lights off', 'Alexa, fan speed 3', 'Alexa, play music', 'Alexa, lock the door', 'Alexa, start the washer'] },
      { key: 'speak', label: 'Speak', type: 'button' }],
    props: { power: true, say: 'Alexa, lights on', listening: false, music: false },
    desc: 'Voice assistant. A spoken command becomes HTTPS to the cloud plus MQTT publishes to the hub.',
    links: [{ label: 'Matter standard (CSA)', url: 'https://csa-iot.org/all-solutions/matter/' }],
  },
  {
    id: 'light-living', name: 'Living Room Light', type: 'light', role: 'device', room: 'living',
    ip: '192.168.0.30', mac: 'b0:ce:18:00:00:30', vendor: 'RGB smart bulb',
    protocols: ['MQTT'], pos: [-4, 2.72, -2.4], rotY: 0,
    controls: [power, { key: 'brightness', label: 'Brightness', type: 'range', min: 0, max: 100, step: 1, unit: '%' }, { key: 'color', label: 'Colour', type: 'color' }],
    props: { power: true, brightness: 80, color: '#ffd8a8' },
    desc: 'MQTT bulb subscribed to home/light-living/set.',
    links: [{ label: 'MQTT v5.0 spec (OASIS)', url: 'https://docs.oasis-open.org/mqtt/mqtt/v5.0/mqtt-v5.0.html' }],
  },
  {
    id: 'ac', name: 'Air Conditioner', type: 'ac', role: 'device', room: 'living',
    ip: '192.168.0.34', mac: '44:23:7c:00:00:34', vendor: 'Inverter split AC',
    protocols: ['MQTT'], pos: [-8.82, 2.2, -3.2], rotY: Math.PI / 2,
    controls: [power, { key: 'temp', label: 'Set temperature', type: 'range', min: 16, max: 30, step: 1, unit: '°C' },
      { key: 'mode', label: 'Mode', type: 'select', options: ['Cool', 'Dry', 'Fan'] }],
    props: { power: false, temp: 24, mode: 'Cool' },
    desc: 'Split AC with a Wi-Fi module. Publishes room temperature telemetry every few seconds.',
    links: [],
  },
  {
    id: 'clock', name: 'Smart Wall Clock', type: 'clock', role: 'device', room: 'living',
    ip: '192.168.0.52', mac: '24:0a:c4:00:00:52', vendor: 'ESP32 NTP clock',
    protocols: ['NTP/UDP'], pos: [-7.3, 2.05, -5.9], rotY: 0,
    controls: [{ key: 'format', label: 'Format', type: 'select', options: ['24h', '12h'] }, { key: 'sync', label: 'Sync NTP now', type: 'button' }],
    props: { format: '24h' },
    desc: 'Keeps time over NTP (UDP 123). Shows the simulation clock.',
    links: [{ label: 'RFC 5905: NTPv4', url: 'https://www.rfc-editor.org/rfc/rfc5905' }],
  },
  {
    id: 'phone', name: "Anirudh's Phone", type: 'phone', role: 'device', room: 'living',
    ip: '192.168.0.61', mac: 'f0:18:98:00:00:61', vendor: 'Android phone',
    protocols: ['HTTPS', 'MQTT', 'DNS'], pos: [-3.4, 0.5, -0.9], rotY: 0.4,
    controls: [{ key: 'wifi', label: 'Wi-Fi', type: 'toggle' }],
    props: { wifi: true },
    desc: 'Runs the home app. Control commands from the UI are sent as if tapped on this phone.',
    links: [],
  },
  {
    id: 'vacuum', name: 'Robot Vacuum', type: 'vacuum', role: 'device', room: 'living',
    ip: '192.168.0.63', mac: '50:ec:50:00:00:63', vendor: 'LiDAR robot vacuum',
    protocols: ['MQTT', 'HTTPS'], pos: [-2, 0.05, -3.5], rotY: 0,
    controls: [{ key: 'running', label: 'Cleaning', type: 'toggle' }, { key: 'dock', label: 'Return to dock', type: 'button' }],
    props: { running: true, battery: 76 },
    desc: 'Roams the living room floor and uploads cleaning maps.',
    links: [],
  },

  // ---- kitchen ----------------------------------------------------------
  {
    id: 'fridge', name: 'Smart Fridge', type: 'fridge', role: 'device', room: 'kitchen',
    ip: '192.168.0.40', mac: '8c:79:f5:00:00:40', vendor: 'Wi-Fi refrigerator',
    protocols: ['MQTT', 'HTTPS'], pos: [8.15, 0, -5.4], rotY: 0,
    controls: [{ key: 'temp', label: 'Fridge temperature', type: 'range', min: 1, max: 8, step: 1, unit: '°C' },
      { key: 'doorOpen', label: 'Door open', type: 'toggle' }],
    props: { temp: 4, doorOpen: false },
    desc: 'Reports compartment temperature and door state over MQTT.',
    links: [],
  },
  {
    id: 'light-kitchen', name: 'Kitchen Light', type: 'light', role: 'device', room: 'kitchen',
    ip: '192.168.0.32', mac: 'b0:ce:18:00:00:32', vendor: 'Smart bulb',
    protocols: ['MQTT'], pos: [5, 2.72, -3], rotY: 0,
    controls: [power, { key: 'brightness', label: 'Brightness', type: 'range', min: 0, max: 100, step: 1, unit: '%' }, { key: 'color', label: 'Colour', type: 'color' }],
    props: { power: true, brightness: 90, color: '#fff4e0' },
    desc: 'MQTT bulb over the kitchen island.',
    links: [],
  },
  {
    id: 'plug', name: 'Coffee Maker Plug', type: 'plug', role: 'device', room: 'kitchen',
    ip: '192.168.0.64', mac: 'c4:4f:33:00:00:64', vendor: 'Smart plug with energy metering',
    protocols: ['MQTT'], pos: [3.4, 0.95, -5.55], rotY: 0,
    controls: [power],
    props: { power: false, watts: 0 },
    desc: 'Switches the coffee maker and reports power draw.',
    links: [],
  },

  // ---- bedroom ----------------------------------------------------------
  {
    id: 'light-bed', name: 'Bedroom Light', type: 'light', role: 'device', room: 'bedroom',
    ip: '192.168.0.31', mac: 'b0:ce:18:00:00:31', vendor: 'RGB smart bulb',
    protocols: ['MQTT'], pos: [-3, 2.72, 3.5], rotY: 0,
    controls: [power, { key: 'brightness', label: 'Brightness', type: 'range', min: 0, max: 100, step: 1, unit: '%' }, { key: 'color', label: 'Colour', type: 'color' }],
    props: { power: true, brightness: 45, color: '#c9a7ff' },
    desc: 'MQTT bulb, dimmed for the evening.',
    links: [],
  },
  {
    id: 'fan', name: 'Ceiling Fan', type: 'fan', role: 'device', room: 'bedroom',
    ip: '192.168.0.33', mac: '84:f3:eb:00:00:33', vendor: 'BLDC smart fan',
    protocols: ['MQTT'], pos: [-6, 2.78, 3.6], rotY: 0,
    controls: [power, { key: 'speed', label: 'Speed', type: 'select', options: [1, 2, 3, 4, 5] }],
    props: { power: true, speed: 3 },
    desc: 'Five-speed ceiling fan controlled over MQTT.',
    links: [],
  },
  {
    id: 'watch', name: 'Smartwatch', type: 'watch', role: 'device', room: 'bedroom',
    ip: '192.168.0.60', mac: 'a8:5b:78:00:00:60', vendor: 'Wear OS watch',
    protocols: ['HTTPS', 'MQTT'], pos: [-8.1, 0.62, 5.2], rotY: 0.3,
    controls: [{ key: 'findPhone', label: 'Find my phone', type: 'button' }],
    props: { heartRate: 72, steps: 6420 },
    desc: 'Syncs heart rate and steps on the bedside table.',
    links: [],
  },

  // ---- hallway ----------------------------------------------------------
  {
    id: 'lock', name: 'Front Door Lock', type: 'lock', role: 'device', room: 'hall',
    ip: '192.168.0.51', mac: '00:17:88:00:00:51', vendor: 'Smart deadbolt',
    protocols: ['MQTT'], pos: [0.45, 1.05, 5.92], rotY: Math.PI,
    controls: [{ key: 'locked', label: 'Locked', type: 'toggle' }],
    props: { locked: true },
    desc: 'Deadbolt on the front door. A prime target: an unlock command from the wrong host is a physical breach.',
    links: [],
  },
  {
    id: 'thermostat', name: 'Thermostat', type: 'thermostat', role: 'device', room: 'hall',
    ip: '192.168.0.35', mac: '18:b4:30:00:00:35', vendor: 'Learning thermostat',
    protocols: ['MQTT', 'HTTPS'], pos: [-0.95, 1.5, 3.2], rotY: Math.PI / 2,
    controls: [{ key: 'target', label: 'Target', type: 'range', min: 16, max: 30, step: 0.5, unit: '°C' }],
    props: { target: 23.5, current: 25.1 },
    desc: 'Reads room temperature and drives the AC schedule.',
    links: [],
  },
  {
    id: 'camera', name: 'Doorbell Camera', type: 'camera', role: 'device', room: 'hall',
    ip: '192.168.0.50', mac: '9c:8e:cd:00:00:50', vendor: '2K security camera',
    protocols: ['RTP/UDP', 'HTTPS'], pos: [1.2, 2.3, 6.08], rotY: 0,
    controls: [power, { key: 'recording', label: 'Recording', type: 'toggle' }],
    props: { power: true, recording: true },
    desc: 'Streams video over UDP to the hub and the cloud. Busy and bursty by design.',
    links: [],
  },

  // ---- utility + study --------------------------------------------------
  {
    id: 'washer', name: 'Washing Machine', type: 'washer', role: 'device', room: 'utility',
    ip: '192.168.0.41', mac: '8c:79:f5:00:00:41', vendor: 'Front-load washer',
    protocols: ['MQTT'], pos: [2.1, 0, 5.4], rotY: Math.PI,
    controls: [{ key: 'program', label: 'Programme', type: 'select', options: ['Cotton 40°', 'Quick 15', 'Wool', 'Eco 20°'] },
      { key: 'running', label: 'Running', type: 'toggle' }],
    props: { program: 'Quick 15', running: false, remainingMin: 15 },
    desc: 'Reports cycle progress over MQTT.',
    links: [],
  },
  {
    id: 'meter', name: 'Energy Meter', type: 'meter', role: 'device', room: 'utility',
    ip: '192.168.0.65', mac: '00:1d:9c:00:00:65', vendor: 'Modbus/TCP smart meter',
    protocols: ['Modbus/TCP'], pos: [4.9, 1.6, 2], rotY: -Math.PI / 2,
    controls: [{ key: 'read', label: 'Read registers', type: 'button' }],
    props: { kw: 1.42 },
    desc: 'An industrial-style Modbus/TCP device (port 502), the IIoT side of Edge-IIoTset.',
    links: [{ label: 'Modbus specifications', url: 'https://modbus.org/specs.php' }],
  },
  {
    id: 'laptop', name: 'Study Laptop', type: 'laptop', role: 'device', room: 'study',
    ip: '192.168.0.62', mac: '3c:22:fb:00:00:62', vendor: 'Ubuntu laptop',
    protocols: ['HTTPS', 'HTTP', 'DNS', 'SSH'], pos: [7, 0.78, 4.9], rotY: Math.PI,
    controls: [{ key: 'wifi', label: 'Wi-Fi', type: 'toggle' }],
    props: { wifi: true },
    desc: 'Browses and uploads files. Its large uploads are why Uploading is a hard class for the ML layer.',
    links: [{ label: 'LightGBM docs', url: 'https://lightgbm.readthedocs.io/' }],
  },

  // ---- outside the LAN ---------------------------------------------------
  {
    id: 'cloud', name: 'Internet / Cloud', type: 'cloud', role: 'cloud', room: null, wan: true,
    ip: '52.95.110.1', mac: null, vendor: 'Streaming, voice and vendor clouds',
    protocols: ['HTTPS', 'NTP', 'DNS'], pos: [0, 7.5, -15], rotY: 0,
    controls: [], props: {},
    desc: 'Everything outside the home: streaming CDN, voice service, NTP pool, vendor APIs.',
    links: [{ label: 'NIST IoT Cybersecurity Program', url: 'https://www.nist.gov/itl/applied-cybersecurity/nist-cybersecurity-iot-program' }],
  },
  {
    id: 'rogue', name: 'Unknown Device', type: 'rogue', role: 'unknown', room: 'utility',
    ip: '192.168.0.170', mac: 'b8:27:eb:de:ad:70', vendor: 'Unregistered board',
    protocols: ['ARP', 'DNS'], pos: [3.7, 0.08, 5.55], rotY: 0.5,
    controls: [{ key: 'plugged', label: 'Plugged in', type: 'toggle' }],
    props: { plugged: false },
    desc: 'A board nobody registered, hidden behind the washer. SHIELD-IoT flags any MAC that is not in the home inventory. 192.168.0.170 is the address the Edge-IIoTset audit found only in attack traffic.',
    links: [],
  },
];

// Edge-IIoTset has Normal plus 14 attack classes; the ML surrogate outputs over these.
export const CLASSES = [
  'Normal', 'Backdoor', 'DDoS_HTTP', 'DDoS_ICMP', 'DDoS_TCP', 'DDoS_UDP', 'Fingerprinting', 'MITM',
  'Password', 'Port_Scanning', 'Ransomware', 'SQL_injection', 'Uploading', 'Vulnerability_scanner', 'XSS',
];

// Numbers quoted from the SHIELD-IoT research book. Only values the book states.
export const RESEARCH = {
  baseline: {
    model: 'LightGBM', accuracy: 0.8890, macroF1: 0.8872, weightedF1: 0.8898, macroPrecision: 0.9050,
    macroRecall: 0.8780, rocAuc: 0.9926, prAuc: 0.9320, trainSec: 34.79, latencyUs: 35.15,
    features: 37, classes: 15, train: 120849, test: 30213, seed: 42, cleanPool: 151062,
  },
  perClassF1: { DDoS_HTTP: 0.72, Uploading: 0.71, SQL_injection: 0.77, Password: 0.80, Fingerprinting: 0.80, MITM: 1.00 },
  tcpAblation: [
    { drop: 'none (baseline)', f1: 0.8872 }, { drop: 'checksum', f1: 0.8913 }, { drop: 'seq', f1: 0.8737 },
    { drop: 'ack_raw', f1: 0.8576 }, { drop: 'ack_raw + checksum', f1: 0.8685 }, { drop: 'ack', f1: 0.8325 },
    { drop: 'all four', f1: 0.6282 },
  ],
  invariants: {
    INV_TCP_01: { rule: 'SYN and FIN both set', train: 0, test: 0, step: '5A' },
    INV_TCP_02: { rule: 'flags = 0 with tcp.len > 0', train: 286, test: 71, step: '5A' },
    INV_TCP_03: { rule: 'initial SYN carries ACK', train: 0, test: 0, step: '5A' },
    INV_ARP_01: { rule: 'opcode not in {1,2} or hardware size != 6', step: '5B' },
    INV_MQTT: { rule: 'MQTT structural rules (planned)', step: '5C', experimental: true },
  },
  tcpEngineTest: { precision: 1.0, recall: 0.002793, f1: 0.00557, specificity: 1.0, fpr: 0 },
};

export const RESOURCES = [
  { label: 'Edge-IIoTset (IEEE DataPort)', url: 'https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot-applications', note: '15-class IoT/IIoT dataset behind the frozen baseline' },
  { label: 'Edge-IIoTset (Kaggle mirror)', url: 'https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot', note: 'CSV download' },
  { label: 'ToN-IoT datasets (UNSW Canberra)', url: 'https://research.unsw.edu.au/projects/toniot-datasets', note: 'Source of the NF-ToN-IoT flows' },
  { label: 'NetFlow NIDS datasets (UQ)', url: 'https://staff.itee.uq.edu.au/marius/NIDS_datasets/', note: 'NF-ToN-IoT-v3, 27.5 M flows' },
  { label: 'MQTT', url: 'https://mqtt.org/', note: 'Protocol of the planned Step 5C' },
  { label: 'OWASP IoT Top 10', url: 'https://owasp.org/www-project-internet-of-things/', note: 'Common IoT weaknesses' },
  { label: 'NIST IoT Cybersecurity Program', url: 'https://www.nist.gov/itl/applied-cybersecurity/nist-cybersecurity-iot-program', note: 'Device baselines (NISTIR 8259)' },
  { label: 'RFC 9293: TCP', url: 'https://www.rfc-editor.org/rfc/rfc9293', note: 'Basis of INV_TCP_01 to 03' },
  { label: 'RFC 826: ARP', url: 'https://www.rfc-editor.org/rfc/rfc826', note: 'Basis of INV_ARP_01' },
  { label: 'LightGBM', url: 'https://lightgbm.readthedocs.io/', note: 'Baseline classifier' },
  { label: 'three.js', url: 'https://threejs.org/', note: '3D engine of this simulator' },
];

export const deviceById = Object.fromEntries(DEVICES.map(d => [d.id, d]));
