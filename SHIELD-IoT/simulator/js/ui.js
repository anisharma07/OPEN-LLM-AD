// The HUD around the 3D view: top bar, Home network panel with the device
// card, the SHIELD-IoT panel (stats, live chart, evidence layers, drift,
// alerts, blocks), the bottom drawer (packet log, research, resources),
// toasts and the how-to overlay.
//
// Event handlers only record what changed; the DOM is patched in tick() at
// about 4 Hz, reusing rows, so a busy network never rebuilds the page.
// ui.js writes state.ids.mode/layers/threshold/autoBlockAfter and
// state.selectedId, always together with the matching event (CONTRACT.md).

import { HOUSE, RESEARCH, RESOURCES } from './catalog.js';
import { createTrafficChart, createBarChart } from './charts.js';

const RENDER_EVERY_MS = 250;           // DOM patch cadence (~4 Hz)
const RATE_EVERY_MS = 1000;            // per-device tx/rx rate window (real time)
const LOG_BUFFER = 2000;               // packets kept for the log and its filter
const LOG_VIEW = 200;                  // rows shown
const FEED_CAP = 60;                   // alert rows kept in the DOM
const TOAST_MS = 4000;
const TOAST_MAX = 5;
const RESET_CONFIRM_MS = 6000;      // the inline confirm cancels itself when left alone
const SLIDER_EMIT_MS = 120;            // throttle for range/colour commands while dragging
const NARROW_QUERY = '(max-width: 1099.98px)';
const IST_OFFSET_MS = 5.5 * 3600 * 1000;   // India Standard Time, no DST
const PREFS_KEY = 'shield-iot-ui';

const DAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const SEVERITY_RANK = { low: 0, medium: 1, high: 2, critical: 3 };

// Controls that are physical actions in the house rather than network
// commands (network.js handles them even on a blocked device).
const PHYSICAL_CONTROLS = new Set(['phone:wifi', 'laptop:wifi', 'rogue:plugged', 'fridge:doorOpen']);
// The router and the sensor carry every packet, so they cannot be blocked.
const UNBLOCKABLE = new Set(['router', 'shield']);

const LOG_PROTOS = ['TCP', 'UDP', 'ICMP', 'ARP', 'MQTT', 'HTTP', 'HTTPS', 'DNS', 'NTP', 'MODBUS', 'RTP'];
const FLAGGED = '__flagged';

// Protocol families for the mix bar: a fixed colour per family, never per rank.
const MIX_FAMILIES = [
  { key: 'web', label: 'Web', color: 'var(--mix-web)', protos: ['HTTPS', 'HTTP'] },
  { key: 'name', label: 'DNS · NTP', color: 'var(--mix-name)', protos: ['DNS', 'NTP'] },
  { key: 'iot', label: 'MQTT · Modbus', color: 'var(--mix-iot)', protos: ['MQTT', 'MODBUS'] },
  { key: 'media', label: 'RTP · UDP', color: 'var(--mix-media)', protos: ['RTP', 'UDP'] },
  { key: 'other', label: 'TCP · ARP · ICMP', color: 'var(--mix-other)', protos: ['TCP', 'ARP', 'ICMP', 'OTHER'] },
];

// Friendly names and units for live device props that are not controls.
const PROP_INFO = {
  clients: ['Wi-Fi clients'],
  automations: ['Automations'],
  battery: ['Battery', '%'],
  watts: ['Power draw', 'W'],
  current: ['Room temperature', '°C'],
  heartRate: ['Heart rate', 'bpm'],
  steps: ['Steps today'],
  kw: ['Load', 'kW', 2],
  remainingMin: ['Programme left', 'min'],
  listening: ['Listening'],
  music: ['Music playing'],
  ringing: ['Ringing'],
};

const ROLE_LABEL = { infra: 'Infrastructure', device: 'Home device', cloud: 'Internet (WAN)', unknown: 'Not in the inventory' };

const ICONS = {
  ext: '<svg class="ext" viewBox="0 0 12 12" aria-hidden="true"><path d="M5 2.5H2.5v7h7V7M7 2.5h2.5V5M9.3 2.7 5.5 6.5" fill="none" stroke="currentColor" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  close: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 4l8 8M12 4l-8 8" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/></svg>',
  focus: '<svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="2.2" fill="none" stroke="currentColor" stroke-width="1.5"/><path d="M8 1.5v2.5M8 12v2.5M1.5 8H4M12 8h2.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>',
  block: '<svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="5.6" fill="none" stroke="currentColor" stroke-width="1.5"/><path d="M4.1 11.9l7.8-7.8" stroke="currentColor" stroke-width="1.5"/></svg>',
  unblock: '<svg viewBox="0 0 16 16" aria-hidden="true"><rect x="3" y="7" width="10" height="7" rx="1.5" fill="none" stroke="currentColor" stroke-width="1.5"/><path d="M5.5 7V5a2.5 2.5 0 0 1 4.8-1" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>',
  pointer: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 3l14 7-6 2-2 6z" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"/></svg>',
  info: '<svg class="toast-ico" viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="6.3" fill="none" stroke="currentColor" stroke-width="1.4"/><path d="M8 7.2v4M8 4.8v.1" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>',
  ok: '<svg class="toast-ico" viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="6.3" fill="none" stroke="currentColor" stroke-width="1.4"/><path d="M5.2 8.2l1.9 1.9 3.7-4" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  warn: '<svg class="toast-ico" viewBox="0 0 16 16" aria-hidden="true"><path d="M8 2 1.8 13h12.4z" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"/><path d="M8 6.5v3M8 11.2v.1" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>',
  danger: '<svg class="toast-ico" viewBox="0 0 16 16" aria-hidden="true"><path d="M8 1.6 2.6 3.7v4c0 3.3 2.3 6 5.4 6.9 3.1-.9 5.4-3.6 5.4-6.9v-4z" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"/><path d="M8 5.3v3.4M8 10.6v.1" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>',
};

// ---------------------------------------------------------------------------
// Small pure helpers

const pad2 = n => String(n).padStart(2, '0');

/** Sim epoch ms -> parts of the IST wall clock. */
function istParts(epochMs) {
  const d = new Date(epochMs + IST_OFFSET_MS);
  return {
    date: `${DAYS[d.getUTCDay()]} ${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]}`,
    time: `${pad2(d.getUTCHours())}:${pad2(d.getUTCMinutes())}:${pad2(d.getUTCSeconds())}`,
    tenth: Math.floor(d.getUTCMilliseconds() / 100),
  };
}

function fmtInt(n) {
  const v = Math.round(Number(n) || 0);
  if (Math.abs(v) < 1e7) return v.toLocaleString('en-US');
  return `${(v / 1e6).toFixed(1)}M`;
}

function fmtRate(n) {
  if (!Number.isFinite(n) || n <= 0) return '0';
  if (n < 10) return n.toFixed(1).replace(/\.0$/, '');
  return fmtInt(n);
}

function fmtAgo(ms) {
  const s = Math.max(0, Math.round(ms / 1000));
  if (s < 2) return 'now';
  if (s < 60) return `${s} s ago`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m} min ago`;
  return `${Math.floor(m / 60)} h ago`;
}

function fmtUnit(value, unit, step) {
  const digits = step && step < 1 ? 1 : 0;
  const v = typeof value === 'number' ? value.toFixed(digits) : String(value);
  if (!unit) return v;
  return unit === '%' ? `${v}%` : `${v} ${unit}`;
}

/** A live device reading with its unit: integers as-is, fractions to the catalog's precision. */
function fmtProp(key, v) {
  const [, unit, digits] = PROP_INFO[key] || [];
  if (typeof v === 'boolean') return v ? 'yes' : 'no';
  if (typeof v === 'number' && Number.isFinite(v)) {
    const n = digits !== undefined ? v.toFixed(digits) : Number.isInteger(v) ? v.toLocaleString('en-US') : v.toFixed(1);
    return unit ? (unit === '%' ? `${n}%` : `${n} ${unit}`) : n;
  }
  return v === undefined || v === null || v === '' ? '—' : String(v);
}

function humanize(key) {
  return key.replace(/([a-z])([A-Z])/g, '$1 $2').replace(/^./, c => c.toUpperCase());
}

/** Creates an element with an optional class and text. */
function h(tag, cls, text) {
  const el = document.createElement(tag);
  if (cls) el.className = cls;
  if (text !== undefined && text !== null) el.textContent = text;
  return el;
}

function setText(el, text) {
  const t = String(text);
  if (el.textContent !== t) el.textContent = t;
}

function setRangeFill(input) {
  const min = Number(input.min) || 0, max = Number(input.max) || 100;
  const pct = max > min ? ((Number(input.value) - min) / (max - min)) * 100 : 0;
  input.style.setProperty('--fill', `${pct}%`);
}

function loadPrefs() {
  try {
    const raw = window.localStorage.getItem(PREFS_KEY);
    const p = raw ? JSON.parse(raw) : null;
    return p && typeof p === 'object' ? p : {};
  } catch {
    return {};
  }
}

function savePrefs(p) {
  try { window.localStorage.setItem(PREFS_KEY, JSON.stringify(p)); } catch { /* storage unavailable: prefs are a convenience */ }
}

// ---------------------------------------------------------------------------

export function createUI({ bus, state }) {
  const $ = id => document.getElementById(id);
  const body = document.body;
  const narrowMq = window.matchMedia(NARROW_QUERY);
  const isNarrow = () => narrowMq.matches;
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const prefs = loadPrefs();

  const dirty = new Set();             // section keys to patch on the next render
  const dirtyDevices = new Set();      // device rows to patch
  const flag = (...keys) => { for (const k of keys) dirty.add(k); };
  // Direct user actions (selection, blocking, filters) patch the DOM right away
  // instead of waiting for the next 4 Hz batch; busy events never do.
  let soon = null;
  const renderSoon = (...keys) => {
    flag(...keys);
    soon ??= setTimeout(() => { soon = null; render(); }, 0);
  };

  let renderAcc = RENDER_EVERY_MS;     // render on the first tick
  let rateAcc = 0;
  let alertEvents = 0;                 // ids:alert emissions (new rows + repeats)
  let metrics = state.ids.metrics;
  let renderErrors = 0;

  const nameOf = (id, ip) => (id && state.devices.get(id)?.name) || ip || 'unknown';
  const deviceIdOfIp = ip => (ip ? state.byIp.get(ip) ?? null : null);

  // =========================================================================
  // 1. Top bar

  const clockDate = $('clock-date');
  const clockTime = $('clock-time');
  const pauseBtn = $('btn-pause');
  const threatPill = $('threat-pill');
  const threatValue = $('threat-value');
  let lastClock = '';

  function renderClock() {
    const p = istParts(state.time.epochMs + state.time.simMs);
    const key = p.date + p.time;
    if (key === lastClock) return;
    lastClock = key;
    clockDate.textContent = `${p.date} · `;
    clockTime.textContent = p.time;
  }

  function renderSimControls() {
    const paused = !!state.time.paused;
    pauseBtn.setAttribute('aria-pressed', String(paused));
    pauseBtn.setAttribute('aria-label', paused ? 'Resume simulation' : 'Pause simulation');
    pauseBtn.title = paused ? 'Resume (Space)' : 'Pause (Space)';
    const sp = $(`speed-${String(state.time.speed).replace('.', '_')}`);
    if (sp && !sp.checked) sp.checked = true;
    const mode = $(state.ids.mode === 'detect' ? 'mode-detect' : 'mode-prevent');
    if (!mode.checked) mode.checked = true;
  }

  function togglePause() {
    bus.emit('sim:pause', { paused: !state.time.paused });
    renderSimControls();
  }

  pauseBtn.addEventListener('click', togglePause);
  for (const input of document.querySelectorAll('input[name="sim-speed"]')) {
    input.addEventListener('change', () => {
      if (input.checked) bus.emit('sim:speed', { speed: Number(input.value) });
    });
  }
  for (const input of document.querySelectorAll('input[name="ids-mode"]')) {
    input.addEventListener('change', () => {
      if (!input.checked) return;
      state.ids.mode = input.value === 'detect' ? 'detect' : 'prevent';
      bus.emit('ids:config', {});
      showToast({
        kind: 'info',
        text: state.ids.mode === 'detect'
          ? 'Detect only: SHIELD-IoT raises alerts but never drops or blocks automatically.'
          : 'Prevent: SHIELD-IoT drops suspicious packets inline and blocks repeat offenders at the router.',
      });
      flag('tuning');
    });
  }

  function renderThreat() {
    const level = metrics?.threat || 'low';
    if (threatPill.dataset.level !== level) {
      threatPill.dataset.level = level;
      threatValue.textContent = level;
      threatPill.setAttribute('aria-label', `Threat level ${level}`);
    }
  }

  // Reset with an inline two-step confirm.
  const resetBtn = $('btn-reset');
  const resetConfirm = $('reset-confirm');
  let resetTimer = null;
  function openResetConfirm() {
    resetConfirm.hidden = false;
    resetBtn.setAttribute('aria-expanded', 'true');
    $('btn-reset-confirm').focus();
    clearTimeout(resetTimer);
    resetTimer = setTimeout(closeResetConfirm, RESET_CONFIRM_MS);
  }
  function closeResetConfirm(refocus = false) {
    clearTimeout(resetTimer);
    if (resetConfirm.hidden) return;
    resetConfirm.hidden = true;
    resetBtn.setAttribute('aria-expanded', 'false');
    if (refocus) resetBtn.focus();
  }
  resetBtn.addEventListener('click', openResetConfirm);
  // While the pointer or focus is on the confirm, keep it open.
  resetConfirm.addEventListener('pointerenter', () => clearTimeout(resetTimer));
  resetConfirm.addEventListener('focusin', () => clearTimeout(resetTimer));
  resetConfirm.addEventListener('pointerleave', () => {
    clearTimeout(resetTimer);
    if (!resetConfirm.hidden) resetTimer = setTimeout(closeResetConfirm, RESET_CONFIRM_MS);
  });
  $('btn-reset-cancel').addEventListener('click', () => closeResetConfirm(true));
  $('btn-reset-confirm').addEventListener('click', () => {
    closeResetConfirm();
    resetBtn.focus();
    bus.emit('sim:reset', {});
  });

  // How-to overlay.
  const helpDialog = $('help-dialog');
  $('btn-help').addEventListener('click', () => {
    if (typeof helpDialog.showModal === 'function') helpDialog.showModal();
    else helpDialog.setAttribute('open', '');
  });
  $('btn-help-close').addEventListener('click', () => {
    if (typeof helpDialog.close === 'function') helpDialog.close();
    else helpDialog.removeAttribute('open');
  });
  helpDialog.addEventListener('click', e => {
    if (e.target !== helpDialog) return;            // clicks on the backdrop land on the dialog itself
    const r = helpDialog.getBoundingClientRect();
    if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) helpDialog.close();
  });

  // =========================================================================
  // 2. Layout: collapsible panels, drawer tabs, narrow-screen sheet tabs

  const panelHome = $('panel-home');
  const panelShield = $('panel-shield');
  const drawerTabs = [...document.querySelectorAll('#drawer-tabs [role="tab"]')];
  const sheetTabs = [...document.querySelectorAll('#sheet-tabs [role="tab"]')];
  let drawerTab = ['log', 'research', 'resources'].includes(prefs.drawerTab) ? prefs.drawerTab : 'log';
  let sheetTab = 'home';
  let unseenAlerts = 0;

  function persist() {
    savePrefs({
      left: body.dataset.left, right: body.dataset.right, drawer: body.dataset.drawer, drawerTab,
    });
  }

  function setSideCollapsed(side, collapsed) {
    const panel = side === 'left' ? panelHome : panelShield;
    const btn = $(side === 'left' ? 'btn-collapse-left' : 'btn-collapse-right');
    const title = side === 'left' ? 'Home network' : 'SHIELD-IoT';
    panel.classList.toggle('is-collapsed', collapsed);
    body.dataset[side] = collapsed ? 'collapsed' : 'open';
    btn.setAttribute('aria-expanded', String(!collapsed));
    btn.setAttribute('aria-label', `${collapsed ? 'Expand' : 'Collapse'} the ${title} panel`);
    btn.title = collapsed ? 'Expand panel' : 'Collapse panel';
  }
  function setDrawerCollapsed(collapsed) {
    body.dataset.drawer = collapsed ? 'collapsed' : 'open';
    const btn = $('btn-drawer');
    btn.setAttribute('aria-expanded', String(!collapsed));
    btn.setAttribute('aria-label', collapsed ? 'Expand the drawer' : 'Collapse the drawer');
    btn.title = collapsed ? 'Expand drawer' : 'Collapse drawer';
    flag('log');
  }

  $('btn-collapse-left').addEventListener('click', () => {
    setSideCollapsed('left', body.dataset.left !== 'collapsed');
    persist();
  });
  $('btn-collapse-right').addEventListener('click', () => {
    setSideCollapsed('right', body.dataset.right !== 'collapsed');
    persist();
  });
  $('btn-drawer').addEventListener('click', () => {
    setDrawerCollapsed(body.dataset.drawer !== 'collapsed');
    persist();
  });

  function selectDrawerTab(tab, { focus = false } = {}) {
    drawerTab = tab;
    for (const t of drawerTabs) {
      const on = t.dataset.dtab === tab;
      t.setAttribute('aria-selected', String(on));
      t.tabIndex = on ? 0 : -1;
      if (on && focus) t.focus();
    }
    $('pane-log').hidden = tab !== 'log';
    $('pane-research').hidden = tab !== 'research';
    $('pane-resources').hidden = tab !== 'resources';
    if (tab === 'research') researchCharts.forEach(c => c.redraw());
    renderSoon('log');
  }

  function selectSheetTab(tab, { focus = false } = {}) {
    sheetTab = tab;
    body.dataset.mtab = tab;
    for (const t of sheetTabs) {
      const on = t.dataset.mtab === tab;
      t.setAttribute('aria-selected', String(on));
      t.tabIndex = on ? 0 : -1;
      if (on && focus) t.focus();
    }
    if (tab === 'log' || tab === 'research' || tab === 'resources') selectDrawerTab(tab);
    if (tab === 'shield') unseenAlerts = 0;
    renderSoon('badge', 'log');
  }

  /** Arrow-key navigation inside a tablist (roving tabindex). */
  function tabKeys(tabs, attr, select) {
    for (const t of tabs) {
      t.addEventListener('click', () => select(t.dataset[attr]));
      t.addEventListener('keydown', e => {
        const i = tabs.indexOf(t);
        let j = null;
        if (e.key === 'ArrowRight') j = (i + 1) % tabs.length;
        else if (e.key === 'ArrowLeft') j = (i - 1 + tabs.length) % tabs.length;
        else if (e.key === 'Home') j = 0;
        else if (e.key === 'End') j = tabs.length - 1;
        if (j === null) return;
        e.preventDefault();
        select(tabs[j].dataset[attr], { focus: true });
      });
    }
  }
  tabKeys(drawerTabs, 'dtab', (tab, o) => { selectDrawerTab(tab, o); persist(); });
  tabKeys(sheetTabs, 'mtab', selectSheetTab);

  function renderBadge() {
    const badge = $('mtab-shield-badge');
    const show = unseenAlerts > 0 && sheetTab !== 'shield';
    badge.hidden = !show;
    if (show) setText(badge, unseenAlerts > 99 ? '99+' : unseenAlerts);
  }

  // =========================================================================
  // 3. Home network: rogue callout and the device list

  const listEl = $('device-list');
  const roguePlug = $('rogue-plugged');
  const rows = new Map();              // device id -> row refs
  const roomHeads = [];                // { head, ul, countEl, ids } per room group
  const rates = new Map();             // device id -> { tx, rx } per sim second
  let rateBase = new Map();            // device id -> { tx, rx } at the last sample
  let rateBaseSim = state.time.simMs;

  roguePlug.addEventListener('change', () => {
    bus.emit('device:command', { id: 'rogue', key: 'plugged', value: roguePlug.checked });
  });
  function renderRogueToggle() {
    const rogue = state.devices.get('rogue');
    roguePlug.disabled = !rogue;
    const plugged = !!rogue?.props.plugged;
    if (roguePlug.checked !== plugged) roguePlug.checked = plugged;
  }

  function isVisibleInList(d) {
    return d.role !== 'unknown' || !!d.props.plugged;
  }

  function statusChip(chip, status) {
    const cls = `chip chip-${status}`;
    if (chip.className !== cls) chip.className = cls;
    setText(chip, status);
  }

  function buildDeviceList() {
    listEl.replaceChildren();
    rows.clear();
    roomHeads.length = 0;
    const groups = HOUSE.rooms.map(r => ({ id: r.id, name: r.name, devices: [] }));
    const wan = { id: '__wan', name: 'Internet', devices: [] };
    const other = { id: '__other', name: 'Other', devices: [] };
    const byRoom = new Map(groups.map(g => [g.id, g]));
    for (const d of state.devices.values()) {
      if (d.role === 'cloud' || d.wan) wan.devices.push(d);
      else (byRoom.get(d.room) || other).devices.push(d);
    }
    for (const g of [...groups, wan, other]) {
      if (!g.devices.length) continue;
      const head = h('div', 'room-head');
      head.id = `room-${g.id.replace(/^_+/, '')}`;
      head.append(h('span', null, g.name));
      const count = h('span', 'count');
      head.append(count);
      const ul = h('ul', 'room-list');
      ul.setAttribute('aria-labelledby', head.id);
      for (const d of g.devices) {
        const li = h('li', 'dev-item');
        const btn = h('button', 'dev-row');
        btn.type = 'button';
        btn.id = `dev-row-${d.id}`;
        btn.setAttribute('aria-pressed', 'false');
        const name = h('span', 'dev-name', d.name);
        const chips = h('span', 'dev-chips');
        const unreg = h('span', 'chip chip-unregistered no-dot', 'Unregistered');
        unreg.hidden = d.role !== 'unknown';
        const status = h('span', 'chip');
        chips.append(unreg, status);
        const ip = h('span', 'dev-ip mono', d.ip);
        const rate = h('span', 'dev-rate mono');
        rate.title = 'Packets per second sent (up) and received (down)';
        const tx = h('b'), rx = h('b');
        rate.append(tx, h('span', 'arrow', '↑'), rx, h('span', 'arrow', '↓'));
        btn.append(name, chips, ip, rate);
        btn.addEventListener('click', () => selectDevice(d.id, { focus: true, from: 'list' }));
        li.append(btn);
        ul.append(li);
        rows.set(d.id, { li, btn, status, unreg, tx, rx, sig: '' });
      }
      listEl.append(head, ul);
      roomHeads.push({ countEl: count, head, ul, ids: g.devices.map(d => d.id) });
    }
    for (const id of rows.keys()) dirtyDevices.add(id);
    flag('rooms');
  }

  function renderDeviceRow(id) {
    const r = rows.get(id);
    const d = state.devices.get(id);
    if (!r || !d) return;
    const visible = isVisibleInList(d);
    if (r.li.hidden === visible) { r.li.hidden = !visible; flag('rooms'); }
    const status = d.status || 'ok';
    const sel = state.selectedId === id;
    const sig = `${status}|${sel}`;
    if (sig !== r.sig) {
      r.sig = sig;
      statusChip(r.status, status);
      r.status.hidden = d.role === 'unknown' && status === 'ok';
      r.btn.setAttribute('aria-pressed', String(sel));
      r.btn.classList.toggle('is-offline', status === 'offline');
      r.btn.setAttribute('aria-label', `${d.name}, ${d.ip}, ${d.role === 'unknown' ? 'unregistered, ' : ''}${status}`);
    }
  }

  function renderRates() {
    for (const [id, r] of rows) {
      const v = rates.get(id);
      setText(r.tx, fmtRate(v?.tx ?? 0));
      setText(r.rx, fmtRate(v?.rx ?? 0));
    }
  }

  function renderRooms() {
    let online = 0, total = 0, blocked = 0;
    for (const g of roomHeads) {
      const n = g.ids.filter(id => { const d = state.devices.get(id); return d && isVisibleInList(d); }).length;
      setText(g.countEl, n);
      g.head.hidden = n === 0;
      g.ul.hidden = n === 0;
    }
    for (const d of state.devices.values()) {
      if (d.role === 'cloud' || !isVisibleInList(d)) continue;
      total++;
      if (d.online) online++;
      if (d.status === 'blocked') blocked++;
    }
    const meta = $('home-meta');
    setText(meta, `${online}/${total} online`);
    meta.title = `${online} of ${total} home devices online${blocked ? `, ${blocked} blocked at the router` : ''}`;
  }

  /** Samples device tx/rx counters into per-second rates (sim time). */
  function sampleRates() {
    const now = state.time.simMs;
    const dt = now - rateBaseSim;
    if (dt < 0) { resetRates(); return; }                // sim reset
    if (dt === 0) return;                                 // paused: keep the last rates
    const next = new Map();
    for (const d of state.devices.values()) {
      const prev = rateBase.get(d.id);
      const s = d.stats || { tx: 0, rx: 0 };
      if (prev) rates.set(d.id, { tx: Math.max(0, s.tx - prev.tx) * 1000 / dt, rx: Math.max(0, s.rx - prev.rx) * 1000 / dt });
      next.set(d.id, { tx: s.tx, rx: s.rx });
    }
    rateBase = next;
    rateBaseSim = now;
    flag('rates');
    if (state.selectedId) flag('cardStats');
  }
  function resetRates() {
    rates.clear();
    rateBase = new Map([...state.devices.values()].map(d => [d.id, { tx: d.stats?.tx ?? 0, rx: d.stats?.rx ?? 0 }]));
    rateBaseSim = state.time.simMs;
    flag('rates');
  }

  // =========================================================================
  // 4. Selection and the selected-device card

  const cardEl = $('device-card');
  let card = null;                     // refs for the rendered card, or null when empty

  function selectDevice(id, { focus = false, from = null } = {}) {
    const next = id && state.devices.has(id) ? id : null;
    if (state.selectedId !== next) {
      state.selectedId = next;
      bus.emit('device:select', { id: next });
    } else {
      flag('card');
    }
    if (next && focus) bus.emit('scene:focus', { id: next });
    if (next && isNarrow()) {
      if (sheetTab !== 'home') selectSheetTab('home');
      if (from) requestAnimationFrame(() => cardEl.scrollIntoView({ block: 'start', behavior: reducedMotion.matches ? 'auto' : 'smooth' }));
    }
  }

  function onSelect(id) {
    const prev = card?.id ?? null;
    if (state.selectedId !== (id ?? null)) state.selectedId = id ?? null;
    if (prev) dirtyDevices.add(prev);
    if (id) dirtyDevices.add(id);
    renderSoon('card');
    // After the card has rendered (it changes the list's height), bring the row into view.
    if (id && !isNarrow()) setTimeout(() => revealRow(id), 0);
  }

  /** Scrolls the device list (only the list) so the selected row is visible. */
  function revealRow(id) {
    const wrap = $('device-list-wrap');
    const row = rows.get(id)?.btn;
    if (!row || row.closest('li')?.hidden) return;
    const w = wrap.getBoundingClientRect();
    const r = row.getBoundingClientRect();
    const head = 26;                                     // sticky column header
    if (r.top < w.top + head) wrap.scrollTop -= w.top + head - r.top + 4;
    else if (r.bottom > w.bottom) wrap.scrollTop += r.bottom - w.bottom + 4;
  }

  function controlId(d, c) { return `ctl-${d.id}-${c.key}`; }

  /** Throttles live commands from sliders and colour pickers while dragging. */
  function makeEmitter(deviceId, key) {
    let last = 0, timer = null, pending;
    const send = v => { last = performance.now(); bus.emit('device:command', { id: deviceId, key, value: v }); };
    return {
      live(v) {
        pending = v;
        const wait = SLIDER_EMIT_MS - (performance.now() - last);
        if (wait <= 0) { clearTimeout(timer); timer = null; send(v); }
        else if (!timer) timer = setTimeout(() => { timer = null; send(pending); }, wait);
      },
      final(v) { clearTimeout(timer); timer = null; send(v); },
    };
  }

  function buildControl(d, c) {
    const id = controlId(d, c);
    const ref = { c, physical: PHYSICAL_CONTROLS.has(`${d.type}:${c.key}`) };
    let wrap;
    switch (c.type) {
      case 'toggle': {
        wrap = h('div', 'ctl ctl-toggle');
        const label = h('label', 'switch');
        label.htmlFor = id;
        const input = h('input');
        input.type = 'checkbox';
        input.id = id;
        input.setAttribute('role', 'switch');
        input.addEventListener('change', () => bus.emit('device:command', { id: d.id, key: c.key, value: input.checked }));
        label.append(input, h('span', 'switch-track'), h('span', 'switch-text', c.label));
        label.querySelector('.switch-track').setAttribute('aria-hidden', 'true');
        wrap.append(label);
        Object.assign(ref, { input, read: v => { if (input.checked !== !!v) input.checked = !!v; } });
        break;
      }
      case 'range': {
        wrap = h('div', 'ctl ctl-range');
        const line = h('div', 'ctl-line');
        const label = h('label', null, c.label);
        label.htmlFor = id;
        const out = h('output', 'mono');
        out.htmlFor = id;
        line.append(label, out);
        const input = h('input');
        input.type = 'range';
        input.id = id;
        input.min = c.min ?? 0;
        input.max = c.max ?? 100;
        input.step = c.step ?? 1;
        const emit = makeEmitter(d.id, c.key);
        const show = () => { out.value = fmtUnit(Number(input.value), c.unit, c.step); setRangeFill(input); };
        input.addEventListener('input', () => { show(); emit.live(Number(input.value)); });
        input.addEventListener('change', () => emit.final(Number(input.value)));
        input.addEventListener('blur', () => flag('card'));
        wrap.append(line, input);
        Object.assign(ref, {
          input,
          read: v => {
            if (document.activeElement === input) return;     // do not fight the user's drag
            const n = Number(v);
            if (Number.isFinite(n) && Number(input.value) !== n) input.value = String(n);
            show();
          },
        });
        break;
      }
      case 'select': {
        wrap = h('div', 'ctl ctl-inline');
        const label = h('label', null, c.label);
        label.htmlFor = id;
        const sel = h('select');
        sel.id = id;
        const options = c.options || [];
        options.forEach((o, i) => {
          const opt = h('option', null, String(o));
          opt.value = String(i);
          sel.append(opt);
        });
        sel.addEventListener('change', () => {
          const o = options[Number(sel.value)];
          if (o !== undefined) bus.emit('device:command', { id: d.id, key: c.key, value: o });
        });
        sel.addEventListener('blur', () => flag('card'));
        wrap.append(label, sel);
        Object.assign(ref, {
          input: sel,
          read: v => {
            if (document.activeElement === sel) return;
            const i = options.findIndex(o => String(o) === String(v));
            if (i >= 0 && sel.value !== String(i)) sel.value = String(i);
          },
        });
        break;
      }
      case 'color': {
        wrap = h('div', 'ctl ctl-inline ctl-color');
        const label = h('label', null, c.label);
        label.htmlFor = id;
        const pair = h('span', 'color-pair');
        const hex = h('span', 'hex mono');
        const input = h('input');
        input.type = 'color';
        input.id = id;
        const emit = makeEmitter(d.id, c.key);
        input.addEventListener('input', () => { hex.textContent = input.value; emit.live(input.value); });
        input.addEventListener('change', () => emit.final(input.value));
        input.addEventListener('blur', () => flag('card'));
        pair.append(hex, input);
        wrap.append(label, pair);
        Object.assign(ref, {
          input,
          read: v => {
            if (document.activeElement === input) return;
            const val = /^#[0-9a-f]{6}$/i.test(String(v)) ? String(v).toLowerCase() : '#ffffff';
            if (input.value !== val) input.value = val;
            setText(hex, val);
          },
        });
        break;
      }
      case 'button': {
        const btn = h('button', 'btn', c.label);
        btn.type = 'button';
        btn.id = id;
        btn.addEventListener('click', () => bus.emit('device:command', { id: d.id, key: c.key }));
        Object.assign(ref, { input: btn, read: () => {}, button: true, el: btn });
        return ref;
      }
      default:
        return null;
    }
    ref.el = wrap;
    return ref;
  }

  function kvList(pairs) {
    const dl = h('dl', 'kv');
    for (const [k, v, mono] of pairs) {
      dl.append(h('dt', null, k), h('dd', mono ? 'mono' : null, v));
    }
    return dl;
  }

  function cardSection(title, ...children) {
    const sec = h('div', 'card-sec');
    if (title) sec.append(h('h4', 'card-sec-title', title));
    sec.append(...children);
    return sec;
  }

  function buildCard() {
    const d = state.selectedId ? state.devices.get(state.selectedId) : null;
    cardEl.replaceChildren();
    card = null;
    if (!d) {
      const empty = h('div', 'card-empty');
      empty.innerHTML = ICONS.pointer;
      empty.append(h('p', null, 'Select a device in the list or in the 3D view to control it and see its traffic.'));
      cardEl.append(empty);
      cardEl.removeAttribute('aria-label');
      return;
    }
    cardEl.setAttribute('aria-label', `${d.name} details`);
    const refs = { id: d.id, controls: [], props: new Map() };

    // Head: name, chips, vendor, clear button.
    const head = h('header', 'card-head');
    const titleRow = h('div', 'card-title');
    const title = h('h3', null, d.name);
    title.id = 'card-title';
    refs.status = h('span', 'chip');
    titleRow.append(title, refs.status);
    if (d.role === 'unknown') titleRow.append(h('span', 'chip chip-unregistered no-dot', 'Unregistered'));
    const close = h('button', 'btn btn-icon btn-ghost card-close');
    close.type = 'button';
    close.id = 'card-close';
    close.setAttribute('aria-label', 'Clear the selection');
    close.title = 'Clear selection (Esc)';
    close.innerHTML = ICONS.close;
    close.addEventListener('click', () => selectDevice(null));
    head.append(titleRow, close, h('p', 'card-vendor', d.vendor || ''));
    cardEl.append(head);

    refs.banner = h('div', 'card-banner');
    refs.banner.hidden = true;
    refs.banner.setAttribute('role', 'note');
    cardEl.append(refs.banner);

    // Controls, generic from the catalog descriptors; buttons share one row.
    if (d.controls?.length) {
      const box = h('div', 'controls');
      let btnRow = null;
      for (const c of d.controls) {
        const ref = buildControl(d, c);
        if (!ref) continue;
        refs.controls.push(ref);
        if (ref.button) {
          if (!btnRow) { btnRow = h('div', 'ctl-buttons'); box.append(btnRow); }
          btnRow.append(ref.el);
        } else {
          btnRow = null;
          box.append(ref.el);
        }
      }
      cardEl.append(cardSection('Controls', box));
    }

    // Actions: block / unblock on the router firewall, focus the camera.
    const actions = h('div', 'card-actions');
    if (!UNBLOCKABLE.has(d.id)) {
      refs.blockBtn = h('button', 'btn btn-danger');
      refs.blockBtn.type = 'button';
      refs.blockBtn.id = 'card-block';
      refs.blockBtn.addEventListener('click', () => {
        const dev = state.devices.get(refs.id);
        if (!dev) return;
        const blocked = dev.blocked || state.ids.blocked.has(dev.ip);
        bus.emit(blocked ? 'ids:request-unblock' : 'ids:request-block', { ip: dev.ip });
        renderSoon('card');
      });
      actions.append(refs.blockBtn);
    }
    const focusBtn = h('button', 'btn btn-ghost');
    focusBtn.type = 'button';
    focusBtn.id = 'card-focus';
    focusBtn.innerHTML = `${ICONS.focus}<span>Focus camera</span>`;
    focusBtn.addEventListener('click', () => bus.emit('scene:focus', { id: d.id }));
    actions.append(focusBtn);
    cardEl.append(actions);

    // Live props that are not controls, then traffic counters.
    const controlKeys = new Set((d.controls || []).map(c => c.key));
    const propKeys = Object.keys(d.props || {}).filter(k => !controlKeys.has(k) && k !== 'say');
    if (propKeys.length) {
      const dl = h('dl', 'kv');
      for (const k of propKeys) {
        const dd = h('dd', 'mono');
        dl.append(h('dt', null, PROP_INFO[k]?.[0] ?? humanize(k)), dd);
        refs.props.set(k, dd);
      }
      cardEl.append(cardSection('Live readings', dl));
    }
    const stats = h('div', 'card-stats');
    const mini = (label) => {
      const box = h('div', 'mini-stat');
      const b = h('b');
      const small = h('small');
      box.append(h('span', null, label), b, small);
      stats.append(box);
      return { b, small };
    };
    refs.sTx = mini('Sent');
    refs.sRx = mini('Received');
    refs.sDrop = mini('Dropped');
    cardEl.append(cardSection('Traffic (packets)', stats));

    // Identity, protocols, description, links.
    const roomName = HOUSE.rooms.find(r => r.id === d.room)?.name ?? (d.wan ? 'Outside the home' : '—');
    cardEl.append(cardSection('Network identity', kvList([
      ['IP', d.ip || '—', true],
      ['MAC', d.mac || 'n/a (beyond the router)', true],
      ['Room', roomName],
      ['Role', ROLE_LABEL[d.role] ?? d.role],
    ])));
    if (d.protocols?.length) {
      const chips = h('div', 'chip-row');
      for (const p of d.protocols) chips.append(h('span', 'chip chip-proto', p));
      cardEl.append(cardSection('Protocols', chips));
    }
    if (d.desc) cardEl.append(cardSection('About', h('p', 'card-desc', d.desc)));
    if (d.links?.length) {
      const ul = h('ul', 'card-links');
      for (const l of d.links) {
        const li = h('li');
        const a = h('a');
        a.href = l.url;
        a.target = '_blank';
        a.rel = 'noopener';
        a.append(document.createTextNode(l.label));
        a.insertAdjacentHTML('beforeend', ICONS.ext);
        li.append(a);
        ul.append(li);
      }
      cardEl.append(cardSection('Read more', ul));
    }
    card = refs;
    updateCard();
    updateCardStats();
  }

  function offlineReason(d) {
    if (d.role === 'unknown' && !d.props.plugged) return 'Unplugged: the board is not on the network.';
    if (d.props.power === false) return 'Switched off.';
    if (d.props.wifi === false) return 'Wi-Fi is off on this device.';
    const router = state.devices.get('router');
    if (router && !router.online && d.id !== 'router') return 'The router is rebooting, so the device is unreachable.';
    if (d.id === 'router') return 'Rebooting: the home network is down for a moment.';
    return 'Not reachable.';
  }

  function updateCard() {
    if (!card) return;
    const d = state.devices.get(card.id);
    if (!d) { buildCard(); return; }
    const status = d.status || 'ok';
    statusChip(card.status, status);
    const block = state.ids.blocked.get(d.ip);
    const blocked = !!(d.blocked || block);

    // Banner: why the device is blocked, offline or flagged.
    const b = card.banner;
    if (blocked) {
      b.hidden = false;
      b.dataset.kind = 'blocked';
      b.replaceChildren();
      const who = block?.auto ? 'automatically by SHIELD-IoT' : 'by the operator';
      const reason = block?.reason && !/^blocked by the operator\.?$/i.test(block.reason) ? `: ${block.reason}` : '';
      b.append(h('b', null, 'Blocked at the router '), document.createTextNode(`${who}${reason}. `));
      const physical = card.controls.some(r => r.physical);
      b.append(document.createTextNode(`Network commands are disabled${physical ? '; physical switches still work' : ''}.`));
    } else if (status === 'offline') {
      b.hidden = false;
      b.dataset.kind = 'offline';
      b.replaceChildren(h('b', null, 'Offline. '), document.createTextNode(offlineReason(d)));
    } else if (d.role === 'unknown') {
      b.hidden = false;
      b.dataset.kind = 'unregistered';
      b.replaceChildren(h('b', null, 'Not in the home inventory. '),
        document.createTextNode('SHIELD-IoT flags every packet from this MAC address.'));
    } else {
      b.hidden = true;
    }

    for (const ref of card.controls) {
      ref.read(d.props?.[ref.c.key]);
      const disabled = blocked && !ref.physical;
      if (ref.input.disabled !== disabled) ref.input.disabled = disabled;
      if (disabled) ref.input.title = 'Disabled: the device is blocked at the router';
      else ref.input.removeAttribute('title');
    }

    for (const [k, dd] of card.props) setText(dd, fmtProp(k, d.props?.[k]));

    if (card.blockBtn) {
      const label = blocked ? 'Unblock' : 'Block on router';
      if (card.blockBtn.dataset.state !== String(blocked)) {
        card.blockBtn.dataset.state = String(blocked);
        card.blockBtn.className = blocked ? 'btn btn-ok' : 'btn btn-danger';
        card.blockBtn.innerHTML = `${blocked ? ICONS.unblock : ICONS.block}<span>${label}</span>`;
      }
    }
  }

  function updateCardStats() {
    if (!card) return;
    const d = state.devices.get(card.id);
    if (!d) return;
    const r = rates.get(d.id);
    setText(card.sTx.b, fmtInt(d.stats?.tx ?? 0));
    setText(card.sRx.b, fmtInt(d.stats?.rx ?? 0));
    setText(card.sDrop.b, fmtInt(d.stats?.dropped ?? 0));
    setText(card.sTx.small, `${fmtRate(r?.tx ?? 0)}/s`);
    setText(card.sRx.small, `${fmtRate(r?.rx ?? 0)}/s`);
    setText(card.sDrop.small, '');
  }

  // =========================================================================
  // 5. SHIELD-IoT panel: stats, chart, layers, tuning, drift, protocol mix

  const chart = createTrafficChart($('traffic-chart'), { windowMs: 60000, label: 'Live traffic' });
  const lastMarkAt = new Map();        // alert id -> sim time of its last chart marker

  const layerInputs = [...document.querySelectorAll('#layer-list input[data-layer]')];
  for (const input of layerInputs) {
    input.addEventListener('change', () => {
      state.ids.layers[input.dataset.layer] = input.checked;
      bus.emit('ids:config', {});
      renderSoon('metrics', 'tuning');
    });
  }
  const thr = $('ids-threshold');
  const autoBlock = $('ids-autoblock');
  thr.addEventListener('input', () => {
    state.ids.threshold = Math.round(Number(thr.value) * 100) / 100;
    bus.emit('ids:config', {});
    renderTuning();
  });
  autoBlock.addEventListener('input', () => {
    state.ids.autoBlockAfter = Math.round(Number(autoBlock.value));
    bus.emit('ids:config', {});
    renderTuning();
  });

  function renderTuning() {
    for (const input of layerInputs) {
      const on = !!state.ids.layers[input.dataset.layer];
      if (input.checked !== on) input.checked = on;
    }
    const t = Number(state.ids.threshold) || 0.8;
    if (document.activeElement !== thr) thr.value = String(Math.min(0.99, Math.max(0.5, t)));
    $('ids-threshold-out').value = t.toFixed(2);
    setRangeFill(thr);
    const a = Math.round(Number(state.ids.autoBlockAfter) || 5);
    if (document.activeElement !== autoBlock) autoBlock.value = String(Math.min(20, Math.max(1, a)));
    $('ids-autoblock-out').value = `${a} alert${a === 1 ? '' : 's'}`;
    setRangeFill(autoBlock);
    const detect = state.ids.mode === 'detect';
    setText($('autoblock-hint'), detect
      ? 'Detect only is on, so nothing is blocked automatically. Switch to Prevent to enable auto-blocking.'
      : 'Alerts from one source within 10 s before it is blocked at the router.');
  }

  function renderStats() {
    const m = metrics;
    setText($('stat-inspected'), fmtInt(m?.inspected ?? 0));
    const alertsEl = $('stat-alerts');
    setText(alertsEl, fmtInt(alertEvents));
    alertsEl.classList.toggle('has-value', alertEvents > 0);
    const dropped = (m?.dropped ?? 0) + (m?.firewallDrops ?? 0);
    const dropEl = $('stat-dropped');
    setText(dropEl, fmtInt(dropped));
    dropEl.classList.toggle('has-value', dropped > 0);
    dropEl.title = `${fmtInt(m?.dropped ?? 0)} dropped inline by SHIELD-IoT, ${fmtInt(m?.firewallDrops ?? 0)} by the router firewall`;
    const blockedEl = $('stat-blocked');
    setText(blockedEl, state.ids.blocked.size);
    blockedEl.classList.toggle('has-value', state.ids.blocked.size > 0);
    setText($('stat-pps'), fmtInt(m?.pps ?? 0));
    const lat = Number(m?.avgLatencyUs ?? 0);
    setText($('stat-latency'), lat >= 100 ? Math.round(lat) : lat.toFixed(1));
    $('stat-alerts').title = 'Alert events raised (repeats from the same source and class are grouped in the feed)';
    $('stat-pps').title = 'Packets per second of sim time, allowed plus dropped';
    $('stat-latency').parentElement.title = 'Mean per-packet inspection time measured in this browser (the book reports 35.15 µs/sample for LightGBM)';
  }

  function renderLayerHits() {
    const L = metrics?.layerHits || {};
    const layers = state.ids.layers;
    const sum = keys => keys.reduce((s, k) => s + (Number(L[k]) || 0), 0);
    const show = (id, value, on, title) => {
      const el = $(id);
      setText(el, on ? fmtInt(value) : 'off');
      el.classList.toggle('has-value', on && value > 0);
      el.classList.toggle('is-off', !on);
      if (title) el.title = title;
    };
    show('hits-ml', Number(L.ml) || 0, layers.ml, 'Packets where the ML surrogate leaned towards an attack class');
    show('hits-tcp', sum(['INV_TCP_01', 'INV_TCP_02', 'INV_TCP_03']), layers.tcp,
      `INV_TCP_01 ${L.INV_TCP_01 || 0} · INV_TCP_02 ${L.INV_TCP_02 || 0} · INV_TCP_03 ${L.INV_TCP_03 || 0}`);
    show('hits-arp', Number(L.INV_ARP_01) || 0, layers.arp, `INV_ARP_01 ${L.INV_ARP_01 || 0}`);
    show('hits-mqtt', sum(['INV_MQTT_01', 'INV_MQTT_02', 'INV_MQTT_03']), layers.mqtt,
      `INV_MQTT_01 ${L.INV_MQTT_01 || 0} · INV_MQTT_02 ${L.INV_MQTT_02 || 0} · INV_MQTT_03 ${L.INV_MQTT_03 || 0}`);
    const driftEl = $('hits-drift');
    const psi = Number(metrics?.drift?.psi) || 0;
    setText(driftEl, layers.drift ? `PSI ${psi.toFixed(2)}` : 'off');
    driftEl.classList.toggle('is-off', !layers.drift);
    driftEl.classList.toggle('has-value', layers.drift && metrics?.drift?.state !== 'stable' && !!metrics?.drift?.state);
    show('hits-inventory', Number(L.inventory) || 0, true);
  }

  function renderDrift() {
    const on = !!state.ids.layers.drift;
    const dr = metrics?.drift || { psi: 0, state: 'stable', phase: on ? 'learning' : 'off' };
    const psi = Math.max(0, Number(dr.psi) || 0);
    const st = on ? (dr.state || 'stable') : 'off';
    const stateEl = $('drift-state');
    if (stateEl.dataset.state !== st) { stateEl.dataset.state = st; stateEl.textContent = st; }
    $('drift-needle').style.left = `${Math.min(psi / 0.5, 1) * 100}%`;
    setText($('drift-psi'), psi.toFixed(3));
    const phase = !on ? 'monitor off' : dr.phase === 'learning' ? 'learning the reference' : dr.phase === 'off' ? 'monitor off' : 'monitoring';
    setText($('drift-phase'), phase);
    setText($('piws-count'), fmtInt(metrics?.piwsBuffer ?? 0));
    const g = $('drift-gauge');
    g.setAttribute('aria-valuenow', String(Math.min(psi, 0.5)));
    g.setAttribute('aria-valuetext', `PSI ${psi.toFixed(3)}, ${st}`);
  }

  // Protocol mix: segments and legend items are built once and updated in place.
  const mixBar = $('mix-bar');
  const mixLegend = $('mix-legend');
  const mixRefs = MIX_FAMILIES.map(f => {
    const seg = h('span', 'mix-seg');
    seg.style.background = f.color;
    seg.style.flexGrow = '0';
    mixBar.append(seg);
    const li = h('li');
    const sw = h('span', 'sw');
    sw.style.background = f.color;
    sw.setAttribute('aria-hidden', 'true');
    const pct = h('b', null, '0%');
    li.append(sw, h('span', null, f.label), pct);
    li.tabIndex = 0;
    mixLegend.append(li);
    return { f, seg, li, pct };
  });

  function renderMix() {
    const mix = metrics?.protoMix || {};
    const known = new Set(MIX_FAMILIES.flatMap(f => f.protos));
    let total = 0;
    for (const n of Object.values(mix)) total += Number(n) || 0;
    const parts = [];
    for (const r of mixRefs) {
      const protos = r.f.key === 'other'
        ? [...r.f.protos, ...Object.keys(mix).filter(p => !known.has(p))]
        : r.f.protos;
      const items = protos.map(p => [p, Number(mix[p]) || 0]).filter(([, n]) => n > 0);
      const n = items.reduce((s, [, v]) => s + v, 0);
      const pct = total ? (n / total) * 100 : 0;
      r.seg.style.flexGrow = String(n);
      r.seg.hidden = n === 0;
      setText(r.pct, `${pct < 1 && n > 0 ? '<1' : Math.round(pct)}%`);
      const detail = items.length ? items.map(([p, v]) => `${p} ${fmtInt(v)}`).join(', ') : 'none';
      r.li.title = `${r.f.label}: ${detail}`;
      r.seg.title = r.li.title;
      parts.push(`${r.f.label} ${Math.round(pct)}%`);
    }
    setText($('mix-total'), `${fmtInt(total)} pkts`);
    mixBar.setAttribute('aria-label', total ? `Protocol mix over the last 10 s: ${parts.join(', ')}` : 'Protocol mix: no packets yet');

    const rate = Number(metrics?.falseAlarmRate) || 0;
    setText($('fa-rate'), `${(rate * 100).toFixed(2)}%`);
    setText($('fa-count'), `${fmtInt(metrics?.falseAlarms ?? 0)} false alarm${metrics?.falseAlarms === 1 ? '' : 's'}`);
  }

  // =========================================================================
  // 6. Alert feed and blocked list

  const feedEl = $('alert-feed');
  const feedRows = new Map();          // alert id -> row refs
  let feedPrimed = false;              // rows created on the first render are not "new"

  function alertRoute(a) {
    const srcId = a.srcId ?? deviceIdOfIp(a.srcIp);
    return [nameOf(srcId, a.srcIp), a.dstId ? nameOf(a.dstId) : null];
  }

  function makeAlertRow(a) {
    const li = h('li', 'alert-item');
    const btn = h('button', 'alert-row');
    btn.type = 'button';
    btn.id = `alert-${a.id}`;
    const cls = h('span', 'a-cls');
    const meta = h('span', 'a-meta');
    const sev = h('span', 'a-sev');
    const count = h('span', 'a-count');
    meta.append(sev, count);
    const route = h('span', 'a-route');
    const rule = h('span', 'a-rule');
    const foot = h('span', 'a-foot');
    const act = h('span', 'a-act');
    const time = h('span', 'a-time');
    foot.append(act, time);
    btn.append(cls, meta, route, rule, foot);
    btn.addEventListener('click', () => {
      const al = state.ids.alerts.find(x => x.id === a.id) || a;
      const id = al.srcId ?? deviceIdOfIp(al.srcIp);
      if (id) selectDevice(id, { from: 'feed' });
    });
    li.append(btn);
    return { li, btn, cls, sev, count, route, rule, act, time, sig: '' };
  }

  function renderFeed() {
    const list = state.ids.alerts.slice(0, FEED_CAP);
    const keep = new Set();
    const now = state.time.simMs;
    list.forEach((a, i) => {
      let r = feedRows.get(a.id);
      if (!r) {
        r = makeAlertRow(a);
        feedRows.set(a.id, r);
        if (feedPrimed && !reducedMotion.matches) {
          r.btn.classList.add('is-new');
          r.btn.addEventListener('animationend', () => r.btn.classList.remove('is-new'), { once: true });
        }
      }
      keep.add(a.id);
      const sig = `${a.cls}|${a.severity}|${a.count}|${a.ruleId}|${a.score}|${a.action}|${a.srcId}|${a.dstId}`;
      if (sig !== r.sig) {
        r.sig = sig;
        r.btn.dataset.sev = a.severity;
        setText(r.cls, a.cls || 'Alert');
        setText(r.sev, a.severity);
        r.count.hidden = !(a.count > 1);
        setText(r.count, `×${a.count}`);
        const [src, dst] = alertRoute(a);
        r.route.replaceChildren(document.createTextNode(src));
        if (dst) r.route.append(h('span', 'arrow', '→'), document.createTextNode(dst));
        r.route.title = `${a.srcIp ?? ''}${a.msg ? ` · ${a.msg}` : ''}`;
        const ml = a.ruleId === 'ML';
        setText(r.rule, ml ? `ML p=${(Number(a.score) || 0).toFixed(2)}` : (a.ruleId || '—'));
        r.rule.title = ml ? 'Fused ML suspicion score' : 'Rule that fired';
        setText(r.act, a.action === 'drop' ? 'dropped inline' : '');
        r.btn.setAttribute('aria-label',
          `${a.severity} alert: ${a.cls}, ${src}${dst ? ` to ${dst}` : ''}, ${r.rule.textContent}${a.count > 1 ? `, ${a.count} times` : ''}. Select the source device.`);
      }
      setText(r.time, fmtAgo(now - (a.lastT ?? a.t)));
      if (feedEl.children[i] !== r.li) feedEl.insertBefore(r.li, feedEl.children[i] || null);
    });
    for (const [id, r] of feedRows) {
      if (!keep.has(id)) { r.li.remove(); feedRows.delete(id); }
    }
    feedPrimed = true;
    $('alert-empty').hidden = list.length > 0;
    const total = state.ids.alerts.length;
    setText($('alerts-meta'), total ? `${fmtInt(total)} row${total === 1 ? '' : 's'}${total > FEED_CAP ? `, newest ${FEED_CAP} shown` : ''}` : 'none yet');
  }

  function renderBlocked() {
    const ul = $('blocked-list');
    ul.replaceChildren();
    const blocks = [...state.ids.blocked.values()].sort((a, b) => (b.at ?? 0) - (a.at ?? 0));
    for (const b of blocks) {
      const li = h('li', 'blocked-item');
      const main = h('div', 'b-main');
      main.append(h('span', 'b-ip', b.ip), h('span', 'b-dev', nameOf(b.deviceId ?? deviceIdOfIp(b.ip), 'unknown host')));
      const reason = h('div', 'b-reason');
      reason.append(h('span', 'tag', b.auto ? 'auto' : 'manual'));
      const at = Number.isFinite(b.at) ? ` · since ${istParts(state.time.epochMs + b.at).time}` : '';
      reason.append(document.createTextNode(`${b.reason || 'Blocked'}${at}`));
      const btn = h('button', 'btn btn-sm btn-ok', 'Unblock');
      btn.type = 'button';
      btn.id = `unblock-${b.ip.replace(/\W/g, '-')}`;
      btn.setAttribute('aria-label', `Unblock ${b.ip}`);
      btn.addEventListener('click', () => bus.emit('ids:request-unblock', { ip: b.ip }));
      li.append(main, btn, reason);
      ul.append(li);
    }
    $('blocked-empty').hidden = blocks.length > 0;
    setText($('blocked-meta'), blocks.length);
  }

  // =========================================================================
  // 7. Packet log

  const logRowsEl = $('log-rows');
  const logPause = $('log-pause');
  const logFilter = $('log-filter');
  let logBuf = [];                     // newest last, at most LOG_BUFFER
  const logById = new Map();           // packet id -> entry
  let logSeq = 0;                      // bumps on any change, so render can skip idle frames
  let logRenderedSeq = -1;
  let openPktId = null;
  let autoPaused = false;
  const logPool = [];                  // reusable row refs
  const detailEl = h('div', 'log-detail');

  for (const p of LOG_PROTOS) {
    const o = h('option', null, p);
    o.value = p;
    logFilter.append(o);
  }
  {
    const o = h('option', null, 'Flagged or dropped');
    o.value = FLAGGED;
    logFilter.append(o);
  }
  logFilter.addEventListener('change', () => { logSeq++; renderSoon('log'); });
  logPause.addEventListener('change', () => {
    autoPaused = false;
    logSeq++;
    renderSoon('log');
  });

  function logVisible() {
    if (isNarrow()) return sheetTab === 'log';
    return body.dataset.drawer !== 'collapsed' && drawerTab === 'log';
  }

  function onPacketCreated(pkt) {
    if (logPause.checked || !pkt) return;
    const entry = { pkt, verdict: null, outcome: 'pending', reason: null };
    logBuf.push(entry);
    logById.set(pkt.id, entry);
    if (logBuf.length > LOG_BUFFER + 200) {
      const cut = logBuf.length - LOG_BUFFER;
      for (let i = 0; i < cut; i++) {
        const e = logBuf[i];
        if (logById.get(e.pkt.id) === e) logById.delete(e.pkt.id);
      }
      logBuf = logBuf.slice(cut);
    }
    logSeq++;
  }
  function onPacketOutcome(pkt, outcome, verdict, reason) {
    const e = pkt ? logById.get(pkt.id) : null;
    if (!e || e.pkt !== pkt) return;
    e.outcome = outcome;
    if (verdict) e.verdict = verdict;
    if (reason) e.reason = reason;
    logSeq++;
  }

  const OUTCOME = {
    pending: ['…', 'v-pending'],
    allow: ['allowed', 'v-allow'],
    alert: ['alert', 'v-alert'],
    drop: ['dropped', 'v-drop'],
    firewall: ['firewall', 'v-firewall'],
    offline: ['router off', 'v-offline'],
  };

  function makeLogRow() {
    const item = h('div', 'log-entry');
    item.setAttribute('role', 'listitem');
    const btn = h('button', 'log-row');
    btn.type = 'button';
    btn.setAttribute('aria-expanded', 'false');
    const time = h('span', 'l-time');
    const route = h('span', 'l-route');
    const src = h('span'), dst = h('span');
    route.append(src, h('span', 'arrow', '→'), dst);
    const proto = h('span', 'l-proto');
    const ports = h('span', 'l-ports');
    const len = h('span', 'l-len');
    const vcell = h('span', 'l-verdict');
    const verdict = h('span', 'verdict');
    vcell.append(verdict);
    btn.append(time, route, proto, ports, len, vcell);
    item.append(btn);
    const ref = { item, btn, time, src, dst, route, proto, ports, len, verdict, pktId: null, outcome: null };
    btn.addEventListener('click', () => toggleLogRow(ref.pktId));
    return ref;
  }

  function toggleLogRow(pktId) {
    if (pktId === null || pktId === undefined) return;
    if (openPktId === pktId) {
      openPktId = null;
      if (autoPaused) { logPause.checked = false; autoPaused = false; }
    } else {
      openPktId = pktId;
      // Keep the opened packet in view: pause the log while it is open.
      if (!logPause.checked) { logPause.checked = true; autoPaused = true; }
    }
    logSeq++;
    flag('log');
    renderLog(true);
  }

  function matchesFilter(e, f) {
    if (!f) return true;
    if (f === FLAGGED) return e.outcome !== 'allow' && e.outcome !== 'pending';
    return e.pkt.proto === f;
  }

  function fillDetail(e) {
    const p = e.pkt;
    detailEl.replaceChildren();
    const fields = [
      ['Packet', `#${p.id} · ${istParts(state.time.epochMs + (p.t ?? 0)).time}`],
      ['Source', `${nameOf(p.src, p.srcIp)} · ${p.srcIp ?? '—'} · ${p.srcMac ?? '—'}`],
      ['Destination', `${nameOf(p.dst, p.dstIp)} · ${p.dstIp ?? '—'} · ${p.dstMac ?? '—'}`],
      ['Protocol', `${p.proto}${p.l4 ? ` over ${p.l4.toUpperCase()}` : ''}`],
      ['Ports', p.sport != null || p.dport != null ? `${p.sport ?? '—'} → ${p.dport ?? '—'}` : 'n/a'],
      ['Length', `${p.len ?? '—'} bytes`],
    ];
    if (p.tcp) {
      const f = p.tcp.flags || {};
      const flags = ['SYN', 'ACK', 'FIN', 'RST', 'PSH', 'URG'].filter(k => f[k]).join(', ') || 'none';
      fields.push(['TCP', `[${flags}] seq ${p.tcp.seq ?? '—'} ack ${p.tcp.ack ?? '—'} win ${p.tcp.window ?? '—'} len ${p.tcp.len ?? 0}${p.tcp.connInit ? ' · connection start' : ''}`]);
    }
    if (p.udp) fields.push(['UDP', `length ${p.udp.len ?? '—'}`]);
    if (p.icmp) fields.push(['ICMP', `type ${p.icmp.type} code ${p.icmp.code}`]);
    if (p.arp) fields.push(['ARP', `opcode ${p.arp.opcode} · hw size ${p.arp.hwSize} · ${p.arp.senderIp} (${p.arp.senderMac}) → ${p.arp.targetIp}`]);
    if (p.mqtt) fields.push(['MQTT', `type ${p.mqtt.msgtype} · QoS ${p.mqtt.qos ?? 0}${p.mqtt.retain ? ' · retain' : ''}${p.mqtt.topic ? ` · ${p.mqtt.topic}` : ''}`]);
    if (p.http) fields.push(['HTTP', `${p.http.method ?? ''} ${p.http.uri ?? ''}${p.http.status ? ` → ${p.http.status}` : ''}`.trim()]);
    if (p.dns) fields.push(['DNS', `${p.dns.qname ?? ''} (${p.dns.qtype ?? '?'})`]);
    if (p.modbus) fields.push(['Modbus', `function ${p.modbus.fn} · unit ${p.modbus.unit} · register ${p.modbus.register}`]);
    fields.push(['Ground truth', `${p.label ?? '—'} (known only in simulation)`]);
    const left = h('div');
    left.append(h('h4', null, 'Fields'), kvList(fields.map(([k, v]) => [k, String(v), true])));

    const right = h('div');
    right.append(h('h4', null, 'Verdict'));
    const v = e.verdict;
    const reasons = h('ul', 'reasons');
    const add = t => reasons.append(h('li', null, t));
    if (e.outcome === 'firewall') {
      const blk = state.ids.blocked.get(p.srcIp);
      add(`Dropped by the router firewall before inspection: ${p.srcIp} is blocked${blk?.reason ? ` (${blk.reason})` : ''}.`);
    } else if (e.outcome === 'offline') {
      add('Lost while the router was rebooting; SHIELD-IoT never saw it.');
    } else if (v) {
      const facts = [
        ['Action', v.action],
        ['Fused score', Number.isFinite(v.score) ? v.score.toFixed(3) : '—'],
        ['ML', v.mlClass ? `${v.mlClass} p=${Number(v.mlProb ?? 0).toFixed(3)}` : 'layer off'],
        ['Inventory', v.inventory],
        ['Latency', Number.isFinite(v.latencyUs) ? `${v.latencyUs.toFixed(1)} µs` : '—'],
      ];
      if (v.invariantHits?.length) facts.push(['Invariants', v.invariantHits.map(x => x.ruleId).join(', ')]);
      right.append(kvList(facts.map(([k, val]) => [k, String(val), true])));
      for (const r of v.reasons || []) add(r);
    } else {
      add('Waiting for the verdict.');
    }
    if (reasons.children.length) right.append(reasons);
    detailEl.append(left, right);
    if (p.payload) detailEl.append(h('p', 'payload', p.payload));
  }

  function renderLog(force = false) {
    if (!force && !logVisible()) return;
    if (!force && logSeq === logRenderedSeq) return;
    logRenderedSeq = logSeq;
    const f = logFilter.value;
    const view = [];
    for (let i = logBuf.length - 1; i >= 0 && view.length < LOG_VIEW; i--) {
      if (matchesFilter(logBuf[i], f)) view.push(logBuf[i]);
    }
    while (logPool.length < view.length) {
      const r = makeLogRow();
      logPool.push(r);
      logRowsEl.append(r.item);
    }
    let openShown = false;
    logPool.forEach((r, i) => {
      const e = view[i];
      if (!e) { r.item.hidden = true; r.pktId = null; return; }
      r.item.hidden = false;
      const p = e.pkt;
      if (r.pktId !== p.id) {
        r.pktId = p.id;
        r.outcome = null;
        const ts = istParts(state.time.epochMs + (p.t ?? 0));
        setText(r.time, `${ts.time}.${ts.tenth}`);
        setText(r.src, nameOf(p.src, p.srcIp));
        setText(r.dst, nameOf(p.dst, p.dstIp));
        r.route.title = `${p.srcIp ?? ''} → ${p.dstIp ?? ''}`;
        setText(r.proto, p.proto);
        setText(r.ports, p.sport != null || p.dport != null ? `${p.sport ?? '—'}→${p.dport ?? '—'}` : '—');
        setText(r.len, p.len ?? '');
      }
      if (r.outcome !== e.outcome) {
        r.outcome = e.outcome;
        const [label, cls] = OUTCOME[e.outcome] || OUTCOME.pending;
        r.verdict.className = `verdict ${cls}`;
        setText(r.verdict, label);
        r.btn.setAttribute('aria-label', `${p.proto} packet from ${nameOf(p.src, p.srcIp)} to ${nameOf(p.dst, p.dstIp)}, ${label}`);
      }
      const open = openPktId === p.id;
      r.btn.setAttribute('aria-expanded', String(open));
      r.item.classList.toggle('is-open', open);
      if (open) {
        openShown = true;
        fillDetail(e);
        if (detailEl.parentElement !== r.item) r.item.append(detailEl);
      }
    });
    if (!openShown) {
      detailEl.remove();
      if (openPktId !== null && !logById.has(openPktId)) openPktId = null;
    }
    $('log-empty').hidden = view.length > 0;
    const meta = $('log-meta');
    meta.replaceChildren();
    meta.append(document.createTextNode(f
      ? `${view.length} of last ${fmtInt(logBuf.length)}`
      : `last ${view.length} packets`));
    if (logPause.checked) {
      const tag = h('span', 'paused', autoPaused ? ' · paused (row open)' : ' · paused');
      meta.append(tag);
    }
    meta.title = f ? `${view.length} packets match the filter among the last ${logBuf.length} captured` : '';
  }

  function clearLog() {
    logBuf = [];
    logById.clear();
    openPktId = null;
    if (autoPaused) { logPause.checked = false; autoPaused = false; }
    logSeq++;
    flag('log');
  }

  // =========================================================================
  // 8. Research and Resources tabs (static, from catalog.js)

  const researchCharts = [];

  function buildResearch() {
    const pane = $('pane-research');
    pane.replaceChildren();
    const B = RESEARCH.baseline;
    const grid = h('div', 'rs-grid');

    // Frozen baseline.
    const base = h('section', 'rs-block');
    const bh = h('h3', null, `Frozen baseline `);
    bh.append(h('span', 'sec-sub', `${B.model}, Edge-IIoTset`));
    base.append(bh);
    const metricsGrid = h('div', 'metric-grid');
    const tile = (label, value, unit) => {
      const t = h('div', 'metric');
      const b = h('b', null, value);
      if (unit) b.append(h('small', null, ` ${unit}`));
      t.append(h('span', null, label), b);
      metricsGrid.append(t);
    };
    tile('Accuracy', B.accuracy.toFixed(4));
    tile('Macro F1', B.macroF1.toFixed(4));
    tile('Weighted F1', B.weightedF1.toFixed(4));
    tile('Macro precision', B.macroPrecision.toFixed(4));
    tile('Macro recall', B.macroRecall.toFixed(4));
    tile('ROC-AUC', B.rocAuc.toFixed(4));
    tile('PR-AUC', B.prAuc.toFixed(4));
    tile('Latency/sample', B.latencyUs.toFixed(2), 'µs');
    tile('Training', B.trainSec.toFixed(2), 's');
    tile('Features', String(B.features));
    tile('Classes', String(B.classes));
    tile('Train rows', fmtInt(B.train));
    tile('Test rows', fmtInt(B.test));
    tile('Clean pool', fmtInt(B.cleanPool));
    tile('Seed', String(B.seed));
    base.append(metricsGrid);
    base.append(h('p', null, 'Frozen split with zero exact feature-vector overlap between train and test. ROC-AUC measures discrimination, not calibration.'));
    grid.append(base);

    // Per-class F1 (weakest classes, plus MITM).
    const pc = h('section', 'rs-block');
    const pch = h('h3', null, 'Per-class F1 ');
    pch.append(h('span', 'sec-sub', 'values the book reports'));
    pc.append(pch);
    const pcBox = h('div', 'rs-chart');
    pc.append(pcBox);
    const pcRows = Object.entries(RESEARCH.perClassF1)
      .map(([cls, f1]) => ({ label: cls.replace(/_/g, ' '), value: f1, raw: cls }))
      .sort((a, b) => a.value - b.value);
    researchCharts.push(createBarChart(pcBox, pcRows, {
      ref: { value: B.macroF1, label: `macro F1 ${B.macroF1.toFixed(4)}` },
      digits: 2,
      ariaLabel: `Per-class F1: ${pcRows.map(r => `${r.label} ${r.value.toFixed(2)}`).join(', ')}`,
      describe: r => `${r.raw}: F1 ${r.value.toFixed(2)}${r.raw === 'MITM' ? ' (retained support 71 only)' : ''}`,
    }));
    pc.append(h('p', null, 'DDoS_HTTP is a major confusion cluster. MITM F1 1.00 applies to a retained support of 71 only.'));
    grid.append(pc);

    // Step 4A TCP ablation.
    const ab = h('section', 'rs-block');
    const abh = h('h3', null, 'Step 4A: TCP ablation ');
    abh.append(h('span', 'sec-sub', 'macro F1 after dropping features'));
    ab.append(abh);
    const abBox = h('div', 'rs-chart');
    ab.append(abBox);
    const abRows = RESEARCH.tcpAblation.map(r => ({
      label: r.drop === 'none (baseline)' ? 'baseline (none)' : `drop ${r.drop}`, value: r.f1, emphasis: r.drop === 'none (baseline)', raw: r.drop,
    }));
    researchCharts.push(createBarChart(abBox, abRows, {
      ref: { value: B.macroF1, label: `baseline ${B.macroF1.toFixed(4)}` },
      digits: 4,
      delta: true,
      ariaLabel: `TCP ablation, macro F1: ${abRows.map(r => `${r.label} ${r.value.toFixed(4)}`).join(', ')}`,
      describe: r => `${r.label}: macro F1 ${r.value.toFixed(4)} (${r.value >= B.macroF1 ? '+' : '−'}${Math.abs(r.value - B.macroF1).toFixed(4)} vs baseline)`,
    }));
    ab.append(h('p', null, 'Sensitivity: ack > ack_raw > seq > checksum. All four are retained; checksum is weaker and noisier but not established as leakage.'));
    grid.append(ab);

    // Invariant rules.
    const inv = h('section', 'rs-block');
    const invh = h('h3', null, 'Protocol invariant rules ');
    invh.append(h('span', 'sec-sub', 'violations, train / test'));
    inv.append(invh);
    const table = h('table', 'rs-table');
    const thead = h('thead');
    const hr = h('tr');
    for (const [t, cls] of [['Rule', null], ['Condition flagged', null], ['Train', 'num'], ['Test', 'num']]) {
      const th = h('th', cls, t);
      th.scope = 'col';
      hr.append(th);
    }
    thead.append(hr);
    const tbody = h('tbody');
    for (const [id, r] of Object.entries(RESEARCH.invariants)) {
      const tr = h('tr');
      const idCell = h('td');
      idCell.append(h('code', null, id), h('span', 'rs-step', `Step ${r.step}`));
      const rule = h('td', null, r.rule);
      if (r.experimental) rule.append(h('span', 'tag-exp', 'experimental'));
      const has = Number.isFinite(r.train);
      const train = h('td', has ? 'num mono' : 'num na', has ? fmtInt(r.train) : 'n/a');
      const test = h('td', has ? 'num mono' : 'num na', has ? fmtInt(r.test) : 'n/a');
      if (!has) { train.title = 'Not in the pages of the book available to this simulator'; test.title = train.title; }
      tr.append(idCell, rule, train, test);
      tbody.append(tr);
    }
    table.append(thead, tbody);
    inv.append(table);
    const E = RESEARCH.tcpEngineTest;
    inv.append(h('p', null,
      `TCP engine on the test split: precision ${E.precision.toFixed(1)}, recall ${E.recall}, F1 ${E.f1}, specificity ${E.specificity.toFixed(1)}, FPR ${E.fpr}. ` +
      'Sparse, high-confidence asymmetric evidence, not a standalone IDS.'));
    grid.append(inv);

    // What is not in the simulator yet.
    const note = h('section', 'rs-block rs-note');
    note.append(h('h3', null, 'Not yet in this simulator'));
    note.append(h('p', null,
      'Pages 4 and 5 of the research book (the Step 5B ARP results and the Step 5C MQTT plan) were not available, so ARP violation counts are not shown ' +
      'and the MQTT invariants here are an experimental placeholder that is off by default.'));
    note.append(h('p', null, 'PIWS (protocol-invariant weak supervision) is a research hypothesis, not a proven result. The ML layer is a surrogate tuned to the baseline, not the trained model.'));
    grid.append(note);

    pane.append(grid);
  }

  function buildResources() {
    const pane = $('pane-resources');
    pane.replaceChildren();
    const grid = h('div', 'res-grid');
    for (const r of RESOURCES) {
      const a = h('a', 'res-card');
      a.href = r.url;
      a.target = '_blank';
      a.rel = 'noopener';
      const label = h('span', 'res-label', r.label);
      label.insertAdjacentHTML('beforeend', ICONS.ext);
      let shown = r.url;
      try {
        const u = new URL(r.url);
        shown = `${u.host}${u.pathname === '/' ? '' : u.pathname}`.replace(/\/$/, '');
      } catch { /* keep the raw URL */ }
      a.append(label, h('span', 'res-note', r.note || ''), h('span', 'res-url', shown));
      grid.append(a);
    }
    pane.append(grid);
  }

  // =========================================================================
  // 9. Toasts

  const toastsEl = $('toasts');
  const recentToasts = new Map();      // text -> real time, to collapse duplicates

  function showToast({ kind = 'info', text = '' } = {}) {
    if (!text) return;
    const k = ['info', 'warn', 'danger', 'ok'].includes(kind) ? kind : 'info';
    const now = performance.now();
    if (now - (recentToasts.get(text) ?? -Infinity) < 1500) return;
    recentToasts.set(text, now);
    if (recentToasts.size > 50) recentToasts.clear();

    const el = h('div', 'toast');
    el.dataset.kind = k;          // announced through the polite live region (#toasts)
    el.insertAdjacentHTML('afterbegin', ICONS[k]);
    el.append(h('span', 'toast-text', text));
    const close = h('button', 'toast-close');
    close.type = 'button';
    close.setAttribute('aria-label', 'Dismiss');
    close.innerHTML = ICONS.close;
    el.append(close);
    let timer = null;
    const dismiss = () => {
      clearTimeout(timer);
      if (el.classList.contains('is-leaving')) return;
      el.classList.add('is-leaving');
      setTimeout(() => el.remove(), reducedMotion.matches ? 0 : 220);
    };
    close.addEventListener('click', dismiss);
    timer = setTimeout(dismiss, TOAST_MS);
    toastsEl.append(el);
    while (toastsEl.children.length > TOAST_MAX) toastsEl.firstElementChild.remove();
  }

  // =========================================================================
  // 10. Keyboard

  const TYPING = new Set(['INPUT', 'SELECT', 'TEXTAREA']);
  const INTERACTIVE = 'button, a[href], summary, [role="tab"], [role="button"], [contenteditable=""], [contenteditable="true"]';

  document.addEventListener('keydown', e => {
    if (e.defaultPrevented || e.ctrlKey || e.metaKey || e.altKey) return;
    if (helpDialog.open) return;                    // the dialog handles its own keys
    const t = e.target instanceof Element ? e.target : null;
    const typing = t && (TYPING.has(t.tagName) || t.isContentEditable);
    if (e.key === 'Escape') {
      if (!resetConfirm.hidden) { closeResetConfirm(true); e.preventDefault(); return; }
      if (typing) return;
      if (state.selectedId) { selectDevice(null); e.preventDefault(); }
    } else if (e.key === ' ' || e.code === 'Space') {
      if (typing || (t && t.closest(INTERACTIVE))) return;
      e.preventDefault();
      togglePause();
    }
  });

  // =========================================================================
  // 11. Bus wiring

  bus.on('device:update', p => {
    const id = p?.id;
    if (!id) return;
    dirtyDevices.add(id);
    flag('rooms');
    if (id === state.selectedId) flag('card');
    if (id === 'rogue') flag('rogue');
  });
  bus.on('device:select', p => onSelect(p?.id ?? null));
  bus.on('device:select', p => {
    // A tap in the 3D view on a narrow screen should show the device card.
    if (p?.id && isNarrow() && sheetTab !== 'home') selectSheetTab('home');
  });
  bus.on('device:update', p => {
    // A command on the selected device answers within the same batch as the click.
    if (p?.id && p.id === state.selectedId && document.activeElement?.closest?.('#device-card, #rogue-callout')) renderSoon('card');
  });

  bus.on('ids:metrics', m => {
    metrics = m || null;
    if (m) chart.push({ t: state.time.simMs, allowed: m.ppsAllowed ?? m.pps ?? 0, dropped: m.ppsDropped ?? 0 });
    flag('metrics', 'chart');
  });
  bus.on('ids:alert', a => {
    if (!a) return;
    alertEvents++;
    const t = Number.isFinite(a.lastT) ? a.lastT : Number.isFinite(a.t) ? a.t : state.time.simMs;
    const prev = lastMarkAt.get(a.id);
    if (prev === undefined || t - prev >= 1000) {
      chart.mark(t, a.severity);
      lastMarkAt.set(a.id, t);
      if (lastMarkAt.size > 400) lastMarkAt.delete(lastMarkAt.keys().next().value);
    }
    if (a.count === 1 && sheetTab !== 'shield') unseenAlerts++;
    flag('feed', 'metrics', 'chart', 'badge');
  });
  bus.on('ids:block', p => {
    const id = p?.deviceId ?? deviceIdOfIp(p?.ip);
    if (id) dirtyDevices.add(id);
    if (id && id === state.selectedId) flag('card');
    renderSoon('blocked', 'metrics', 'rooms');
    const name = nameOf(id, p?.ip);
    showToast({
      kind: 'danger',
      text: `${name} (${p?.ip}) blocked at the router${p?.auto ? ' by SHIELD-IoT' : ''}${p?.reason ? `: ${p.reason}` : '.'}`,
    });
  });
  bus.on('ids:unblock', p => {
    const id = p?.deviceId ?? deviceIdOfIp(p?.ip);
    if (id) dirtyDevices.add(id);
    if (id && id === state.selectedId) flag('card');
    renderSoon('blocked', 'metrics', 'rooms');
    showToast({ kind: 'ok', text: `${nameOf(id, p?.ip)} (${p?.ip}) unblocked. SHIELD-IoT will not auto-block it again until reset.` });
  });
  bus.on('ids:drift', d => {
    flag('metrics');
    if (!d || !d.msg) return;
    if (d.state === 'drift') showToast({ kind: 'warn', text: d.msg });
    else if (d.state === 'warning') showToast({ kind: 'info', text: d.msg });
  });

  bus.on('packet:created', onPacketCreated);
  bus.on('packet:verdict', p => { if (p?.pkt) onPacketOutcome(p.pkt, 'pending', p.verdict, null); });
  bus.on('packet:delivered', p => {
    if (p?.pkt) onPacketOutcome(p.pkt, p.verdict?.action === 'alert' ? 'alert' : 'allow', p.verdict, null);
  });
  bus.on('packet:dropped', p => {
    if (!p?.pkt) return;
    const outcome = p.reason === 'ids' ? 'drop' : p.reason === 'firewall' ? 'firewall' : 'offline';
    onPacketOutcome(p.pkt, outcome, p.verdict, p.reason);
  });

  bus.on('toast', showToast);

  bus.on('scene:error', p => {
    const box = $('scene-error');
    box.hidden = false;
    if (p?.message) box.title = p.message;
  });

  bus.on('sim:pause', () => renderSoon('sim'));
  bus.on('sim:speed', () => renderSoon('sim'));
  bus.on('sim:after-reset', rebuildAll);

  function rebuildAll() {
    metrics = state.ids.metrics;
    alertEvents = 0;
    unseenAlerts = 0;
    lastMarkAt.clear();
    if (state.selectedId && !state.devices.has(state.selectedId)) state.selectedId = null;
    buildDeviceList();
    buildCard();
    for (const r of feedRows.values()) r.li.remove();
    feedRows.clear();
    feedPrimed = false;
    chart.clear();
    clearLog();
    resetRates();
    lastClock = '';
    flag('sim', 'metrics', 'chart', 'feed', 'blocked', 'rooms', 'rogue', 'tuning', 'badge', 'log', 'rates');
    render();
  }

  // =========================================================================
  // 12. Render and tick

  function guard(name, fn) {
    try { fn(); } catch (err) {
      if (renderErrors++ < 5) console.error(`[ui] ${name} render failed`, err);
    }
  }

  function render() {
    const has = k => dirty.has(k);
    if (has('sim')) guard('sim', renderSimControls);
    if (has('rogue')) guard('rogue', renderRogueToggle);
    if (dirtyDevices.size) {
      guard('rows', () => { for (const id of dirtyDevices) renderDeviceRow(id); });
      dirtyDevices.clear();
    }
    if (has('rooms')) guard('rooms', renderRooms);
    if (has('rates')) guard('rates', renderRates);
    if (has('card')) {
      guard('card', () => {
        if ((card?.id ?? null) !== (state.selectedId ?? null)) buildCard();
        else updateCard();
      });
    }
    if (has('card') || has('cardStats')) guard('cardStats', updateCardStats);
    if (has('metrics')) {
      guard('stats', renderStats);
      guard('layers', renderLayerHits);
      guard('drift', renderDrift);
      guard('mix', renderMix);
      guard('threat', renderThreat);
    }
    if (has('tuning')) guard('tuning', renderTuning);
    if (has('chart')) guard('chart', () => chart.render());
    if (has('feed')) guard('feed', renderFeed);
    if (has('blocked')) guard('blocked', renderBlocked);
    if (has('badge')) guard('badge', renderBadge);
    // The log is patched only while it is on screen; its flag stays set until then.
    const logPending = has('log') || logSeq !== logRenderedSeq;
    dirty.clear();
    if (logPending) {
      if (logVisible()) guard('log', () => renderLog());
      else dirty.add('log');
    }
  }

  function tick(realDtMs) {
    const dt = Number.isFinite(realDtMs) && realDtMs > 0 ? realDtMs : 16;
    renderClock();
    rateAcc += dt;
    if (rateAcc >= RATE_EVERY_MS) {
      rateAcc = 0;
      sampleRates();
      if (state.ids.alerts.length) flag('feed');      // refresh "12 s ago"
      flag('sim');
    }
    renderAcc += dt;
    if (renderAcc >= RENDER_EVERY_MS) {
      renderAcc = 0;
      render();
    }
  }

  // =========================================================================
  // Boot

  setSideCollapsed('left', prefs.left === 'collapsed');
  setSideCollapsed('right', prefs.right === 'collapsed');
  setDrawerCollapsed(prefs.drawer === 'collapsed');
  selectDrawerTab(drawerTab);
  selectSheetTab('home');
  buildResearch();
  buildResources();
  buildDeviceList();
  buildCard();
  resetRates();
  flag('sim', 'metrics', 'chart', 'feed', 'blocked', 'rooms', 'rogue', 'tuning', 'badge', 'log');
  render();
  renderClock();

  return { tick };
}
