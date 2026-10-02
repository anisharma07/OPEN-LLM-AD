// The home network: one Wi-Fi router with the SHIELD-IoT sensor inline.
//
// network.js owns
//   - routing: every packet goes router firewall → IDS → delivery (CONTRACT.md),
//     a sim-time event queue and a hard rate cap,
//   - device commands (device:command) and their effects on device props,
//   - device behaviour over time (washer cycle, vacuum battery, room temperature…),
//   - device online state and status, and sim:reset.
// What packets look like and when everyday traffic happens lives in traffic.js.

import { resetState } from './state.js';
import { createRng, createTraffic, SEED } from './traffic.js';

const MAX_PPS = 600;                // hard cap, packets per second of sim time
const BURST = 300;                  // token-bucket depth for the cap
const REBOOT_MS = 2000;             // router offline time
const ALERT_HOLD_MS = 8000;         // 'alert' status lasts this long after the last alert
const UPDATE_EVERY_MS = 500;        // behaviour-driven device:update at most ~2 Hz per device (real time)
const SAMPLE_EVERY_MS = 1000;       // sensor sampling period (sim time)
const LISTEN_MS = 2000;
const RING_MS = 4000;
const FRIDGE_DOOR_WARN_MS = 60000;
const AMBIENT_C = 28;               // a warm Indian evening, without the AC
const VACUUM_LOW = 15;
const BASE_LOAD_KW = 1.21;          // untracked household load behind the meter

// Programme durations; one sim second counts as one programme minute.
const WASHER_MINUTES = { 'Cotton 40°': 90, 'Quick 15': 15, Wool: 45, 'Eco 20°': 120 };

// Controls that are physical actions in the house rather than network commands:
// they work even on a blocked device and need neither the phone nor the router.
const LOCAL = new Set(['phone:wifi', 'laptop:wifi', 'rogue:plugged', 'fridge:doorOpen', 'speaker:say']);

const passVerdict = reason => ({
  action: 'allow', score: 0, mlClass: null, mlProb: null, invariantHits: [], inventory: 'known',
  drift: false, reasons: reason ? [reason] : [], latencyUs: 0,
});

/** Binary min-heap of {t, fn}; ties keep insertion order. */
function createQueue() {
  let heap = [];
  let seq = 0;
  const less = (a, b) => a.t < b.t || (a.t === b.t && a.s < b.s);
  const swap = (i, j) => { const x = heap[i]; heap[i] = heap[j]; heap[j] = x; };
  return {
    push(t, fn) {
      heap.push({ t, s: seq++, fn });
      let i = heap.length - 1;
      while (i > 0) {
        const p = (i - 1) >> 1;
        if (!less(heap[i], heap[p])) break;
        swap(i, p);
        i = p;
      }
    },
    pop() {
      const top = heap[0];
      const last = heap.pop();
      if (heap.length) {
        heap[0] = last;
        let i = 0;
        for (;;) {
          const l = 2 * i + 1, r = l + 1;
          let m = i;
          if (l < heap.length && less(heap[l], heap[m])) m = l;
          if (r < heap.length && less(heap[r], heap[m])) m = r;
          if (m === i) break;
          swap(i, m);
          i = m;
        }
      }
      return top;
    },
    peekT: () => (heap.length ? heap[0].t : Infinity),
    clear() { heap = []; },
    get size() { return heap.length; },
  };
}

export function createNetwork({ bus, state, ids }) {
  const queue = createQueue();
  const lastAlertAt = new Map();      // device id → sim time of its latest alert
  const dirty = new Set();            // devices with throttled behaviour updates pending
  const lastEmit = new Map();         // device id → realMs of its last device:update
  const sliders = new Map();          // 'id:key' → pending coalescible command
  let rng, life, traffic;             // traffic PRNG, behaviour PRNG, traffic engine
  let hidden;                         // behaviour accumulators (fractional values behind integer props)
  let tokens = BURST;
  let suppressed = 0;
  let routerDownUntil = null;         // sim time the router comes back, null while it is up
  let realMs = 0;                     // real time elapsed while running (simDt / speed)
  let inspectErrors = 0;

  const get = id => state.devices.get(id);

  // ---- power, blocks, status ----------------------------------------------------

  /** Power / link only: switched on, Wi-Fi on, plugged in. Ignores the router and blocks. */
  function powered(id) {
    const d = get(id);
    if (!d) return false;
    if (d.role === 'cloud') return true;
    if (d.role === 'unknown') return !!d.props.plugged;
    if (d.props.power === false || d.props.wifi === false) return false;
    return true;
  }

  function firewallBlocks(ip) {
    try { return !!ids.isBlocked(ip); } catch { return state.ids.blocked.has(ip); }
  }

  const isBlocked = d => !!d && (d.blocked || state.ids.blocked.has(d.ip));
  const onLan = d => d.role !== 'cloud' && d.id !== 'shield';   // the sensor is an appliance; it stays up

  function touch(id) {
    lastEmit.set(id, realMs);
    dirty.delete(id);
    bus.emit('device:update', { id });
  }
  const markDirty = id => dirty.add(id);

  function flushDirty() {
    for (const id of dirty) {
      if (realMs - (lastEmit.get(id) ?? -Infinity) >= UPDATE_EVERY_MS) touch(id);
    }
  }

  const toast = (kind, text) => bus.emit('toast', { kind, text });

  function countClients(emit) {
    const r = get('router');
    if (!r) return;
    let n = 0;
    for (const d of state.devices.values()) if (onLan(d) && d.id !== 'router' && d.online) n++;
    if (r.props.clients !== n) {
      r.props.clients = n;
      if (emit) touch('router');
    }
  }

  /**
   * Recomputes online + status (precedence blocked > offline > alert > ok).
   * With emit, sends device:update on a change and returns true when it did.
   */
  function refresh(id, emit = true) {
    const d = get(id);
    if (!d) return false;
    const online = powered(id) && !(routerDownUntil !== null && onLan(d));
    const onlineChanged = online !== d.online;
    d.online = online;
    let status = 'ok';
    if (isBlocked(d)) status = 'blocked';
    else if (!online) status = 'offline';
    else if (lastAlertAt.has(id) && state.time.simMs - lastAlertAt.get(id) < ALERT_HOLD_MS) status = 'alert';
    const statusChanged = status !== d.status;
    d.status = status;
    if (onlineChanged) countClients(emit);
    if (emit && (onlineChanged || statusChanged)) { touch(id); return true; }
    return false;
  }
  /** After a prop change: always one device:update, plus any status change. */
  const settle = id => { if (!refresh(id)) touch(id); };

  function refreshAll(emit = true) {
    for (const id of state.devices.keys()) refresh(id, emit);
  }

  // ---- routing ------------------------------------------------------------------

  function inspect(pkt) {
    try {
      return ids.inspect(pkt) || passVerdict('no verdict');
    } catch (err) {
      if (inspectErrors++ < 3) console.error('[network] ids.inspect failed; packet allowed', err);
      return passVerdict('inspection error');
    }
  }

  function drop(pkt, src, reason, verdict) {
    if (src) src.stats.dropped++;
    bus.emit('packet:dropped', verdict ? { pkt, reason, verdict } : { pkt, reason });
    return false;
  }

  /** CONTRACT.md pipeline. Returns true when the packet reached its destination. */
  function send(pkt) {
    if (tokens < 1) { suppressed++; return false; }   // over the rate cap: the packet is never sent
    tokens -= 1;
    const src = get(pkt.src);
    const dst = get(pkt.dst);
    if (src) src.stats.tx++;
    bus.emit('packet:created', pkt);
    if (routerDownUntil !== null) return drop(pkt, src, 'router-offline');
    if (firewallBlocks(pkt.srcIp)) return drop(pkt, src, 'firewall');
    const verdict = inspect(pkt);
    bus.emit('packet:verdict', { pkt, verdict });
    if (verdict.action === 'drop') return drop(pkt, src, 'ids', verdict);
    if (dst) dst.stats.rx++;
    bus.emit('packet:delivered', { pkt, verdict });
    return true;
  }

  const net = {
    at: (t, fn) => queue.push(t, fn),
    send,
    budget: n => tokens >= n,
    powered,
    blocked: id => isBlocked(get(id)),
  };

  function drain(until) {
    let n = 0;
    while (queue.peekT() <= until && n++ < 20000) queue.pop().fn();
  }

  // ---- router reboot ------------------------------------------------------------

  function routerDown(at) {
    if (routerDownUntil !== null) return;
    routerDownUntil = at + REBOOT_MS;
    traffic.dropAll();
    refreshAll();
    toast('warn', `${get('router')?.name ?? 'Router'} is rebooting: the home network is offline for about 2 s.`);
  }

  function routerBack(at) {
    routerDownUntil = null;
    refreshAll();
    toast('ok', `${get('router')?.name ?? 'Router'} is back online; devices are reconnecting.`);
    traffic.rejoinAll(at);
  }

  // ---- device commands --------------------------------------------------------------

  /** Validates a control value; undefined when it is not acceptable. */
  function coerce(control, value, current) {
    switch (control.type) {
      case 'toggle':
        if (value === undefined || value === null) return !current;
        if (typeof value === 'string') return ['true', 'on', '1', 'yes'].includes(value.toLowerCase());
        return Boolean(value);
      case 'range': {
        const n = Number(value);
        if (value === null || value === '' || !Number.isFinite(n)) return undefined;
        const min = Number.isFinite(control.min) ? control.min : -Infinity;
        const max = Number.isFinite(control.max) ? control.max : Infinity;
        const step = control.step > 0 ? control.step : 1;
        const base = Number.isFinite(min) ? min : 0;
        const snapped = base + Math.round((Math.min(max, Math.max(min, n)) - base) / step) * step;
        return Number(snapped.toFixed(6));
      }
      case 'select':
        return (control.options || []).find(o => String(o) === String(value));
      case 'color':
        return typeof value === 'string' && /^#[0-9a-f]{6}$/i.test(value) ? value.toLowerCase() : undefined;
      default:
        return undefined;
    }
  }

  function resetWasher(d) {
    const minutes = WASHER_MINUTES[d.props.program] ?? 30;
    hidden.washer = minutes;
    d.props.remainingMin = minutes;
  }

  /** Sets one prop and its direct consequences; false when nothing changed. */
  function apply(d, key, value) {
    if (Object.is(d.props[key], value)) return false;
    d.props[key] = value;
    switch (`${d.type}:${key}`) {
      case 'washer:running':
        if (value && !(hidden.washer > 0)) resetWasher(d);
        break;
      case 'washer:program':
        resetWasher(d);   // a new programme starts from its full duration
        break;
      case 'speaker:power':
        if (!value) { d.props.music = false; d.props.listening = false; }
        break;
      case 'plug:power':
        hidden.plugOnMs = 0;
        d.props.watts = value ? 1000 : 0;
        break;
      case 'fridge:doorOpen':
        hidden.doorOpenSince = value ? state.time.simMs : null;
        hidden.doorWarned = false;
        break;
      default:
    }
    return true;
  }

  /**
   * Sends the control packets for a changed device. Handles power transitions
   * (a device switched off says goodbye; one switched on joins and reconnects)
   * and folds rapid slider/colour moves into one exchange.
   */
  function sendControl(d, key, origin, wasUp, sliderValue) {
    const now = state.time.simMs;
    const isUp = powered(d.id);
    const retire = wasUp && !isUp ? traffic.detach(d.id) : undefined;
    if (!wasUp && isUp) traffic.deviceUp(now, d.id);
    const deliver = d.type === 'router' ? 'https' : d.type === 'meter' ? 'modbus' : wasUp || isUp ? 'mqtt' : 'none';
    const k = `${d.id}:${key}`;
    if (sliderValue !== undefined) {
      const pending = sliders.get(k);
      if (pending && !pending.ctl.started && !pending.ctl.flow.dead) { pending.box.value = sliderValue; return; }
    }
    const box = { value: d.props[key] };
    const ctl = traffic.control(now, { origin, target: d.id, body: () => ({ [key]: box.value }), deliver, retire });
    if (sliderValue !== undefined) sliders.set(k, { ctl, box });
  }

  function onCommand(cmd) {
    const { id, key, value } = cmd || {};
    const d = get(id);
    const control = d?.controls?.find(c => c.key === key);
    if (!d || !control) {
      console.warn(`[network] ignoring command for unknown control ${id}.${key}`);
      return;
    }
    const local = LOCAL.has(`${d.type}:${key}`);
    if (!local && isBlocked(d)) {
      toast('warn', `${d.name} is blocked by SHIELD-IoT, so it ignores commands.`);
      touch(id);
      return;
    }
    let next;
    if (control.type !== 'button') {
      next = coerce(control, value, d.props[key]);
      if (next === undefined) {
        console.warn(`[network] invalid value for ${id}.${key}:`, value);
        touch(id);
        return;
      }
    }
    if (local) { localAction(d, key, next); return; }

    // Network commands start on the phone app (or the vendor cloud when the
    // phone is off Wi-Fi), on Alexa for voice, on the watch for "find my phone".
    const origin = key === 'speak' ? 'speaker' : key === 'findPhone' ? 'watch' : powered('phone') ? 'phone' : 'cloud';
    const now = state.time.simMs;
    if (routerDownUntil !== null) {
      toast('warn', key === 'reboot' ? 'The router is already rebooting.' : `The router is rebooting, so the command did not reach ${d.name}.`);
      if (origin !== 'speaker' || powered('speaker')) {
        traffic.control(now, { origin, target: id, body: () => ({ [key]: next ?? true }), deliver: 'none' });
      }
      touch(id);
      return;
    }
    const relay = [get(origin), get('hub')].find(x => x && x.id !== id && isBlocked(x));
    if (relay) {
      toast('warn', `Command not delivered: ${relay.name} is blocked by SHIELD-IoT.`);
      touch(id);
      return;
    }

    switch (key) {
      case 'reboot':
        traffic.reboot(now, origin, routerDown);   // goes down once the router has the request
        return;
      case 'speak':
        speak(d, value);
        return;
      case 'dock':
        if (!d.props.running) { toast('info', `${d.name} is already on its dock.`); touch(id); return; }
        apply(d, 'running', false);
        traffic.control(now, { origin, target: id, body: () => ({ dock: true }), deliver: 'mqtt' });
        toast('info', `${d.name} is returning to its dock.`);
        touch(id);
        return;
      case 'sync': {
        const offset = life.int(-40, 40);
        traffic.clockSync(now, origin);
        toast('ok', `${d.name} synced over NTP (offset ${offset >= 0 ? '+' : ''}${offset} ms).`);
        return;
      }
      case 'read':
        traffic.meterRead(now, origin);
        toast('info', `${d.name} reads ${Number(d.props.kw).toFixed(2)} kW (Modbus/TCP, holding register 0x000C).`);
        return;
      case 'findPhone':
        findPhone(d);
        return;
      default:
        setControl(d, control, key, next, origin);
    }
  }

  /** toggle / range / select / color controls. */
  function setControl(d, control, key, next, origin) {
    const id = d.id;
    if (d.type === 'vacuum' && key === 'running' && next && d.props.battery <= VACUUM_LOW) {
      toast('warn', `${d.name} battery is too low (${d.props.battery}%); it stays on the dock to charge.`);
      touch(id);
      return;
    }
    const wasUp = powered(id);
    if (!apply(d, key, next)) { touch(id); return; }
    const slider = control.type === 'range' || control.type === 'color';
    sendControl(d, key, origin, wasUp, slider ? next : undefined);

    switch (`${d.type}:${key}`) {
      case 'tv:channel':
        if (powered(id)) traffic.tvChannel(state.time.simMs + 120);
        break;
      case 'lock:locked':
        toast(next ? 'ok' : 'warn', `${d.name}: ${next ? 'locked' : 'unlocked'}.`);
        break;
      case 'router:guestWifi':
        toast('info', `Guest Wi-Fi turned ${next ? 'on' : 'off'}.`);
        break;
      case 'washer:running':
        if (next) toast('ok', `${d.name} started: ${d.props.program} (${d.props.remainingMin} min).`);
        else toast('info', `${d.name} paused with ${d.props.remainingMin} min left.`);
        break;
      case 'vacuum:running':
        toast('info', next ? `${d.name} started cleaning.` : `${d.name} stopped; returning to its dock.`);
        break;
      default:
    }
    settle(id);
  }

  /** Physical actions: no app command, no router needed, allowed while blocked. */
  function localAction(d, key, next) {
    const now = state.time.simMs;
    const wasUp = powered(d.id);
    if (!apply(d, key, next)) { touch(d.id); return; }
    const isUp = powered(d.id);
    if (wasUp && !isUp) traffic.dropDevice(d.id);
    if (!wasUp && isUp) traffic.deviceUp(now, d.id);
    if (d.type === 'rogue') {
      const blocked = isBlocked(d) ? ' It is still blocked by SHIELD-IoT.' : '';
      toast(next ? 'warn' : 'info', next
        ? `Unknown device joined the network (${d.ip}, MAC ${d.mac}).${blocked}`
        : 'The unknown device was unplugged from the network.');
    }
    if (d.type === 'fridge') traffic.statePublish(now, d.id);
    settle(d.id);
  }

  /** What each Alexa phrase does: per target device, the props to set. */
  function voiceIntent(phrase) {
    const all = [...state.devices.values()];
    const ofType = type => all.filter(x => x.type === type);
    switch (phrase) {
      case 'Alexa, lights on': return { targets: ofType('light').map(l => [l, { power: true }]), text: 'Alexa: lights on' };
      case 'Alexa, lights off': return { targets: ofType('light').map(l => [l, { power: false }]), text: 'Alexa: lights off' };
      case 'Alexa, fan speed 3': return { targets: ofType('fan').map(f => [f, { power: true, speed: 3 }]), text: 'Alexa: fan set to speed 3' };
      case 'Alexa, play music': return { targets: [], music: true, text: 'Alexa: playing music' };
      case 'Alexa, lock the door': return { targets: ofType('lock').map(l => [l, { locked: true }]), text: 'Alexa: front door locked' };
      case 'Alexa, start the washer': {
        const w = ofType('washer')[0];
        if (w?.props.running) return { targets: [], text: `Alexa: the washer is already running (${w.props.remainingMin} min left)` };
        return { targets: w ? [[w, { running: true }]] : [], washer: w, text: 'Alexa: washer started' };
      }
      default: return { targets: [], text: `Alexa: sorry, I can't help with "${phrase}"` };
    }
  }

  function speak(sp, value) {
    const now = state.time.simMs;
    if (!powered(sp.id)) { toast('warn', `${sp.name} is switched off.`); touch(sp.id); return; }
    const options = sp.controls.find(c => c.key === 'say')?.options || [];
    const phrase = typeof value === 'string' && options.includes(value) ? value : sp.props.say;
    sp.props.say = phrase;
    sp.props.listening = true;
    hidden.listenUntil = now + LISTEN_MS;

    const intent = voiceIntent(phrase);
    const actions = [];
    const refused = [];
    for (const [target, changes] of intent.targets) {
      if (isBlocked(target)) { refused.push(target.name); continue; }
      const wasUp = powered(target.id);
      let changed = false;
      for (const [k, v] of Object.entries(changes)) changed = apply(target, k, v) || changed;
      const isUp = powered(target.id);
      const retire = wasUp && !isUp ? traffic.detach(target.id) : undefined;
      if (!wasUp && isUp) traffic.deviceUp(now, target.id);
      actions.push({ target: target.id, body: () => changes, deliver: wasUp || isUp ? 'mqtt' : 'none', retire });
      if (changed) settle(target.id);
    }
    if (intent.music) sp.props.music = true;
    traffic.voice(now, actions);

    let text = intent.text;
    if (intent.washer && actions.length) text = `Alexa: washer started (${intent.washer.props.program}, ${intent.washer.props.remainingMin} min)`;
    if (refused.length) toast('warn', `${text}, but ${refused.join(', ')} ${refused.length > 1 ? 'are' : 'is'} blocked by SHIELD-IoT.`);
    else toast(intent.targets.length || intent.music ? 'ok' : 'info', `${text}.`);
    touch(sp.id);
  }

  function findPhone(watch) {
    const phone = get('phone');
    if (!phone || !powered('phone')) {
      toast('warn', `${phone?.name ?? 'The phone'} is not on Wi-Fi, so ${watch.name} cannot ring it.`);
      return;
    }
    if (isBlocked(phone)) {
      toast('warn', `${phone.name} is blocked by SHIELD-IoT, so it cannot ring.`);
      return;
    }
    phone.props.ringing = true;
    hidden.ringUntil = state.time.simMs + RING_MS;
    traffic.ringPhone(state.time.simMs);
    toast('info', `${phone.name} is ringing.`);
    touch('phone');
  }

  // ---- behaviour over time ------------------------------------------------------------

  function homeLoadKw() {
    const p = id => get(id)?.props || {};
    let kw = BASE_LOAD_KW;
    if (p('tv').power) kw += 0.11;
    const ac = p('ac');
    if (ac.power) kw += ac.mode === 'Cool' ? 1.35 : ac.mode === 'Dry' ? 0.85 : 0.07;
    for (const d of state.devices.values()) {
      if (d.type === 'light' && d.props.power) kw += 0.009 * (d.props.brightness ?? 100) / 100;
    }
    if (p('fan').power) kw += 0.012 * Number(p('fan').speed || 0);
    kw += (p('plug').watts || 0) / 1000;
    if (p('washer').running) kw += 0.5;
    if (!p('vacuum').running && p('vacuum').battery < 100) kw += 0.03;
    if (p('laptop').wifi) kw += 0.045;
    if (p('camera').power) kw += 0.006;
    return kw;
  }

  /** Sensors that change in steps: sampled once a second of sim time. */
  function sample(now) {
    const plug = get('plug');
    if (plug) {
      const heating = hidden.plugOnMs < 45000;   // brewing, then keep-warm
      const watts = plug.props.power ? (heating ? 1000 + life.int(-25, 25) : 70 + life.int(-8, 8)) : 0;
      if (watts !== plug.props.watts) { plug.props.watts = watts; markDirty('plug'); }
    }
    const watch = get('watch');
    if (watch) {
      let hr = watch.props.heartRate + life.int(-2, 2);
      if (hr > 80) hr -= 1;
      if (hr < 64) hr += 1;
      hr = Math.max(58, Math.min(92, hr));
      if (hr !== watch.props.heartRate) { watch.props.heartRate = hr; markDirty('watch'); }
      if (life.chance(0.2)) { watch.props.steps += life.int(1, 4); markDirty('watch'); }
    }
    const meter = get('meter');
    if (meter) {
      const kw = Math.round((homeLoadKw() + life.range(-0.015, 0.015)) * 100) / 100;
      if (kw !== meter.props.kw) { meter.props.kw = kw; markDirty('meter'); }
    }
    const fridge = get('fridge');
    if (fridge?.props.doorOpen && hidden.doorOpenSince !== null && !hidden.doorWarned && now - hidden.doorOpenSince >= FRIDGE_DOOR_WARN_MS) {
      hidden.doorWarned = true;
      toast('warn', `${fridge.name} door has been open for a minute.`);
    }
  }

  function behave(dt, now) {
    const sec = dt / 1000;

    const washer = get('washer');
    if (washer?.props.running) {
      hidden.washer = Math.max(0, hidden.washer - sec);
      const shown = Math.ceil(hidden.washer);
      if (hidden.washer <= 0) {
        washer.props.running = false;
        washer.props.remainingMin = 0;
        touch(washer.id);
        toast('ok', `${washer.name} finished: ${washer.props.program}.`);
        traffic.statePublish(now, washer.id);
      } else if (shown !== washer.props.remainingMin) {
        washer.props.remainingMin = shown;
        markDirty(washer.id);
      }
    }

    const vac = get('vacuum');
    if (vac) {
      if (vac.props.running) {
        hidden.battery -= sec / 5;   // 1 % every 5 s while cleaning
        if (hidden.battery <= VACUUM_LOW) {
          hidden.battery = VACUUM_LOW;
          vac.props.running = false;
          vac.props.battery = VACUUM_LOW;
          touch(vac.id);
          toast('warn', `${vac.name} battery low (${VACUUM_LOW}%): returning to its dock.`);
          traffic.statePublish(now, vac.id);
        }
      } else if (hidden.battery < 100) {
        hidden.battery = Math.min(100, hidden.battery + sec / 2.5);   // 1 % every 2.5 s on the dock
      }
      const shown = Math.round(hidden.battery);
      if (shown !== vac.props.battery) { vac.props.battery = shown; markDirty(vac.id); }
    }

    const th = get('thermostat');
    if (th) {
      const ac = get('ac');
      const cooling = !!ac?.props.power && ac.props.mode !== 'Fan';
      const target = cooling ? Number(th.props.target) : AMBIENT_C;
      let rate = 0.008;   // °C per sim second, natural drift
      if (cooling && hidden.current > target) rate = ac.props.mode === 'Dry' ? 0.03 : 0.06;
      const diff = target - hidden.current;
      hidden.current += Math.sign(diff) * Math.min(Math.abs(diff), rate * sec);
      const shown = Math.round(hidden.current * 10) / 10;
      if (shown !== th.props.current) { th.props.current = shown; markDirty(th.id); }
    }

    const plug = get('plug');
    if (plug?.props.power) hidden.plugOnMs += dt;

    const sp = get('speaker');
    if (sp?.props.listening && hidden.listenUntil !== null && now >= hidden.listenUntil) {
      sp.props.listening = false;
      hidden.listenUntil = null;
      touch(sp.id);
    }
    const phone = get('phone');
    if (phone?.props.ringing && hidden.ringUntil !== null && now >= hidden.ringUntil) {
      phone.props.ringing = false;
      hidden.ringUntil = null;
      touch(phone.id);
    }

    hidden.sampleMs += dt;
    if (hidden.sampleMs >= SAMPLE_EVERY_MS) {
      hidden.sampleMs %= SAMPLE_EVERY_MS;
      sample(now);
    }
  }

  function decayAlerts(now) {
    for (const [id, t] of lastAlertAt) {
      if (now - t >= ALERT_HOLD_MS) { lastAlertAt.delete(id); refresh(id); }
    }
  }

  // ---- lifecycle --------------------------------------------------------------------

  function boot() {
    rng = createRng(SEED);
    life = createRng(SEED ^ 0x9e3779b9);
    const p = id => get(id)?.props || {};
    hidden = {
      washer: Number(p('washer').running ? p('washer').remainingMin : 0) || 0,
      battery: Number(p('vacuum').battery ?? 100),
      current: Number(p('thermostat').current ?? AMBIENT_C),
      plugOnMs: p('plug').power ? 60000 : 0,
      listenUntil: null,
      ringUntil: null,
      doorOpenSince: p('fridge').doorOpen ? 0 : null,
      doorWarned: false,
      sampleMs: 0,
    };
    if (get('phone')) get('phone').props.ringing = false;
    refreshAll(false);
    countClients(false);
    traffic = createTraffic({ state, rng, net });
  }

  function reset() {
    resetState(state);
    try { ids.reset(); } catch (err) { console.error('[network] ids.reset failed', err); }
    queue.clear();
    sliders.clear();
    lastAlertAt.clear();
    dirty.clear();
    lastEmit.clear();
    tokens = BURST;
    suppressed = 0;
    routerDownUntil = null;
    realMs = 0;
    boot();
    bus.emit('sim:after-reset', {});
  }

  function tick(simDt) {
    if (!(simDt > 0)) return;
    const now = state.time.simMs;
    realMs += simDt / Math.max(0.25, Number(state.time.speed) || 1);
    tokens = Math.min(BURST, tokens + (simDt * MAX_PPS) / 1000);
    behave(simDt, now);
    traffic.tick(now);
    if (routerDownUntil !== null && routerDownUntil <= now) {
      const back = routerDownUntil;
      drain(back);
      routerBack(back);
    }
    drain(now);
    decayAlerts(now);
    flushDirty();
  }

  // ---- wiring -----------------------------------------------------------------------

  function onBlockChange(p) {
    const id = p?.deviceId || state.byIp.get(p?.ip);
    if (!id) return;
    refresh(id);
    queueMicrotask(() => refresh(id));   // in case the block map is updated after the event
  }

  bus.on('device:command', onCommand);
  bus.on('ids:alert', a => {
    const id = a?.srcId || state.byIp.get(a?.srcIp);
    if (!id || !get(id)) return;
    lastAlertAt.set(id, state.time.simMs);
    refresh(id);
  });
  bus.on('ids:block', onBlockChange);
  bus.on('ids:unblock', onBlockChange);
  bus.on('sim:reset', reset);

  boot();

  return {
    tick,
    /** Extra, read-only: counters for debugging and tests. */
    info: () => ({
      queued: queue.size, tokens, suppressed, routerDown: routerDownUntil !== null, ...traffic.info(),
    }),
  };
}
