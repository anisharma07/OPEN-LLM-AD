// Small shared helpers: fetch wrapper, DOM builder, toasts, lightbox, menus.

export async function api(path, opts = {}) {
  const init = { ...opts };
  if (opts.body && typeof opts.body !== "string" && !(opts.body instanceof FormData)) {
    init.body = JSON.stringify(opts.body);
    init.headers = { "Content-Type": "application/json", ...(opts.headers || {}) };
  }
  const r = await fetch(path, init);
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try { const j = await r.json(); msg = j.detail || msg; } catch (_) { /* not json */ }
    throw new Error(msg);
  }
  const ct = r.headers.get("content-type") || "";
  return ct.includes("application/json") ? r.json() : r.text();
}

export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
    else if (k === "html") el.innerHTML = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    el.appendChild(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

export function toast(msg, isErr = false, ms = 3500) {
  const t = h("div", { class: "toast" + (isErr ? " err" : "") }, msg);
  document.getElementById("toasts").appendChild(t);
  setTimeout(() => t.remove(), ms);
}

export function lightbox(url, caption = "") {
  const lb = h("div", { id: "lightbox", onclick: () => lb.remove() }, h("img", { src: url }), caption ? h("div", { class: "cap" }, caption) : null);
  document.body.appendChild(lb);
}

let openMenu = null;
export function closeMenus() {
  if (openMenu) { openMenu.remove(); openMenu = null; }
}
document.addEventListener("pointerdown", (e) => {
  if (openMenu && !openMenu.contains(e.target)) closeMenus();
}, true);

// items: [{label, key, onClick, header, sep, children:[...]}]
export function buildMenu(items) {
  const m = h("div", { class: "menu" });
  for (const it of items) {
    if (it.sep) { m.appendChild(h("hr")); continue; }
    if (it.header) { m.appendChild(h("div", { class: "mh" }, it.header)); continue; }
    if (it.children) {
      const sub = h("div", { class: "mi submenu" }, h("span", {}, it.label), h("span", { class: "k" }, "▸"));
      sub.appendChild(buildMenu(it.children));
      m.appendChild(sub);
      continue;
    }
    m.appendChild(h("div", {
      class: "mi", onclick: (e) => { e.stopPropagation(); closeMenus(); it.onClick && it.onClick(); },
    }, h("span", {}, it.label), it.key ? h("span", { class: "k" }, it.key) : null));
  }
  return m;
}

export function showMenuAt(x, y, items) {
  closeMenus();
  const wrap = h("div", { id: "ctx" }, buildMenu(items));
  document.body.appendChild(wrap);
  const r = wrap.getBoundingClientRect();
  wrap.style.left = Math.min(x, innerWidth - r.width - 8) + "px";
  wrap.style.top = Math.min(y, innerHeight - r.height - 8) + "px";
  openMenu = wrap;
}

export function showMenuBelow(btn, items) {
  const r = btn.getBoundingClientRect();
  showMenuAt(r.left, r.bottom + 4, items);
}

export function fmtPct(x, d = 1) { return x === null || x === undefined ? "–" : (x * 100).toFixed(d) + "%"; }
export function fmtNum(x, d = 3) { return x === null || x === undefined || Number.isNaN(x) ? "–" : Number(x).toFixed(d); }
export function fmtTime(s) {
  if (s === null || s === undefined) return "–";
  if (s < 1) return (s * 1000).toFixed(0) + " ms";
  if (s < 90) return s.toFixed(1) + " s";
  if (s < 5400) return (s / 60).toFixed(1) + " min";
  return (s / 3600).toFixed(1) + " h";
}
export const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
