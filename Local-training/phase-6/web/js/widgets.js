// Parameter widgets rendered inside nodes.
import { api, h, toast } from "./util.js";

export function buildWidget(editor, node, p) {
  const val = node.params[p.name];
  const set = (v, silent = false) => editor.setParam(node.id, p.name, v, { silent });
  const label = h("label", { title: p.help || p.name }, p.name.replace(/_/g, " "));
  label.addEventListener("contextmenu", (e) => {
    if (p.sweepable === false) return;
    e.preventDefault(); e.stopPropagation();
    editor.onSweepRequest?.(node.id, p.name);
  });
  const stop = (el) => { el.addEventListener("pointerdown", (e) => e.stopPropagation()); return el; };
  let ctl;

  switch (p.kind) {
    case "choice": {
      const choices = [...(p.choices || [])];
      if (val !== undefined && val !== null && !choices.includes(val)) choices.unshift(val);
      ctl = stop(h("select", { onchange: (e) => set(e.target.value) },
        choices.map((c) => h("option", { value: c, selected: c === val }, c))));
      break;
    }
    case "int":
    case "float": {
      const isInt = p.kind === "int";
      const num = stop(h("input", { type: "number", value: val, step: p.step ?? (isInt ? 1 : 0.01), min: p.min, max: p.max }));
      const parse = (v) => (isInt ? parseInt(v, 10) : parseFloat(v));
      const wrap = h("div", { class: "num" });
      const span = (p.max ?? 0) - (p.min ?? 0);
      let range = null;
      if (p.min !== undefined && p.max !== undefined && span > 0 && span <= 5000 && !p.control) {
        range = stop(h("input", { type: "range", min: p.min, max: p.max, step: p.step ?? (isInt ? 1 : span / 100), value: val }));
        range.addEventListener("input", () => { num.value = range.value; set(parse(range.value), true); });
        range.addEventListener("change", () => set(parse(range.value)));
        wrap.appendChild(range);
      }
      num.addEventListener("input", () => { const v = parse(num.value); if (!Number.isNaN(v)) { if (range) range.value = v; set(v, true); } });
      num.addEventListener("change", () => { const v = parse(num.value); if (!Number.isNaN(v)) set(v); });
      wrap.appendChild(num);
      if (p.control) {
        const mode = node.controls?.[p.name] || "fixed";
        const sel = stop(h("select", {
          class: "ctl", title: "what happens to this value after each Run",
          onchange: (e) => { node.controls[p.name] = e.target.value; editor.onChange(); },
        }, ["fixed", "increment", "decrement", "randomize"].map((m) => h("option", { value: m, selected: m === mode }, "after run: " + m))));
        wrap.appendChild(sel);
      }
      ctl = wrap;
      break;
    }
    case "bool":
      ctl = stop(h("input", { type: "checkbox", class: "toggle", checked: !!val, onchange: (e) => set(e.target.checked) }));
      break;
    case "textarea": {
      const ta = stop(h("textarea", { rows: node.type === "Note" ? 5 : 3, spellcheck: "false" }));
      ta.value = val ?? "";
      ta.addEventListener("input", () => set(ta.value, true));
      ta.addEventListener("change", () => set(ta.value));
      if (node.type === "Note") return h("div", { class: "w full" }, ta);
      return h("div", { class: "w full" }, label, ta);
    }
    case "upload": {
      const choices = [...(p.choices || [])].filter(Boolean);
      const sel = stop(h("select", { onchange: (e) => set(e.target.value) },
        h("option", { value: "" }, "— choose —"),
        choices.map((c) => h("option", { value: c, selected: c === val }, c))));
      const file = h("input", { type: "file", accept: "image/*", style: { display: "none" } });
      file.addEventListener("change", async () => {
        if (!file.files[0]) return;
        const fd = new FormData(); fd.append("file", file.files[0]);
        try {
          const r = await api("/api/upload", { method: "POST", body: fd });
          p.choices = [...(p.choices || []).filter(Boolean), r.filename];
          editor.setParam(node.id, p.name, r.filename);
          editor.refreshWidget(node.id, p.name);
        } catch (e) { toast(e.message, true); }
      });
      const btn = stop(h("button", { title: "upload an image", onclick: () => file.click() }, "⬆"));
      ctl = h("div", { class: "up" }, sel, btn, file);
      break;
    }
    default: {
      const inp = stop(h("input", { type: "text", value: val ?? "" }));
      inp.addEventListener("input", () => set(inp.value, true));
      inp.addEventListener("change", () => set(inp.value));
      ctl = inp;
    }
  }
  return h("div", { class: "w", title: p.help || "" }, label, ctl);
}
