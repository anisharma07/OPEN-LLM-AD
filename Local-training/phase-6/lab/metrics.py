"""
Experiment statistics.

Input: execution records (one per sample x variant) as written by the batch
runner, each holding `answers` (one per Score node) and `detectors` (one per
Detector Metrics node) plus merged `context` fields.
"""

import math
from collections import defaultdict
from itertools import combinations

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import cohen_kappa_score, roc_auc_score


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def ece(conf, correct, bins=10):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    if conf.size == 0:
        return None
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(e)


def answer_rows(execs):
    rows = []
    for ex in execs:
        for a in ex.get("answers", []):
            rows.append(dict(a, qid=ex["qid"], subtask=ex["subtask"], dataset=ex["dataset"], category=ex["category"],
                             is_anomalous=ex["is_anomalous"], variant=ex["variant"], ctx=ex.get("context", {}),
                             vparams=ex.get("variant_params", {})))
    return rows


def detector_rows(execs):
    rows = []
    for ex in execs:
        for d in ex.get("detectors", []):
            rows.append(dict(d, qid=ex["qid"], variant=ex["variant"], vparams=ex.get("variant_params", {})))
    return rows


def summarize_answers(rows):
    n = len(rows)
    if n == 0:
        return None
    k = sum(r["correct"] for r in rows)
    lo, hi = wilson(k, n)
    preds = [r["pred"] or "∅" for r in rows]
    gts = [r["gt"] or "?" for r in rows]
    try:
        kappa = float(cohen_kappa_score(gts, preds)) if len(set(gts) | set(preds)) > 1 else None
    except Exception:
        kappa = None
    per_st = defaultdict(list)
    per_ds = defaultdict(list)
    for r in rows:
        per_st[r["subtask"]].append(r["correct"])
        per_ds[r["dataset"]].append(r["correct"])
    st_acc = {s: {"n": len(v), "acc": float(np.mean(v))} for s, v in sorted(per_st.items())}
    out = {
        "n": n, "correct": int(k), "acc": k / n, "ci_lo": lo, "ci_hi": hi, "kappa": kappa,
        "macro_acc": float(np.mean([v["acc"] for v in st_acc.values()])),
        "parse_fail": sum(not r["parsed"] for r in rows) / n,
        "latency": _mean([r.get("latency") for r in rows]),
        "per_subtask": st_acc,
        "per_dataset": {s: {"n": len(v), "acc": float(np.mean(v))} for s, v in sorted(per_ds.items())},
    }
    conf = [(r["confidence"], r["correct"]) for r in rows if r.get("confidence") is not None]
    if conf:
        c, y = zip(*conf)
        out["ece"] = ece(c, y)
        out["mean_conf"] = float(np.mean(c))
        pg = [r.get("p_gt") for r in rows if r.get("p_gt") is not None]
        out["brier_gt"] = float(np.mean([(1 - p) ** 2 for p in pg])) if pg else None
    fired = [r for r in rows if r["ctx"].get("cue_fired") is True]
    unfired = [r for r in rows if r["ctx"].get("cue_fired") is False]
    if fired or unfired:
        out["n_fired"] = len(fired)
        out["acc_fired"] = float(np.mean([r["correct"] for r in fired])) if fired else None
        out["acc_not_fired"] = float(np.mean([r["correct"] for r in unfired])) if unfired else None
    return out


def paired(rows_a, rows_b):
    a = {r["qid"]: r["correct"] for r in rows_a}
    b = {r["qid"]: r["correct"] for r in rows_b}
    common = sorted(set(a) & set(b))
    if not common:
        return None
    fixes = sum((not a[q]) and b[q] for q in common)
    breaks = sum(a[q] and not b[q] for q in common)
    p = binomtest(fixes, fixes + breaks, 0.5).pvalue if fixes + breaks > 0 else 1.0
    st = {r["qid"]: r["subtask"] for r in rows_a}
    per = defaultdict(lambda: [0, 0, 0])
    for q in common:
        s = per[st[q]]
        s[0] += 1
        s[1] += a[q]
        s[2] += b[q]
    acc_a = float(np.mean([a[q] for q in common]))
    acc_b = float(np.mean([b[q] for q in common]))
    return {"n": len(common), "acc_a": acc_a, "acc_b": acc_b, "delta": acc_b - acc_a, "fixes": int(fixes),
            "breaks": int(breaks), "p_mcnemar": float(p),
            "per_subtask_delta": {k: (v[2] - v[1]) / v[0] for k, v in sorted(per.items())}}


def summarize_detector(rows):
    if not rows:
        return None
    by_img = {}
    for r in rows:
        by_img.setdefault(r["image_key"], r)          # one row per image (questions share images)
    imgs = list(by_img.values())
    y = [r["defective"] for r in imgs]
    # Raw detector scores live on per-category scales (e.g. PatchCore distances), so pooled
    # AUROC uses the tau-normalised score when every image has one.
    use_norm = all(r.get("score_norm") is not None for r in imgs)
    s = [r["score_norm"] if use_norm else r["score"] for r in imgs]
    out = {"n_images": len(imgs), "n_defective": int(sum(y)), "detector": imgs[0].get("detector"),
           "auroc_score": "score/tau" if use_norm else "raw score"}
    if 0 < sum(y) < len(y):
        out["image_auroc"] = float(roc_auc_score(y, s))
    per_cat = defaultdict(list)
    for r in imgs:
        per_cat[r["image_key"].split("/")[0] + "/" + r["image_key"].split("/")[1]].append((r["defective"], r["score"]))
    cat_aucs = [roc_auc_score([d for d, _ in v], [x for _, x in v]) for v in per_cat.values()
                if 0 < sum(d for d, _ in v) < len(v)]
    out["image_auroc_macro"] = float(np.mean(cat_aucs)) if cat_aucs else None
    out["n_categories_auroc"] = len(cat_aucs)
    out["pixel_auroc"] = _mean([r.get("pixel_auroc") for r in imgs])
    out["pointing"] = _mean([float(r["pointing_hit"]) for r in imgs if "pointing_hit" in r])
    out["box_iou"] = _mean([r.get("box_iou") for r in imgs])
    out["box_hit"] = _mean([float(r["box_hit"]) for r in imgs if "box_hit" in r])
    fired_def = [r["fired"] for r in imgs if "fired" in r and r["defective"]]
    fired_nor = [r["fired"] for r in imgs if "fired" in r and not r["defective"]]
    out["tpr"] = float(np.mean(fired_def)) if fired_def else None
    out["fpr"] = float(np.mean(fired_nor)) if fired_nor else None
    return out


def compute(execs, sweep_axes=None):
    sweep_axes = sweep_axes or []
    ans = answer_rows(execs)
    det = detector_rows(execs)
    variants = sorted({e["variant"] for e in execs}, key=lambda v: int(v[1:]) if v[1:].isdigit() else v)
    labels = sorted({r["label"] for r in ans})
    vparams = {e["variant"]: e.get("variant_params", {}) for e in execs}

    groups = defaultdict(list)
    for r in ans:
        groups[(r["variant"], r["label"])].append(r)

    answer = []
    for v in variants:
        for lab in labels:
            s = summarize_answers(groups.get((v, lab), []))
            if s:
                answer.append(dict(s, variant=v, label=lab, params=vparams.get(v, {})))

    pairs = []
    for v in variants:
        for la, lb in combinations(labels, 2):
            pr = paired(groups.get((v, la), []), groups.get((v, lb), []))
            if pr:
                pairs.append(dict(pr, a=f"{la}", b=f"{lb}", variant=v, kind="between arms"))
    if len(variants) > 1:
        base = variants[0]
        for lab in labels:
            for v in variants[1:]:
                pr = paired(groups.get((base, lab), []), groups.get((v, lab), []))
                if pr:
                    pairs.append(dict(pr, a=f"{lab}@{base}", b=f"{lab}@{v}", variant=v, kind="vs baseline variant"))

    dgroups = defaultdict(list)
    for r in det:
        dgroups[(r["variant"], r["label"])].append(r)
    detector = []
    for (v, lab), rows in sorted(dgroups.items()):
        s = summarize_detector(rows)
        if s:
            detector.append(dict(s, variant=v, label=lab, params=vparams.get(v, {})))

    sweep, robustness, spread = [], [], []
    for ax in sweep_axes:
        key = ax["key"]
        for lab in labels:
            pts = defaultdict(list)
            for r in ans:
                if r["label"] == lab and key in r["vparams"]:
                    pts[json_key(r["vparams"][key])].append(r["correct"])
            if not pts:
                continue
            points = [{"value": k, "acc": float(np.mean(v)), "n": len(v)} for k, v in pts.items()]
            try:
                points.sort(key=lambda p: float(p["value"]))
            except (TypeError, ValueError):
                pass
            sweep.append({"axis": key, "label": lab, "points": points})
            accs = [p["acc"] for p in points]
            spread.append({"axis": key, "label": lab, "max": max(accs), "min": min(accs),
                           "range": max(accs) - min(accs), "std": float(np.std(accs))})
            if key.endswith(".severity"):
                clean = next((p["acc"] for p in points if str(p["value"]) == "0"), None)
                if clean is not None:
                    rds = {str(p["value"]): (clean - p["acc"]) / float(p["value"])
                           for p in points if str(p["value"]) != "0"}
                    robustness.append({"axis": key, "label": lab, "clean_acc": clean, "rds": rds,
                                       "mean_rds": float(np.mean(list(rds.values()))) if rds else None})

    return {"n_executions": len(execs), "n_errors": sum(1 for e in execs if e.get("errors")),
            "labels": labels, "variants": [{"id": v, "params": vparams.get(v, {})} for v in variants],
            "answer": answer, "paired": pairs, "detector": detector, "sweep": sweep,
            "robustness": robustness, "spread": spread}


def json_key(v):
    return v if isinstance(v, (str, int, float, bool)) or v is None else str(v)
