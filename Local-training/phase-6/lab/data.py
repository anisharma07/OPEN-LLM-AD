"""
MMAD index: question flattening, robust path resolution, stratified sampling,
normal-reference lookup and ground-truth mask loading.

Differences from the Phase 1-5 loaders that matter for Arm B:
  * Normal references come from MMAD's own `similar_templates` /
    `random_templates` first, then fall back to the category's good folder
    (`train/good` or DS-MVTec's `image/good`), never including the query image.
  * GT masks are resolved per dataset (binary PNG, RGB PNG, or MVTec-LOCO's
    folder of per-defect PNGs) so detectors can be scored against them.
"""

import json
import random
import threading
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from .paths import MMAD_DIR

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".JPG", ".PNG")

SUBTASKS = [
    "Anomaly Detection",
    "Defect Classification",
    "Defect Localization",
    "Defect Description",
    "Defect Analysis",
    "Object Classification",
    "Object Structure",
    "Object Details",
    "Object Analysis",
]


def _resolve(rel: str):
    """Resolve an MMAD-relative path, tolerating the nested `X/X/...` layouts."""
    if not rel:
        return None
    rel = rel.lstrip("/")
    cands = [MMAD_DIR / rel]
    parts = rel.split("/")
    if len(parts) > 1:
        cands.append(MMAD_DIR / parts[0] / parts[0] / "/".join(parts[1:]))
    for c in cands:
        if c.exists():
            return c
    return None


class MMADIndex:
    def __init__(self):
        self._lock = threading.Lock()
        self._loaded = False
        self.images = {}        # image_key -> record
        self.questions = []     # flat list of question dicts
        self.by_qid = {}
        self._good_cache = {}

    # ------------------------------------------------------------------ load
    def ensure_loaded(self):
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            with open(MMAD_DIR / "mmad.json", "r", encoding="utf-8") as f:
                raw = json.load(f)
            for key, val in raw.items():
                img_path = _resolve(key)
                if img_path is None:
                    continue
                parts = key.split("/")
                dataset, category = parts[0], parts[1]
                image_rel = val.get("image_path", "")
                base = key[: -len(image_rel)] if image_rel and key.endswith(image_rel) else f"{dataset}/{category}/"
                mask_rel = val.get("mask_path")
                rec = {
                    "image_key": key,
                    "image_path": str(img_path),
                    "dataset": dataset,
                    "category": category,
                    "mask_rel": (base + mask_rel) if mask_rel else None,
                    "is_anomalous": bool(mask_rel),
                    "similar_templates": val.get("similar_templates", []),
                    "random_templates": val.get("random_templates", []),
                }
                self.images[key] = rec
                for i, conv in enumerate(val.get("conversation", [])):
                    q = {
                        "qid": f"{key}#{i}",
                        "image_key": key,
                        "dataset": dataset,
                        "category": category,
                        "subtask": conv.get("type", "Unknown"),
                        "question": str(conv.get("Question", "")).strip(),
                        "options": dict(conv.get("Options", {})),
                        "answer": str(conv.get("Answer", "")).strip().upper(),
                        "is_anomalous": rec["is_anomalous"],
                    }
                    self.questions.append(q)
                    self.by_qid[q["qid"]] = q
            self._loaded = True

    # --------------------------------------------------------------- queries
    def sources(self):
        self.ensure_loaded()
        return sorted({f"{r['dataset']}/{r['category']}" for r in self.images.values()})

    def filter(self, source="any", subtask="any", condition="any", datasets=None, subtasks=None):
        self.ensure_loaded()
        out = []
        for q in self.questions:
            if source not in (None, "", "any") and f"{q['dataset']}/{q['category']}" != source:
                continue
            if subtask not in (None, "", "any") and q["subtask"] != subtask:
                continue
            if datasets and q["dataset"] not in datasets:
                continue
            if subtasks and q["subtask"] not in subtasks:
                continue
            if condition == "defective" and not q["is_anomalous"]:
                continue
            if condition == "normal" and q["is_anomalous"]:
                continue
            out.append(q)
        return out

    def stratified_sample(self, n, seed=42, datasets=None, subtasks=None, condition="any", source="any"):
        """Equal draw per subtask (Phase 1/4 protocol), topped up at random."""
        pool = self.filter(source=source, condition=condition, datasets=datasets, subtasks=subtasks)
        if not pool:
            return []
        n = min(n, len(pool))
        bins = defaultdict(list)
        for q in pool:
            bins[q["subtask"]].append(q)
        rng = random.Random(seed)
        per = max(1, n // len(bins))
        picked, seen = [], set()
        for st in sorted(bins):
            for q in rng.sample(bins[st], min(per, len(bins[st]))):
                picked.append(q)
                seen.add(q["qid"])
        if len(picked) < n:
            rest = [q for q in pool if q["qid"] not in seen]
            picked.extend(rng.sample(rest, n - len(picked)))
        return picked[:n]

    def image_record(self, image_key):
        self.ensure_loaded()
        return self.images[image_key]

    # ------------------------------------------------------------ references
    def _category_goods(self, dataset, category):
        ck = (dataset, category)
        if ck not in self._good_cache:
            goods = []
            for rel in (f"{dataset}/{category}/train/good", f"{dataset}/{category}/image/good", f"{dataset}/{category}/good"):
                d = _resolve(rel)
                if d is not None and d.is_dir():
                    goods = sorted(p for p in d.iterdir() if p.suffix in IMG_EXTS)
                    if goods:
                        break
            self._good_cache[ck] = [str(p) for p in goods]
        return self._good_cache[ck]

    def normal_refs(self, image_key, k=8, strategy="similar", seed=0):
        """Up to k normal reference image paths for this query (query excluded)."""
        rec = self.image_record(image_key)
        query = rec["image_path"]
        pool = []
        if strategy in ("similar", "random"):
            tmpl = rec["similar_templates"] if strategy == "similar" else rec["random_templates"]
            pool = [str(p) for p in (_resolve(t) for t in tmpl) if p is not None]
        goods = [g for g in self._category_goods(rec["dataset"], rec["category"]) if g != query]
        if strategy == "category_random" or len(pool) < k:
            extra = [g for g in goods if g not in pool]
            random.Random(f"{image_key}|{seed}").shuffle(extra)
            pool = pool + extra
        pool = [p for p in pool if p != query]
        return pool[:k]

    def category_refs(self, source, k=8, seed=0):
        dataset, category = source.split("/", 1)
        goods = list(self._category_goods(dataset, category))
        random.Random(seed).shuffle(goods)
        return goods[:k]

    # ----------------------------------------------------------------- masks
    def gt_mask(self, image_key, size=None):
        """Boolean HxW defect mask in image coordinates (all False for normal images)."""
        rec = self.image_record(image_key)
        if size is None:
            size = Image.open(rec["image_path"]).size
        return _load_mask(rec["mask_rel"], tuple(size))


@lru_cache(maxsize=64)
def _load_mask(mask_rel, size):
    w, h = size
    mask = np.zeros((h, w), dtype=bool)
    if not mask_rel:
        return mask
    p = _resolve(mask_rel)
    if p is None:
        # MVTec-LOCO sometimes stores "<id>" dirs without extension; other datasets may miss it.
        return mask
    files = sorted(x for x in p.iterdir() if x.suffix in IMG_EXTS) if p.is_dir() else [p]
    for fp in files:
        m = np.array(Image.open(fp).convert("L").resize((w, h), Image.NEAREST)) > 0
        mask |= m
    return mask


INDEX = MMADIndex()


def load_domain_knowledge():
    p = MMAD_DIR / "domain_knowledge.json"
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


_DK_ALIASES = {"DS-MVTec": "MVTec", "MVTec-AD": "MVTec"}


def domain_knowledge_for(dataset, category):
    dk = load_domain_knowledge_cached()
    entry = dk.get(_DK_ALIASES.get(dataset, dataset), {}).get(category, {})
    if isinstance(entry, dict):
        return entry.get("combined", "") or ""
    return str(entry)


@lru_cache(maxsize=1)
def load_domain_knowledge_cached():
    return load_domain_knowledge()


def open_image(path) -> Image.Image:
    return Image.open(path).convert("RGB")


def resolve_path(rel):
    return _resolve(rel)


__all__ = ["INDEX", "SUBTASKS", "open_image", "domain_knowledge_for", "resolve_path", "Path"]
