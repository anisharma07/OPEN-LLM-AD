"""Small image helpers shared by nodes (overlays, contact sheets, region words)."""

import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageDraw


def to_map_res(amap, res=256):
    """Resample an anomaly map to res x res (the shared frame of all nodes)."""
    if amap.shape == (res, res):
        return amap
    return np.asarray(Image.fromarray(amap.astype(np.float32), mode="F").resize((res, res), Image.BILINEAR))


def normalize_map(amap, lo=None, hi=None):
    lo = float(np.min(amap)) if lo is None else lo
    hi = float(np.max(amap)) if hi is None else hi
    return np.clip((amap - lo) / (hi - lo + 1e-8), 0, 1)


def heatmap_image(amap, size, cmap="jet", lo=None, hi=None):
    rgba = colormaps[cmap](normalize_map(amap, lo, hi))
    im = Image.fromarray((rgba[..., :3] * 255).astype(np.uint8))
    return im.resize(size, Image.BILINEAR)


def overlay_heatmap(img, amap, alpha=0.45, cmap="jet", lo=None, hi=None):
    base = img.convert("RGB")
    heat = heatmap_image(amap, base.size, cmap, lo, hi)
    return Image.blend(base, heat, alpha)


def draw_boxes(img, boxes, color=(255, 30, 30), width=None, dashed=False):
    im = img.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    w = width or max(3, int(min(im.size) / 150))
    for (x0, y0, x1, y1) in boxes:
        if dashed:
            seg = max(6, w * 3)
            for x in range(int(x0), int(x1), seg * 2):
                d.line([(x, y0), (min(x + seg, x1), y0)], fill=color, width=w)
                d.line([(x, y1), (min(x + seg, x1), y1)], fill=color, width=w)
            for y in range(int(y0), int(y1), seg * 2):
                d.line([(x0, y), (x0, min(y + seg, y1))], fill=color, width=w)
                d.line([(x1, y), (x1, min(y + seg, y1))], fill=color, width=w)
        else:
            d.rectangle([x0, y0, x1, y1], outline=color, width=w)
    return im


def mask_overlay(img, mask, color=(0, 255, 120), alpha=0.45):
    base = img.convert("RGB")
    m = Image.fromarray((mask.astype(np.uint8) * 255)).resize(base.size, Image.NEAREST)
    col = Image.new("RGB", base.size, color)
    blended = Image.blend(base, col, alpha)
    return Image.composite(blended, base, m)


def contact_sheet(images, thumb=160, cols=4, pad=4, bg=(30, 30, 30)):
    if not images:
        return None
    rows = (len(images) + cols - 1) // cols
    cols = min(cols, len(images))
    sheet = Image.new("RGB", (cols * (thumb + pad) + pad, rows * (thumb + pad) + pad), bg)
    for i, im in enumerate(images):
        t = im.convert("RGB").copy()
        t.thumbnail((thumb, thumb))
        x = pad + (i % cols) * (thumb + pad) + (thumb - t.size[0]) // 2
        y = pad + (i // cols) * (thumb + pad) + (thumb - t.size[1]) // 2
        sheet.paste(t, (x, y))
    return sheet


def side_by_side(a, b, height=512, pad=8, bg=(255, 255, 255)):
    def fit(im):
        w, h = im.size
        return im.resize((max(1, int(w * height / h)), height), Image.BICUBIC)
    a, b = fit(a.convert("RGB")), fit(b.convert("RGB"))
    out = Image.new("RGB", (a.size[0] + b.size[0] + pad, height), bg)
    out.paste(a, (0, 0))
    out.paste(b, (a.size[0] + pad, 0))
    return out


def region_words(box, size):
    """Map a box centre to MMAD's coarse location vocabulary (3x3 grid)."""
    x0, y0, x1, y1 = box
    w, h = size
    cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
    col = "left" if cx < 1 / 3 else ("right" if cx > 2 / 3 else "center")
    row = "top" if cy < 1 / 3 else ("bottom" if cy > 2 / 3 else "center")
    if row == "center" and col == "center":
        return "center"
    if row == "center":
        return col
    if col == "center":
        return row
    return f"{row} {col}"
