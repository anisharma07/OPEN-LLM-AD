"""
WinCLIP / WinCLIP+ (Jeong et al., CVPR 2023) - re-implementation on HF CLIP.

What is faithful to the paper
  * Compositional Prompt Ensemble: state words x template sentences, averaged
    per class (normal / anomalous) in the joint embedding space.
  * Window-based masking: for every k x k window of patch tokens the ViT is run
    on [CLS] + that window's tokens only (with their original position
    embeddings); the window's [CLS] embedding is scored against the text.
  * Harmonic aggregation of window scores onto the patches each window covers,
    averaged over window scales.
  * WinCLIP+ (few-shot): window embeddings and penultimate-layer patch tokens
    of k normal references form a memory; score = (1 - max cosine) / 2;
    averaged with the zero-shot map.

What differs
  * Backbone is OpenAI ViT-B/16 @224 (14x14 grid) instead of LAION ViT-B/16+
    @240 (15x15), and the prompt ensemble is a subset of the paper's lists.
    Absolute numbers will therefore not match the paper's tables.
"""

import threading

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter

CLIP_MODELS = {
    "ViT-B/16 (openai)": "openai/clip-vit-base-patch16",
    "ViT-B/32 (openai)": "openai/clip-vit-base-patch32",
    "ViT-L/14 (openai)": "openai/clip-vit-large-patch14",
}

NORMAL_STATES = ["{}", "flawless {}", "perfect {}", "unblemished {}",
                 "{} without flaw", "{} without defect", "{} without damage"]
ANOMALY_STATES = ["damaged {}", "broken {}", "{} with flaw", "{} with defect", "{} with damage"]
TEMPLATES = [
    "a cropped photo of the {}.", "a close-up photo of a {}.", "a close-up photo of the {}.",
    "a bright photo of a {}.", "a dark photo of the {}.", "a photo of the {}.",
    "a photo of a {}.", "a jpeg corrupted photo of a {}.", "a blurry photo of the {}.",
    "a photo of the small {}.", "a photo of the large {}.",
    "there is a {} in the scene.", "this is a {} in the scene.",
    "a photo of a {} for visual inspection.", "a photo of the {} for anomaly detection.",
]

_LOCK = threading.Lock()
_MODELS = {}
_TEXT_CACHE = {}
_REF_CACHE = {}
_EMB_CACHE = {}

MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)


def _device(pref="auto"):
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def _load(model_name, device):
    key = (model_name, str(device))
    with _LOCK:
        if key not in _MODELS:
            from transformers import CLIPModel, CLIPTokenizer
            hf_id = CLIP_MODELS.get(model_name, model_name)
            model = CLIPModel.from_pretrained(hf_id).to(device).eval()
            tok = CLIPTokenizer.from_pretrained(hf_id)
            _MODELS[key] = (model, tok)
        return _MODELS[key]


def _pooled(out):
    return out if torch.is_tensor(out) else out.pooler_output


class WinCLIP:
    def __init__(self, model_name="ViT-B/16 (openai)", scales=(2, 3), n_templates=15,
                 sigma=4.0, device="auto"):
        self.model_name = model_name
        self.device = _device(device)
        self.model, self.tok = _load(model_name, self.device)
        vm = self.model.vision_model
        self.vt = getattr(vm, "vision_model", vm)          # tolerate v4/v5 layouts
        cfg = self.vt.config if hasattr(self.vt, "config") else self.model.config.vision_config
        self.image_size = cfg.image_size
        self.grid = cfg.image_size // cfg.patch_size
        self.scales = tuple(int(s) for s in scales if 1 <= int(s) <= self.grid)
        self.templates = TEMPLATES[: max(1, int(n_templates))]
        self.sigma = float(sigma)
        self.windows = {s: self._window_index(s) for s in self.scales}

    # ------------------------------------------------------------------ text
    @torch.no_grad()
    def text_features(self, object_name):
        key = (self.model_name, object_name, len(self.templates))
        if key in _TEXT_CACHE:
            return _TEXT_CACHE[key]
        feats = []
        for states in (NORMAL_STATES, ANOMALY_STATES):
            sents = [t.format(s.format(object_name)) for s in states for t in self.templates]
            tok = self.tok(sents, padding=True, return_tensors="pt").to(self.device)
            e = _pooled(self.model.get_text_features(**tok))
            e = F.normalize(e, dim=-1).mean(0)
            feats.append(F.normalize(e, dim=-1))
        T = torch.stack(feats)                                 # 2, D
        _TEXT_CACHE[key] = T
        return T

    # ----------------------------------------------------------------- image
    def _pixels(self, img: Image.Image):
        im = img.convert("RGB").resize((self.image_size, self.image_size), Image.BICUBIC)
        x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)
        return ((x - MEAN) / STD).unsqueeze(0).to(self.device)

    def _window_index(self, s):
        g = self.grid
        idx = []
        for r in range(g - s + 1):
            for c in range(g - s + 1):
                idx.append([(r + i) * g + (c + j) for i in range(s) for j in range(s)])
        return torch.tensor(idx, dtype=torch.long)             # Nw, s*s

    @torch.no_grad()
    def _tokens(self, px):
        emb = self.vt.embeddings
        patches = emb.patch_embedding(px.to(emb.patch_embedding.weight.dtype)).flatten(2).transpose(1, 2)
        pos = emb.position_embedding.weight
        cls = (emb.class_embedding + pos[0]).view(1, 1, -1)
        return cls, patches[0] + pos[1:]                       # (1,1,D), (G*G, D)

    def _encode_seq(self, seq, capture=None):
        """Run the ViT encoder layer by layer; optionally return the output of layer `capture`."""
        h = self.vt.pre_layrnorm(seq)
        mid = None
        for i, layer in enumerate(self.vt.encoder.layers):
            out = layer(h, None)
            h = out[0] if isinstance(out, tuple) else out
            if capture is not None and i == capture:
                mid = h
        return (h, mid) if capture is not None else h

    @torch.no_grad()
    def embed(self, img):
        """Image-level embedding, per-scale window [CLS] embeddings, and penultimate-layer
        patch tokens (used only for visual-visual matching in WinCLIP+: CLIP's last-layer
        patch tokens are poorly aligned with text and give inverted zero-shot maps)."""
        cls, toks = self._tokens(self._pixels(img))
        n_layers = len(self.vt.encoder.layers)
        full, mid = self._encode_seq(torch.cat([cls, toks.unsqueeze(0)], 1), capture=n_layers - 2)
        img_emb = F.normalize(self.model.visual_projection(self.vt.post_layernorm(full[:, 0])), dim=-1)[0]
        patch_emb = F.normalize(mid[0, 1:], dim=-1)
        wins = {}
        for s, idx in self.windows.items():
            seq = torch.cat([cls.expand(idx.shape[0], 1, -1), toks[idx.to(toks.device)]], 1)
            out = []
            for i in range(0, seq.shape[0], 256):
                h = self._encode_seq(seq[i:i + 256])
                out.append(self.model.visual_projection(self.vt.post_layernorm(h[:, 0])))
            wins[s] = F.normalize(torch.cat(out), dim=-1)      # Nw, D
        return {"image": img_emb, "windows": wins, "patches": patch_emb}

    # ------------------------------------------------------------- scoring
    def _harmonic(self, win_scores, s):
        g = self.grid
        inv_sum = torch.zeros(g * g, device=win_scores.device)
        cnt = torch.zeros(g * g, device=win_scores.device)
        idx = self.windows[s].to(win_scores.device)
        inv = 1.0 / win_scores.clamp_min(1e-6)
        for j in range(idx.shape[1]):
            inv_sum.index_add_(0, idx[:, j], inv)
            cnt.index_add_(0, idx[:, j], torch.ones_like(inv))
        return (cnt / inv_sum.clamp_min(1e-6)).view(g, g)

    def _upsample(self, grid_map):
        m = F.interpolate(grid_map[None, None], size=(256, 256), mode="bilinear", align_corners=False)
        m = m.squeeze().float().cpu().numpy()
        return gaussian_filter(m, sigma=self.sigma) if self.sigma > 0 else m

    @torch.no_grad()
    def zero_shot(self, emb, T, temperature=100.0):
        img_p = torch.softmax(temperature * emb["image"] @ T.T, -1)[1].item()
        maps = []
        for s, w in emb["windows"].items():
            p = torch.softmax(temperature * w @ T.T, -1)[:, 1]
            maps.append(self._harmonic(p, s))
        return img_p, torch.stack(maps).mean(0)

    @torch.no_grad()
    def few_shot(self, emb, memory):
        maps = []
        for s, w in emb["windows"].items():
            d = (1 - (w @ memory["windows"][s].T).max(1).values) / 2
            maps.append(self._harmonic(d.clamp_min(1e-6), s))
        d = (1 - (emb["patches"] @ memory["patches"].T).max(1).values) / 2
        maps.append(d.view(self.grid, self.grid))
        return torch.stack(maps).mean(0)

    def _ref_embed(self, path, loader):
        key = (self.model_name, self.scales, path)
        if key not in _EMB_CACHE:
            _EMB_CACHE[key] = self.embed(loader(path))
            if len(_EMB_CACHE) > 256:
                _EMB_CACHE.pop(next(iter(_EMB_CACHE)))
        return _EMB_CACHE[key]

    @torch.no_grad()
    def build_memory(self, ref_paths, loader):
        key = (self.model_name, self.scales, tuple(ref_paths))
        if key in _REF_CACHE:
            return _REF_CACHE[key]
        embs = [self._ref_embed(p, loader) for p in ref_paths]
        mem = {
            "windows": {s: torch.cat([e["windows"][s] for e in embs]) for s in self.scales},
            "patches": torch.cat([e["patches"] for e in embs]),
        }
        _REF_CACHE[key] = mem
        if len(_REF_CACHE) > 24:
            _REF_CACHE.pop(next(iter(_REF_CACHE)))
        return mem

    @torch.no_grad()
    def predict(self, img, object_name, mode="zero-shot", ref_paths=(), loader=None,
                temperature=100.0, fs_weight=0.5, emb=None):
        """Returns (map[256x256], image_score, info)."""
        T = self.text_features(object_name)
        emb = emb if emb is not None else self.embed(img)
        img_p, zs_grid = self.zero_shot(emb, T, temperature)
        info = {"zero_shot_image_prob": round(img_p, 4)}
        if mode == "few-shot (WinCLIP+)" and ref_paths:
            mem = self.build_memory(list(ref_paths), loader)
            fs_grid = self.few_shot(emb, mem)
            grid_map = (1 - fs_weight) * zs_grid + fs_weight * fs_grid
            score = (1 - fs_weight) * img_p + fs_weight * float(fs_grid.max())
            info["few_shot_max"] = round(float(fs_grid.max()), 4)
        else:
            grid_map, score = zs_grid, img_p
        return self._upsample(grid_map).astype(np.float32), float(score), info

    @torch.no_grad()
    def calibrate(self, object_name, ref_paths, loader, mode, temperature=100.0, fs_weight=0.5):
        """Calibrate on normal refs: held-out refs for WinCLIP+, all refs for zero-shot.

        Returns (tau_map, tau_score, calib_scores); taus are the maxima seen on normals."""
        refs = list(ref_paths)
        if (mode.startswith("few") and len(refs) < 2) or not refs:
            return None, None, []
        n_cal = max(1, len(refs) // 4)
        fit, cal = refs[: len(refs) - n_cal], refs[len(refs) - n_cal:]
        scores, map_maxes = [], []
        for p in (cal if mode.startswith("few") else refs):
            m, s, _ = self.predict(loader(p), object_name, mode, fit, loader, temperature, fs_weight,
                                   emb=self._ref_embed(p, loader))
            scores.append(s)
            map_maxes.append(float(m.max()))
        return max(map_maxes), max(scores), scores
