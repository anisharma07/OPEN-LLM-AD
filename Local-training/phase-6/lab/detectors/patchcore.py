"""
PatchCore (Roth et al., CVPR 2022) - configurable re-implementation for the lab.

Pipeline
  1. Mid-level features from a frozen ImageNet backbone (default layer2+layer3).
  2. Local neighbourhood aggregation (3x3 average pooling), layer3 upsampled to
     layer2 resolution, concatenated, then adaptive-average-pooled to `feat_dim`.
  3. Memory bank = patches of k normal references, reduced by greedy (k-center)
     coreset subsampling computed on a random 128-d projection.
  4. Anomaly map = distance to the nearest memory patch (k-NN mean), upsampled
     and Gaussian-smoothed. Image score = max (or top-1% mean) of the map.
  5. tau (normal threshold) is calibrated on held-out normal references: the
     largest map value observed on normal images, times a margin.

Differences from Phase 5's `patchcore_expert.py`: backbone/layers/coreset/k-NN
are parameters, coreset is greedy instead of random, banks are cached per
(config, reference set) instead of per category, and calibration never scores
references against a bank that already contains them.
"""

import threading
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as tvm
from PIL import Image
from scipy.ndimage import gaussian_filter
from torchvision import transforms

BACKBONES = {
    "wide_resnet50_2": lambda: tvm.wide_resnet50_2(weights=tvm.Wide_ResNet50_2_Weights.DEFAULT),
    "resnet50": lambda: tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT),
    "resnet18": lambda: tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT),
}
LAYER_SETS = {
    "layer2+layer3": ("layer2", "layer3"),
    "layer2": ("layer2",),
    "layer3": ("layer3",),
    "layer1+layer2+layer3": ("layer1", "layer2", "layer3"),
}

MAP_OUT = 256
_MODEL_LOCK = threading.Lock()
_BACKBONE_CACHE = {}
_BANK_CACHE = OrderedDict()
_BANK_CACHE_MAX = 24
_FEAT_CACHE = OrderedDict()   # per reference image features (overlapping ref sets reuse them)
_FEAT_CACHE_MAX = 96


def _device(pref="auto"):
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


class _FeatureExtractor:
    def __init__(self, backbone, device):
        self.device = device
        self.model = BACKBONES[backbone]().to(device).eval()
        self._feats = {}
        for name in ("layer1", "layer2", "layer3"):
            getattr(self.model, name).register_forward_hook(self._hook(name))

    def _hook(self, name):
        def fn(_m, _i, out):
            self._feats[name] = out
        return fn

    @torch.no_grad()
    def __call__(self, batch, layers, feat_dim):
        self._feats.clear()
        self.model(batch)
        ref_hw = self._feats[layers[0]].shape[-2:]
        maps = []
        for ln in layers:
            f = F.avg_pool2d(self._feats[ln], 3, stride=1, padding=1)
            if f.shape[-2:] != ref_hw:
                f = F.interpolate(f, size=ref_hw, mode="bilinear", align_corners=False)
            maps.append(f)
        feats = torch.cat(maps, dim=1)                    # B,C,H,W
        b, c, h, w = feats.shape
        patches = feats.permute(0, 2, 3, 1).reshape(-1, c)
        if feat_dim and feat_dim < c:
            patches = F.adaptive_avg_pool1d(patches.unsqueeze(1), feat_dim).squeeze(1)
        return patches.reshape(b, h * w, -1), (h, w)


def _get_extractor(backbone, device):
    key = (backbone, str(device))
    with _MODEL_LOCK:
        if key not in _BACKBONE_CACHE:
            _BACKBONE_CACHE[key] = _FeatureExtractor(backbone, device)
        return _BACKBONE_CACHE[key]


def greedy_coreset(features, ratio, proj_dim=128, seed=0):
    """k-center greedy subsampling (PatchCore Alg. 1) on a random projection."""
    n = features.shape[0]
    m = max(1, int(n * ratio))
    if ratio >= 1.0 or m >= n:
        return features
    g = torch.Generator(device="cpu").manual_seed(seed)
    proj = torch.randn(features.shape[1], proj_dim, generator=g).to(features.device) / proj_dim ** 0.5
    z = features @ proj
    idx = [int(torch.randint(n, (1,), generator=g))]
    min_d = torch.cdist(z, z[idx[-1]].unsqueeze(0)).squeeze(1)
    for _ in range(m - 1):
        nxt = int(torch.argmax(min_d))
        idx.append(nxt)
        min_d = torch.minimum(min_d, torch.cdist(z, z[nxt].unsqueeze(0)).squeeze(1))
    return features[idx]


class PatchCore:
    def __init__(self, backbone="wide_resnet50_2", layers="layer2+layer3", resolution=256,
                 feat_dim=1024, coreset_ratio=0.1, knn=1, sigma=4.0, score_agg="max",
                 calib_margin=1.0, device="auto"):
        self.backbone = backbone
        self.layers = LAYER_SETS[layers]
        self.layers_name = layers
        self.res = int(resolution)
        self.feat_dim = int(feat_dim)
        self.coreset_ratio = float(coreset_ratio)
        self.knn = max(1, int(knn))
        self.sigma = float(sigma)
        self.score_agg = score_agg
        self.calib_margin = float(calib_margin)
        self.device = _device(device)
        self.tf = transforms.Compose([
            transforms.Resize((self.res, self.res)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    # ------------------------------------------------------------------ utils
    def _cfg_key(self):
        return (self.backbone, self.layers_name, self.res, self.feat_dim, self.coreset_ratio, str(self.device))

    def _embed(self, images):
        ext = _get_extractor(self.backbone, self.device)
        out, hw = [], None
        for i in range(0, len(images), 8):
            batch = torch.stack([self.tf(im) for im in images[i:i + 8]]).to(self.device)
            p, hw = ext(batch, self.layers, self.feat_dim)
            out.append(p)
        return torch.cat(out, 0), hw                       # N, HW, C

    def _score(self, patches, bank, hw):
        k = min(self.knn, bank.shape[0])
        out = []
        for i in range(0, patches.shape[0], 2048):          # chunked: bounded memory at high resolution
            d = torch.cdist(patches[i:i + 2048], bank)
            out.append(torch.topk(d, k, dim=1, largest=False).values.mean(1) if k > 1 else d.min(1).values)
        d = torch.cat(out)
        m = d.reshape(1, 1, *hw)
        # maps are always emitted at MAP_OUT x MAP_OUT, whatever the processing resolution,
        # so downstream nodes (box, metrics) share one coordinate frame
        m = F.interpolate(m, size=(MAP_OUT, MAP_OUT), mode="bilinear", align_corners=False)
        m = m.squeeze().cpu().numpy()
        if self.sigma > 0:
            m = gaussian_filter(m, sigma=self.sigma)
        return m

    def _agg(self, amap):
        if self.score_agg == "top1pct_mean":
            flat = np.sort(amap.ravel())
            return float(flat[-max(1, flat.size // 100):].mean())
        return float(amap.max())

    # ------------------------------------------------------------------- bank
    def fit(self, ref_paths, ref_loader):
        """Build (or fetch) the memory bank and calibrated tau for these references."""
        key = self._cfg_key() + (tuple(ref_paths),)
        if key in _BANK_CACHE:
            _BANK_CACHE.move_to_end(key)
            return _BANK_CACHE[key]
        todo = [p for p in ref_paths if self._cfg_key() + (p,) not in _FEAT_CACHE]
        if todo:
            f_new, _ = self._embed([ref_loader(p) for p in todo])
            for p, f in zip(todo, f_new):
                _FEAT_CACHE[self._cfg_key() + (p,)] = f
                while len(_FEAT_CACHE) > _FEAT_CACHE_MAX:
                    _FEAT_CACHE.popitem(last=False)
        feats = torch.stack([_FEAT_CACHE[self._cfg_key() + (p,)] for p in ref_paths])   # N,HW,C
        g = int(round(feats.shape[1] ** 0.5))
        hw = (g, g)
        bank = greedy_coreset(feats.reshape(-1, feats.shape[-1]), self.coreset_ratio)

        tau_map = tau_score = None
        calib_scores = []
        n = len(ref_paths)
        if n >= 2:
            n_cal = max(1, n // 4)
            fit_part = feats[: n - n_cal].reshape(-1, feats.shape[-1])
            cal_bank = greedy_coreset(fit_part, self.coreset_ratio, seed=1)
            map_maxes = []
            for i in range(n - n_cal, n):
                amap = self._score(feats[i], cal_bank, hw)
                map_maxes.append(float(amap.max()))
                calib_scores.append(self._agg(amap))
            tau_map = max(map_maxes) * self.calib_margin
            tau_score = max(calib_scores) * self.calib_margin
        entry = {"bank": bank, "hw": hw, "tau_map": tau_map, "tau_score": tau_score,
                 "calib_scores": calib_scores, "n_refs": n, "bank_size": int(bank.shape[0])}
        _BANK_CACHE[key] = entry
        while len(_BANK_CACHE) > _BANK_CACHE_MAX:
            _BANK_CACHE.popitem(last=False)
        return entry

    @torch.no_grad()
    def predict(self, image: Image.Image, fitted):
        feats, hw = self._embed([image])
        amap = self._score(feats[0], fitted["bank"], hw)
        score = self._agg(amap)
        return amap.astype(np.float32), score
