"""
MLLM backend: discovers vision-language models in the local HF cache, keeps a
single model resident (swapping on demand), and answers multiple-choice
questions either by greedy/sampled generation or by scoring the option-letter
logits of the first answer token (gives a probability per option).
"""

import gc
import hashlib
import os
import random
import re
import threading
import time
from pathlib import Path

import torch
from PIL import Image

CHANCE_BASELINE = "chance-baseline (uniform random letter)"

# Families the generic AutoModelForImageTextToText path is known to handle.
KNOWN_VLM_PREFIXES = (
    "Qwen/Qwen3-VL", "Qwen/Qwen2.5-VL", "Qwen/Qwen2-VL",
    "HuggingFaceTB/SmolVLM", "google/gemma-3", "google/gemma-4",
)
PREFERRED_ORDER = [
    "Qwen/Qwen3-VL-2B-Instruct", "Qwen/Qwen3-VL-4B-Instruct", "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen/Qwen2.5-VL-3B-Instruct", "Qwen/Qwen2-VL-2B-Instruct",
    "HuggingFaceTB/SmolVLM-256M-Instruct", "HuggingFaceTB/SmolVLM-500M-Instruct",
    "google/gemma-4-E2B-it", "google/gemma-4-E4B-it",
]


def hf_cache_dir():
    base = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    return Path(base)


def list_local_models():
    found = []
    d = hf_cache_dir()
    if d.exists():
        for p in d.iterdir():
            if not p.name.startswith("models--") or not (p / "snapshots").exists():
                continue
            mid = p.name[len("models--"):].replace("--", "/", 1)
            if mid.startswith(KNOWN_VLM_PREFIXES) and any((p / "snapshots").iterdir()):
                found.append(mid)
    ordered = [m for m in PREFERRED_ORDER if m in found] + sorted(m for m in found if m not in PREFERRED_ORDER)
    if not ordered:
        ordered = list(PREFERRED_ORDER[:1])
    return ordered + [CHANCE_BASELINE]


def device_info():
    info = {"cuda": torch.cuda.is_available(), "device": "cuda" if torch.cuda.is_available() else "cpu"}
    if info["cuda"]:
        props = torch.cuda.get_device_properties(0)
        info.update(gpu=torch.cuda.get_device_name(0), vram_gb=round(props.total_memory / 1e9, 2),
                    vram_used_gb=round(torch.cuda.memory_allocated(0) / 1e9, 2))
    try:
        with open("/proc/meminfo") as f:
            mem = {l.split(":")[0]: int(l.split()[1]) for l in f}
        info["ram_total_gb"] = round(mem["MemTotal"] / 1e6, 1)
        info["ram_avail_gb"] = round(mem["MemAvailable"] / 1e6, 1)
    except Exception:
        pass
    return info


def _limit_pixels(img: Image.Image, max_pixels: int):
    w, h = img.size
    if max_pixels and w * h > max_pixels:
        s = (max_pixels / (w * h)) ** 0.5
        img = img.resize((max(28, int(w * s)), max(28, int(h * s))), Image.BICUBIC)
    return img


class ModelManager:
    def __init__(self):
        self.lock = threading.RLock()
        self.model = None
        self.processor = None
        self.key = None
        self.load_seconds = None

    # -------------------------------------------------------------- loading
    def status(self):
        return {"loaded": self.key[0] if self.key else None,
                "precision": self.key[1] if self.key else None,
                "device": self.key[2] if self.key else None,
                "load_seconds": self.load_seconds}

    def unload(self):
        with self.lock:
            self.model = self.processor = self.key = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _resolve_precision(self, model_id, precision, device):
        if precision != "auto":
            return precision
        if device == "cuda":
            # fp16 needs ~2 bytes/param + vision tower + activations; on an 8 GB card
            # anything above ~2.5B params goes 4-bit (as Phase 4 did for the larger models).
            m = re.search(r"(\d+(?:\.\d+)?)B", model_id.split("/")[-1])
            params_b = float(m.group(1)) if m else 2.0
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            return "fp16" if params_b * 2.2 + 1.5 <= vram_gb * 0.8 else "4bit-nf4"
        small = any(s in model_id for s in ("256M", "500M"))
        return "fp32" if small else "bf16"

    def load(self, model_id, precision="auto", device="auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False "
                               "(is the NVIDIA kernel module loaded? try `nvidia-smi`).")
        precision = self._resolve_precision(model_id, precision, device)
        key = (model_id, precision, device)
        with self.lock:
            if self.key == key:
                return
            self.unload()
            t0 = time.time()
            from transformers import AutoModelForImageTextToText, AutoProcessor
            kwargs = {"trust_remote_code": True}
            if precision in ("4bit-nf4", "8bit"):
                if device != "cuda":
                    raise RuntimeError(f"{precision} quantization needs CUDA (bitsandbytes).")
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = (
                    BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                       bnb_4bit_compute_dtype=torch.float16)
                    if precision == "4bit-nf4" else BitsAndBytesConfig(load_in_8bit=True))
                kwargs["device_map"] = "auto"
            else:
                kwargs["dtype"] = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]
                kwargs["device_map"] = device
            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            self.model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs).eval()
            self.key = key
            self.load_seconds = round(time.time() - t0, 1)

    # ------------------------------------------------------------ inference
    def _inputs(self, images, prompt, system):
        content = [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": prompt}]
        messages = []
        if system:
            messages.append({"role": "system", "content": [{"type": "text", "text": system}]})
        messages.append({"role": "user", "content": content})
        try:
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        except Exception:
            # Processors whose chat template only accepts image placeholders (e.g. Idefics3).
            for m in messages:
                for c in m["content"]:
                    c.pop("image", None)
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=text, images=list(images) or None, return_tensors="pt")
        dev = next(self.model.parameters()).device
        dtype = next(p.dtype for p in self.model.parameters() if p.is_floating_point())
        out = {}
        for k, v in inputs.items():
            if torch.is_tensor(v):
                v = v.to(dev)
                if v.is_floating_point():
                    v = v.to(dtype)
            out[k] = v
        return out

    def _letter_ids(self, letters):
        tok = getattr(self.processor, "tokenizer", self.processor)
        ids = {}
        for L in letters:
            cands = set()
            for variant in (L, " " + L):
                enc = tok.encode(variant, add_special_tokens=False)
                if enc:
                    cands.add(enc[0])
            ids[L] = sorted(cands)
        return ids

    @torch.inference_mode()
    def answer(self, model_id, images, prompt, *, system="", precision="auto", device="auto",
               answer_mode="generate", max_new_tokens=8, temperature=0.0, top_p=1.0, seed=0,
               max_pixels=401408, letters=("A", "B", "C", "D"), rng_key=""):
        t0 = time.time()
        if model_id == CHANCE_BASELINE:
            h = int(hashlib.md5(f"{rng_key}|{seed}".encode()).hexdigest(), 16)
            letter = random.Random(h).choice(list(letters))
            return {"text": letter, "probs": None,
                    "latency": time.time() - t0, "input_tokens": 0}
        self.load(model_id, precision, device)
        t0 = time.time()   # latency excludes the one-off model load
        with self.lock:
            imgs = [_limit_pixels(im.convert("RGB"), max_pixels) for im in images]
            inputs = self._inputs(imgs, prompt, system)
            n_in = int(inputs["input_ids"].shape[-1])
            torch.manual_seed(seed)
            probs = None
            if answer_mode == "letter-logits":
                logits = self.model(**inputs).logits[0, -1].float()
                ids = self._letter_ids(letters)
                scores = {L: torch.logsumexp(logits[v], 0) for L, v in ids.items() if v}
                stacked = torch.softmax(torch.stack(list(scores.values())), 0)
                probs = {L: round(float(p), 5) for L, p in zip(scores, stacked)}
                text = max(probs, key=probs.get)
            else:
                gen = {"max_new_tokens": int(max_new_tokens)}
                if temperature and temperature > 0:
                    gen.update(do_sample=True, temperature=float(temperature), top_p=float(top_p))
                else:
                    gen.update(do_sample=False)
                out = self.model.generate(**inputs, **gen)
                text = self.processor.decode(out[0][n_in:], skip_special_tokens=True).strip()
            return {"text": text, "probs": probs, "latency": time.time() - t0, "input_tokens": n_in}


MODELS = ModelManager()
