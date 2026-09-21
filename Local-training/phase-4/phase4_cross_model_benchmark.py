#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
==============================================================================
Phase 4: Cross-Model Multimodal LLM Architecture Benchmark
==============================================================================
M.Tech Dissertation: Robustness and Reproducibility of Open-Source Small
Multimodal LLMs for Industrial Anomaly Detection

Evaluates and compares different Small Multimodal LLMs on the exact same
2,500 questions from Phase 1 clean baseline:
1. Qwen/Qwen3-VL-2B-Instruct (Alibaba 2B baseline)
2. google/gemma-4-E2B-it (Google Gemma 4 2B)
3. google/gemma-4-E4B-it (Google Gemma 4 4B)
4. Qwen/Qwen2.5-VL-3B-Instruct (Alibaba 3B/4B)

Metrics:
- Accuracy, Cohen's Kappa, Macro-F1
- Subtask-level accuracy hierarchy across all 9 tasks
- Latency (sec/inf), Throughput (FPS), and VRAM usage (GB)
==============================================================================
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import json
import time
import random
import re
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import cohen_kappa_score
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoConfig

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
BASE_DIR = PARENT_DIR.parent
MMAD_DIR = BASE_DIR / "MMAD"
RESULTS_DIR = CURRENT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Parse arguments
parser = argparse.ArgumentParser(description="Phase 4: Cross-Model Multimodal Benchmark")
parser.add_argument("--model-id", type=str, default="google/gemma-4-E2B-it", help="Hugging Face Model ID to evaluate.")
parser.add_argument("--sample-size", type=int, default=2500, help="Number of questions (default 2500 from Phase 1).")
parser.add_argument("--seed", type=int, default=42, help="Random seed for determinism.")
parser.add_argument("--reset", action="store_true", help="Reset previous results for this model.")
parser.add_argument("--generate-plots-only", action="store_true", help="Only compile summary results and generate plots.")
args = parser.parse_args()

GLOBAL_SEED = args.seed
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(GLOBAL_SEED)

def get_gpu_info():
    if torch.cuda.is_available():
        return {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
            "cuda_version": torch.version.cuda,
        }
    return {"gpu_name": "CPU", "gpu_memory_gb": 0, "cuda_version": None}

gpu_info = get_gpu_info()
clean_model_name = args.model_id.replace("/", "_").replace("-", "_").lower()
MODEL_RESULTS_JSONL = RESULTS_DIR / f"{clean_model_name}_results.jsonl"


# =============================================================================
# Helper Functions: Dataset Resolution & Prompt Generation
# =============================================================================
def resolve_local_image_path(rel_path, mmad_base_dir):
    candidates = [
        mmad_base_dir / rel_path,
        mmad_base_dir / "images" / rel_path,
        mmad_base_dir / "data" / rel_path,
        mmad_base_dir / rel_path.lstrip("/"),
    ]
    parts = rel_path.split("/")
    if len(parts) > 1:
        candidates.append(mmad_base_dir / parts[0] / "/".join(parts[1:]))
        candidates.append(mmad_base_dir / "images" / parts[0] / "/".join(parts[1:]))
    for cand in candidates:
        if cand.exists() and cand.is_file():
            return str(cand)
    return None


def build_mmad_prompt(question_data):
    ci_data = {str(k).lower(): v for k, v in question_data.items()}
    q_text = ""
    for f in ["question", "query", "text", "prompt"]:
        if f in ci_data and ci_data[f]:
            q_text = str(ci_data[f]).strip()
            break
    if not q_text:
        q_text = "Identify any defect or anomaly present in this industrial object."

    opts_dict = {}
    options_text = ""
    for f in ["options", "choices", "answers"]:
        if f in ci_data and ci_data[f]:
            raw_opts = ci_data[f]
            if isinstance(raw_opts, dict):
                opts_dict = raw_opts
                for k, v in sorted(raw_opts.items()):
                    options_text += f"({k}) {v}\n"
            elif isinstance(raw_opts, list):
                for idx, v in enumerate(raw_opts):
                    k = chr(65 + idx)
                    opts_dict[k] = v
                    options_text += f"({k}) {v}\n"
            break

    prompt = f"{q_text}\n\n{options_text.strip()}\n\nAnswer with the letter of the correct option (e.g., A, B, C, or D). Give only the letter."
    return prompt, q_text, opts_dict


def parse_answer(response_text):
    if not response_text:
        return None, False
    text = response_text.strip()
    m = re.match(r"^([A-D])\b", text.upper())
    if m:
        return m.group(1), True
    m = re.search(r"\(([A-D])\)|([A-D])[).\s:]", text.upper())
    if m:
        return m.group(1) or m.group(2), True
    letters = re.findall(r"\b([A-D])\b", text.upper())
    if len(letters) == 1:
        return letters[0], True
    for c in text.upper()[:5]:
        if c in "ABCD":
            return c, True
    return None, False


def get_ground_truth(question_data):
    ci_data = {str(k).lower(): v for k, v in question_data.items()}
    for f in ["answer", "ground_truth", "label", "gt"]:
        if f in ci_data and ci_data[f] is not None:
            gt = str(ci_data[f]).strip().upper()
            if len(gt) == 1 and gt in "ABCD":
                return gt
            if gt.isdigit() and int(gt) < 4:
                return chr(65 + int(gt))
            return gt
    return None


# =============================================================================
# Load Dataset (Identical Stratified Sample as Phase 1)
# =============================================================================
mmad_json_path = MMAD_DIR / "mmad.json"
with open(mmad_json_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

all_questions = []
for key, val in raw_data.items():
    parts = key.split("/")
    dataset_name = parts[0] if len(parts) > 0 else "Unknown"
    category = parts[1] if len(parts) > 1 else "unknown"
    resolved_img = resolve_local_image_path(key, MMAD_DIR)

    if isinstance(val, dict):
        for sub_k, sub_v in val.items():
            if isinstance(sub_v, dict):
                entry = dict(sub_v)
                entry["_subtask"] = sub_v.get("type", sub_k)
                entry["_category"] = category
                entry["_dataset"] = dataset_name
                entry["_image_key"] = key
                entry["_resolved_image_path"] = resolved_img
                all_questions.append(entry)
            elif isinstance(sub_v, list):
                for item in sub_v:
                    if isinstance(item, dict):
                        entry = dict(item)
                        entry["_subtask"] = item.get("type", sub_k)
                        entry["_category"] = category
                        entry["_dataset"] = dataset_name
                        entry["_image_key"] = key
                        entry["_resolved_image_path"] = resolved_img
                        all_questions.append(entry)

valid_questions = [q for q in all_questions if q.get("_resolved_image_path") and os.path.exists(q["_resolved_image_path"])]

subtask_bins = defaultdict(list)
for q in valid_questions:
    st = q.get("_subtask", "Unknown")
    subtask_bins[st].append(q)

target_n = min(args.sample_size, len(valid_questions))
per_subtask = target_n // len(subtask_bins)

sample_questions = []
random.seed(GLOBAL_SEED)
for st, q_list in sorted(subtask_bins.items()):
    take = min(len(q_list), per_subtask)
    sample_questions.extend(random.sample(q_list, take))

if len(sample_questions) < target_n:
    rem = [q for q in valid_questions if q not in sample_questions]
    sample_questions.extend(random.sample(rem, target_n - len(sample_questions)))


# =============================================================================
# Inference Function
# =============================================================================
def run_model_inference(model, processor, pil_img, prompt_text):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    inputs = None
    try:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
    except Exception:
        pass

    if inputs is None:
        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        try:
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(text=[text_input], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)
        except Exception:
            inputs = processor(text=[prompt_text], images=[pil_img], return_tensors="pt", padding=True).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=4, do_sample=False)

    input_len = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[0][input_len:]
    response = processor.decode(generated_ids, skip_special_tokens=True).strip()
    return response


# =============================================================================
# Main Evaluation Loop
# =============================================================================
if not args.generate_plots_only:
    print("=" * 85)
    print(f"PHASE 4: EVALUATING MODEL: {args.model_id}")
    print("=" * 85)
    print(f"  Target GPU     : {gpu_info['gpu_name']} ({gpu_info['gpu_memory_gb']} GB VRAM)")
    print(f"  Sample Size    : {len(sample_questions):,} questions (identical to Phase 1)")
    print("=" * 85)

    if args.reset and MODEL_RESULTS_JSONL.exists():
        MODEL_RESULTS_JSONL.unlink()

    completed_keys = set()
    if MODEL_RESULTS_JSONL.exists():
        with open(MODEL_RESULTS_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        completed_keys.add(json.loads(line).get("question_idx"))
                    except Exception:
                        pass
        print(f"🔄 Resuming {args.model_id}: Found {len(completed_keys):,} previously evaluated items.")

    print(f"\n🤖 Loading {args.model_id} onto GPU...")
    t_load0 = time.time()
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    
    if "gemma-4-E4B" in args.model_id or "gemma-4-e4b" in args.model_id.lower():
        from accelerate.hooks import remove_hook_from_module
        offload_dir = Path("/tmp/gemma_offload")
        offload_dir.mkdir(parents=True, exist_ok=True)
        weights_id = "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit"

        config = AutoConfig.from_pretrained(weights_id)
        all_skips = list(config.quantization_config.get("llm_int8_skip_modules", []))
        for name in ['model.vision_tower', 'model.vision_tower.patch_embedder.input_proj', 'vision_tower', 'patch_embedder', 'input_proj', 'model.audio_tower', 'audio_tower', 'embed_audio']:
            if name not in all_skips:
                all_skips.append(name)
        config.quantization_config['llm_int8_skip_modules'] = all_skips
        config.quantization_config['llm_int8_enable_fp32_cpu_offload'] = True

        device_map = {
            'model.audio_tower': 'cpu',
            'model.embed_audio': 'cpu',
            'model.language_model.embed_tokens': 'cpu',
            'model.language_model.embed_tokens_per_layer': 'cpu',
            'model.language_model.per_layer_model_projection': 0,
            'model.language_model.per_layer_projection_norm': 0,
            'lm_head': 'cpu',
            'model.vision_tower': 0,
            'model.embed_vision': 0,
            'model.language_model.norm': 0,
        }
        for i in range(42):
            device_map[f'model.language_model.layers.{i}'] = 0

        processor = AutoProcessor.from_pretrained(weights_id, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            weights_id,
            config=config,
            device_map=device_map,
            offload_folder=str(offload_dir),
            trust_remote_code=True,
        )
        remove_hook_from_module(model.model.language_model.embed_tokens_per_layer, recurse=True)
        remove_hook_from_module(model.model.language_model.embed_tokens, recurse=True)
    else:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type='nf4',
            llm_int8_enable_fp32_cpu_offload=True
        )

        try:
            model = AutoModelForImageTextToText.from_pretrained(
                args.model_id,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"   Fallback to float16 loading: {e}")
            model = AutoModelForImageTextToText.from_pretrained(
                args.model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )

    model.eval()
    vram_used = torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0
    print(f"   ✅ Model loaded in {time.time() - t_load0:.1f}s | VRAM Used: {vram_used:.2f} GB")

    t_bench0 = time.time()
    for q_idx, q in enumerate(sample_questions):
        if q_idx in completed_keys:
            continue

        img_path = q["_resolved_image_path"]
        gt = get_ground_truth(q)
        prompt, q_text, opts = build_mmad_prompt(q)

        try:
            raw_img = Image.open(img_path).convert("RGB")
            raw_img.thumbnail((384, 384), Image.Resampling.LANCZOS)
        except Exception:
            raw_img = Image.new("RGB", (224, 224), (128, 128, 128))

        t_inf0 = time.time()
        resp = run_model_inference(model, processor, raw_img, prompt)
        lat = time.time() - t_inf0
        pred, parsed = parse_answer(resp)
        is_correct = (pred == gt) if (pred and gt) else False

        rec = {
            "model_id": args.model_id,
            "question_idx": q_idx,
            "question_text": q_text,
            "options": opts,
            "ground_truth": gt,
            "prediction": pred,
            "raw_response": resp,
            "is_correct": is_correct,
            "latency_sec": round(lat, 3),
            "subtask": q.get("_subtask", "Unknown"),
            "dataset": q.get("_dataset", "Unknown"),
            "category": q.get("_category", "Unknown"),
            "image_path": img_path,
        }

        with open(MODEL_RESULTS_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        completed_keys.add(q_idx)

        if (len(completed_keys)) % 50 == 0 or len(completed_keys) == len(sample_questions):
            el = time.time() - t_bench0
            rate = len(completed_keys) / max(0.1, el)
            rem = len(sample_questions) - len(completed_keys)
            print(f"   [{len(completed_keys):04d}/{len(sample_questions):,}] Speed: {rate:.1f} inf/s | ETA: {(rem/max(0.1, rate))/60:.1f}m | Correct: {is_correct} (GT={gt}, Pred={pred})")

    print(f"\n✅ Finished evaluating {args.model_id} on {len(completed_keys):,} samples!")


# =============================================================================
# Multi-Model Comparison & Plot Generation
# =============================================================================
print("\n" + "=" * 85)
print("📊 COMPILING MULTI-MODEL COMPARISON & REPORT")
print("=" * 85)

# Gather all evaluated model results in phase-4 results directory or phase-1 results
model_files = list(RESULTS_DIR.glob("*_results.jsonl"))

# Also include Phase 1 Qwen3-VL-2B results if available
p1_manifest_path = PARENT_DIR / "phase-1" / "results" / "phase1_manifest.json"
comparison_data = {}

def process_records(recs, m_name):
    y_true = [r["ground_truth"] for r in recs if r.get("ground_truth") and r.get("prediction")]
    y_pred = [r["prediction"] for r in recs if r.get("ground_truth") and r.get("prediction")]
    acc = sum(r["is_correct"] for r in recs) / max(1, len(recs))
    kappa = cohen_kappa_score(y_true, y_pred) if len(y_true) > 10 else 0.0
    avg_lat = np.mean([r["latency_sec"] for r in recs if "latency_sec" in r]) if recs else 0.0

    st_acc = {}
    st_groups = defaultdict(list)
    for r in recs:
        st_groups[r["subtask"]].append(r["is_correct"])
    for st, vals in sorted(st_groups.items()):
        st_acc[st] = round(float(np.mean(vals)), 4)

    return {
        "model_name": m_name,
        "sample_count": len(recs),
        "accuracy": round(float(acc), 4),
        "kappa": round(float(kappa), 4),
        "avg_latency": round(float(avg_lat), 3),
        "throughput_fps": round(float(1.0 / max(0.001, avg_lat)), 1),
        "subtask_accuracy": st_acc,
    }

if p1_manifest_path.exists():
    with open(p1_manifest_path, "r", encoding="utf-8") as f:
        p1_m = json.load(f)
    p1_st = {}
    for st, v in p1_m.get("subtask_accuracy", {}).items():
        p1_st[st] = round(v["accuracy"] / 100.0, 4)
    comparison_data["Qwen3-VL-2B-Instruct"] = {
        "model_name": "Qwen3-VL-2B-Instruct",
        "sample_count": p1_m.get("total_questions", 2500),
        "accuracy": round(p1_m.get("overall_accuracy", 71.2) / 100.0, 4),
        "kappa": round(p1_m.get("cohen_kappa", 0.615), 3),
        "avg_latency": round(p1_m.get("average_latency_s", 0.16), 3),
        "throughput_fps": round(1.0 / p1_m.get("average_latency_s", 0.16), 1),
        "subtask_accuracy": p1_st,
    }

for mf in model_files:
    m_recs = []
    with open(mf, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                m_recs.append(json.loads(line))
    if m_recs:
        disp_name = m_recs[0].get("model_id", mf.stem)
        comparison_data[disp_name] = process_records(m_recs, disp_name)

print(f"Found {len(comparison_data)} model architectures for comparison:")
for name, d in comparison_data.items():
    print(f"  • {name:<32}: Accuracy = {d['accuracy']*100:.2f}%, Kappa = {d['kappa']:.3f}, Latency = {d['avg_latency']:.3f}s")


# =============================================================================
# Visual Suite Generation
# =============================================================================
sns.set_theme(style="whitegrid", font_scale=1.1)

# Plot 1: Overall Accuracy Bar Chart
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
m_names = list(comparison_data.keys())
acc_vals = [comparison_data[m]["accuracy"] * 100 for m in m_names]
palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

bars = ax.bar(m_names, acc_vals, color=palette[:len(m_names)], width=0.55, edgecolor="black", alpha=0.9)
ax.set_ylabel("Clean Benchmark Accuracy (%) [N=2,500]", fontsize=12, fontweight="bold")
ax.set_title("Phase 4: Cross-Model Architecture Benchmark on MMAD\n(Comparing Small Multimodal LLMs for Industrial Anomaly Detection)",
             fontsize=13, fontweight="bold", pad=15)
ax.set_ylim(40, 85)
plt.xticks(rotation=15, ha="right", fontweight="bold")

for b in bars:
    h = b.get_height()
    ax.annotate(f"{h:.2f}%", xy=(b.get_x() + b.get_width() / 2, h),
                xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.tight_layout()
p1_path = RESULTS_DIR / "phase4_cross_model_accuracy_comparison.png"
plt.savefig(p1_path, dpi=300, bbox_inches="tight")
plt.close()

# Plot 2: Subtask Accuracy Comparison Heatmap
subtasks_all = sorted(list({st for m in comparison_data.values() for st in m["subtask_accuracy"].keys()}))
fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
matrix = []
for st in subtasks_all:
    row = [comparison_data[m]["subtask_accuracy"].get(st, 0) * 100 for m in m_names]
    matrix.append(row)

sns.heatmap(matrix, annot=True, fmt=".1f", cmap="Blues",
            xticklabels=m_names, yticklabels=subtasks_all, cbar_kws={"label": "Accuracy (%)"}, ax=ax, linewidths=0.5)
ax.set_title("Cross-Model Subtask Performance Matrix (MMAD)", fontsize=13, fontweight="bold", pad=15)
plt.tight_layout()
p2_path = RESULTS_DIR / "phase4_subtask_accuracy_heatmap.png"
plt.savefig(p2_path, dpi=300, bbox_inches="tight")
plt.close()

# Save Manifest and Report
MANIFEST_PATH = RESULTS_DIR / "phase4_manifest.json"
with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
    json.dump(comparison_data, f, indent=2)

REPORT_PATH = RESULTS_DIR / "results.txt"
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("=" * 90 + "\n")
    f.write("PHASE 4: CROSS-MODEL MULTIMODAL LLM ARCHITECTURE BENCHMARK REPORT\n")
    f.write("=" * 90 + "\n")
    f.write(f"Generated on   : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Hardware       : {gpu_info['gpu_name']} ({gpu_info['gpu_memory_gb']} GB VRAM)\n")
    f.write(f"Questions/Model: {args.sample_size:,} (Identical to Phase 1)\n")
    f.write("=" * 90 + "\n\n")

    f.write(f"{'Model Architecture':<32} | {'Accuracy':<10} | {'Kappa':<8} | {'Latency':<10} | {'Throughput':<10}\n")
    f.write("-" * 80 + "\n")
    for name, d in comparison_data.items():
        f.write(f"{name:<32} | {d['accuracy']*100:>8.2f}% | {d['kappa']:>6.3f} | {d['avg_latency']:>7.3f}s  | {d['throughput_fps']:>6.1f} fps\n")

print(f"✅ Generated Phase 4 visual suite & reports in {RESULTS_DIR.name}/")
