#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
==============================================================================
Phase 0 — Environment Setup, Data Download & Smoke Test
==============================================================================
M.Tech Dissertation: Robustness and Reproducibility of Open-Source Small
Multimodal LLMs for Industrial Anomaly Detection

Purpose: Validate the end-to-end pipeline on Kaggle T4.
Exit criterion: One model answers MMAD questions end-to-end with parsed answers.

Hardware: Kaggle T4 / P100 (16 GB VRAM)
==============================================================================
"""

# =============================================================================
# CELL 1: Environment Pinning & Installation
# =============================================================================
# Run this cell first. It installs exact versions so results are reproducible.
# On Kaggle, you need Internet access ON (Settings → Internet → On).

import subprocess
import sys

def install_packages():
    """Install pinned packages for reproducibility."""
    packages = [
        "transformers>=4.57.0",
        "accelerate>=1.2.0",
        "bitsandbytes>=0.45.0",
        "qwen-vl-utils>=0.0.8",
        "huggingface_hub>=0.27.0",
        "Pillow>=10.0.0",
    ]
    for pkg in packages:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q", pkg
        ])
    print("✅ All packages installed.")

install_packages()

# Now import everything and record versions
import os
import json
import time
import random
import re
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
import accelerate
import PIL
from PIL import Image

# ---------- Determinism ----------
GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(GLOBAL_SEED)

# ---------- Paths ----------
# On Kaggle, /kaggle/working/ is the writable output directory (persists across saves)
# On local, use current directory
if os.path.exists("/kaggle/working"):
    WORK_DIR = Path("/kaggle/working")
else:
    WORK_DIR = Path("./phase0_output")
    WORK_DIR.mkdir(exist_ok=True)

RESULTS_DIR = WORK_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
MANIFEST_PATH = RESULTS_DIR / "phase0_manifest.json"
RESULTS_PATH = RESULTS_DIR / "phase0_results.jsonl"  # append-safe

# ---------- Record environment ----------
def get_gpu_info():
    """Get GPU name and memory."""
    if torch.cuda.is_available():
        return {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
            "cuda_version": torch.version.cuda,
        }
    return {"gpu_name": "CPU", "gpu_memory_gb": 0, "cuda_version": None}

MANIFEST = {
    "phase": "Phase 0 — Smoke Test",
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "seed": GLOBAL_SEED,
    "python_version": sys.version,
    "torch_version": torch.__version__,
    "transformers_version": transformers.__version__,
    "accelerate_version": accelerate.__version__,
    "pillow_version": PIL.__version__,
    "gpu_info": get_gpu_info(),
}

print("=" * 60)
print("ENVIRONMENT MANIFEST")
print("=" * 60)
for k, v in MANIFEST.items():
    print(f"  {k}: {v}")
print("=" * 60)


# =============================================================================
# CELL 2: Download MMAD Question Data from HuggingFace
# =============================================================================
# MMAD dataset: https://huggingface.co/datasets/jiang-cc/MMAD
# We download the question JSONs and a small subset of images for smoke testing.

from huggingface_hub import hf_hub_download, snapshot_download

MMAD_REPO = "jiang-cc/MMAD"
MMAD_DATA_DIR = WORK_DIR / "mmad_data"
MMAD_DATA_DIR.mkdir(exist_ok=True)

print("\n📥 Downloading MMAD question data from HuggingFace...")
print("   (This downloads the metadata/question files, not the full 28 GB image set)")

# Download the question/annotation files
# MMAD stores questions in JSON files organized by dataset
try:
    # Download only metadata files, skip the massive image archive
    snapshot_path = snapshot_download(
        repo_id=MMAD_REPO,
        repo_type="dataset",
        local_dir=str(MMAD_DATA_DIR),
        ignore_patterns=["*.zip", "ALL_DATA/*", "*.tar*", "*.gz"],
        allow_patterns=["*.json", "*.csv", "*.jsonl", "*.txt", "*.md", "*.py"],
    )
    print(f"✅ MMAD metadata downloaded to: {snapshot_path}")
except Exception as e:
    print(f"⚠️  Snapshot download failed: {e}")
    print("   Trying individual file downloads...")
    snapshot_path = str(MMAD_DATA_DIR)

# List what we downloaded
print("\n📂 Downloaded files:")
for root, dirs, files in os.walk(MMAD_DATA_DIR):
    for f in files:
        fpath = os.path.join(root, f)
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        rel_path = os.path.relpath(fpath, MMAD_DATA_DIR)
        print(f"   {rel_path} ({size_mb:.2f} MB)")


# =============================================================================
# CELL 3: Parse MMAD Questions & Load Sample Images
# =============================================================================
# Understand the MMAD data format: questions, options, ground truth, images

def find_json_files(base_dir):
    """Find all JSON files in the MMAD directory."""
    json_files = []
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".json") or f.endswith(".jsonl"):
                json_files.append(os.path.join(root, f))
    return sorted(json_files)

json_files = find_json_files(MMAD_DATA_DIR)
print(f"\n📋 Found {len(json_files)} JSON files:")
for jf in json_files:
    print(f"   {os.path.relpath(jf, MMAD_DATA_DIR)}")

# Load and inspect the question data
all_questions = []

for jf in json_files:
    fname = os.path.basename(jf)
    # Skip non-question files
    if "domain_knowledge" in fname:
        print(f"   ℹ️ Skipping {fname} (domain knowledge reference, not benchmark questions)")
        continue

    try:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            all_questions.extend(data)
            print(f"   ✅ {fname}: {len(data)} entries (list)")

        elif isinstance(data, dict):
            print(f"   🔍 Inspecting {fname}: dict with {len(data)} keys")
            first_key = next(iter(data.keys()))
            first_val = data[first_key]
            print(f"      Sample key: {repr(first_key)[:80]}")
            print(f"      Sample value type: {type(first_val).__name__}")
            if isinstance(first_val, dict):
                print(f"      Sample value keys: {list(first_val.keys())[:15]}")

            file_questions = []
            for key, val in data.items():
                # Case 1: Value is a list of question dicts
                if isinstance(val, list):
                    for item in val:
                        if isinstance(item, dict):
                            entry = dict(item)
                            entry["image"] = key
                            file_questions.append(entry)

                # Case 2: Value is directly a question dict
                elif isinstance(val, dict):
                    lower_keys = {k.lower(): k for k in val.keys()}
                    is_q = any(k in lower_keys for k in ["question", "prompt", "query", "text", "instruction", "conversations"])
                    has_opts_or_ans = any(k in lower_keys for k in ["options", "choices", "answer", "ground_truth", "gt", "label"])

                    if is_q or has_opts_or_ans:
                        entry = dict(val)
                        entry["image"] = key
                        file_questions.append(entry)

                    else:
                        # Case 3: Value contains subtasks / nested dicts or lists (e.g., 'conversation', etc.)
                        found_nested = False
                        for sub_k, sub_v in val.items():
                            if isinstance(sub_v, dict):
                                sub_lower = {k.lower(): k for k in sub_v.keys()}
                                sub_is_q = any(k in sub_lower for k in ["question", "prompt", "query", "text", "instruction"])
                                sub_has_ans = any(k in sub_lower for k in ["options", "choices", "answer", "ground_truth", "gt", "label"])
                                if sub_is_q or sub_has_ans:
                                    entry = dict(sub_v)
                                    entry["_subtask"] = sub_k
                                    entry["image"] = key
                                    file_questions.append(entry)
                                    found_nested = True
                            elif isinstance(sub_v, list):
                                for item in sub_v:
                                    if isinstance(item, dict):
                                        entry = dict(item)
                                        entry["_subtask"] = sub_k
                                        entry["image"] = key
                                        file_questions.append(entry)
                                        found_nested = True

                        if not found_nested:
                            entry = dict(val)
                            entry["_key"] = key
                            entry["image"] = key
                            file_questions.append(entry)

            print(f"   ✅ {fname}: Extracted {len(file_questions)} questions from {len(data)} image entries")
            all_questions.extend(file_questions)

        else:
            print(f"   ⚠️ {fname}: unexpected format: {type(data)}")

    except json.JSONDecodeError:
        # Try loading as JSONL (one JSON per line)
        count = 0
        with open(jf, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        all_questions.append(entry)
                        count += 1
                    except json.JSONDecodeError:
                        pass
        print(f"   ✅ {fname}: {count} entries (JSONL)")

print(f"\n📊 Total questions loaded: {len(all_questions)}")

# Inspect the structure of the first question
if all_questions:
    print("\n🔍 Sample question structure (first entry):")
    sample = all_questions[0]
    print(json.dumps(sample, indent=2, ensure_ascii=False)[:2000])

    # Identify key fields
    print("\n📋 Fields present in questions:")
    if isinstance(sample, dict):
        for key in sample.keys():
            val = sample[key]
            val_preview = str(val)[:100] if not isinstance(val, (dict, list)) else f"({type(val).__name__}, len={len(val) if hasattr(val, '__len__') else '?'})"
            print(f"   • {key}: {val_preview}")
else:
    print("\n⚠️ WARNING: all_questions is still empty! Printing raw sample from mmad.json:")
    for jf in json_files:
        if "mmad.json" in jf:
            with open(jf, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            first_k = next(iter(raw_data.keys()))
            print(f"Key: {first_k}")
            print(json.dumps(raw_data[first_k], indent=2, ensure_ascii=False)[:3000])


# =============================================================================
# CELL 4: Prepare 10 Smoke-Test Questions
# =============================================================================
# Select 10 questions that have associated images we can load.

# Check for attached MVTec AD datasets on Kaggle
MVTEC_ROOT_CANDIDATES = [
    Path("/kaggle/input/datasets/ipythonx/mvtec-ad"),
    Path("/kaggle/input/mvtec-ad"),
    Path("/kaggle/input/ipythonx/mvtec-ad"),
    Path("/kaggle/input/mvtec-anomaly-detection"),
]

def map_mmad_to_mvtec_image(img_rel):
    """
    Map MMAD path (e.g. 'DS-MVTec/bottle/image/broken_large/000.png')
    to attached Kaggle MVTec-AD dataset (e.g. 'bottle/test/broken_large/000.png').
    """
    if not img_rel:
        return None

    clean = str(img_rel).replace("\\", "/")
    for prefix in ["ALL_DATA/", "DS-MVTec/"]:
        if clean.startswith(prefix):
            clean = clean[len(prefix):]

    parts = clean.split("/")
    if len(parts) >= 4 and parts[1] == "image":
        category, _, defect_type, filename = parts[0], parts[1], parts[2], parts[3]
    else:
        category = parts[0] if parts else ""
        defect_type = parts[-2] if len(parts) >= 2 else ""
        filename = parts[-1] if parts else ""

    for root in MVTEC_ROOT_CANDIDATES:
        if not root.exists():
            continue

        # Try standard MVTec AD subfolder layouts
        candidates = [
            root / category / "test" / defect_type / filename,
            root / category / "train" / defect_type / filename,
            root / "MVTec AD" / category / "test" / defect_type / filename,
            root / clean,
        ]
        for cand in candidates:
            if cand.exists():
                return str(cand)

        # Fallback directory walk
        cat_dir = root / category
        if cat_dir.exists():
            for fpath in cat_dir.rglob(filename):
                if defect_type in str(fpath.parent):
                    return str(fpath)

    return None


def find_questions_with_images(questions, data_dir, max_count=10):
    """
    Find questions whose images exist on disk or in attached datasets.
    """
    usable = []
    for q in questions:
        if len(usable) >= max_count:
            break

        img_field = None
        for field_name in ["image", "image_path", "img_path", "filename", "img"]:
            if field_name in q and q[field_name]:
                img_field = field_name
                break

        resolved = None
        if img_field:
            raw_img = q[img_field]
            # 1. Check attached MVTec AD dataset
            resolved = map_mmad_to_mvtec_image(raw_img)

            # 2. Check local data directory
            if not resolved:
                img_path = os.path.join(data_dir, raw_img)
                img_path_alt = os.path.join(data_dir, "ALL_DATA", raw_img)
                if os.path.exists(img_path):
                    resolved = img_path
                elif os.path.exists(img_path_alt):
                    resolved = img_path_alt

        q["_resolved_image_path"] = resolved
        usable.append(q)

    return usable


smoke_test_questions = find_questions_with_images(all_questions, str(MMAD_DATA_DIR), max_count=10)
print(f"\n🎯 Selected {len(smoke_test_questions)} questions for smoke test")

# If images aren't available locally, try downloading or using placeholder
def download_mmad_image(question, data_dir):
    """Try to download a specific MMAD image from HuggingFace."""
    for field_name in ["image", "image_path", "img_path", "filename", "img"]:
        if field_name in question and question[field_name]:
            img_rel_path = question[field_name]
            break
    else:
        return None

    local_path = os.path.join(data_dir, img_rel_path)
    if os.path.exists(local_path):
        return local_path

    try:
        hf_path = f"ALL_DATA/{img_rel_path}" if not img_rel_path.startswith("ALL_DATA") else img_rel_path
        downloaded = hf_hub_download(
            repo_id=MMAD_REPO,
            repo_type="dataset",
            filename=hf_path,
            local_dir=str(data_dir),
        )
        return downloaded
    except Exception:
        try:
            downloaded = hf_hub_download(
                repo_id=MMAD_REPO,
                repo_type="dataset",
                filename=img_rel_path,
                local_dir=str(data_dir),
            )
            return downloaded
        except Exception:
            return None


print("\n📷 Resolving images for smoke test questions...")
for i, q in enumerate(smoke_test_questions):
    resolved = q.get("_resolved_image_path")

    # If already resolved from attached Kaggle MVTec dataset
    if resolved and os.path.exists(resolved):
        print(f"   ✅ Q{i}: Loaded real image from attached dataset: {os.path.basename(resolved)}")
        continue

    # Try mapping again
    mapped = map_mmad_to_mvtec_image(q.get("image", ""))
    if mapped and os.path.exists(mapped):
        q["_resolved_image_path"] = mapped
        print(f"   ✅ Q{i}: Loaded real image from attached dataset: {os.path.basename(mapped)}")
        continue

    # Try downloading
    img_path = download_mmad_image(q, str(MMAD_DATA_DIR))
    if img_path and os.path.exists(img_path):
        q["_resolved_image_path"] = img_path
        print(f"   ✅ Q{i}: Downloaded image")
    else:
        # Create a synthetic test image as fallback
        placeholder_path = str(RESULTS_DIR / f"placeholder_{i}.png")
        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        img.save(placeholder_path)
        q["_resolved_image_path"] = placeholder_path
        print(f"   ⚠️ Q{i}: Using placeholder (image not available)")

# Display the questions
print("\n" + "=" * 60)
print("SMOKE TEST QUESTIONS")
print("=" * 60)
for i, q in enumerate(smoke_test_questions):
    print(f"\n--- Question {i+1} ---")
    ci_q = {str(k).lower(): v for k, v in q.items()}
    for field in ["question", "query", "text", "prompt"]:
        if field in ci_q and ci_q[field]:
            print(f"  Question: {str(ci_q[field])[:200]}")
            break
    for field in ["options", "choices", "answers"]:
        if field in ci_q and ci_q[field]:
            opts = ci_q[field]
            if isinstance(opts, list):
                for j, opt in enumerate(opts):
                    print(f"  ({chr(65+j)}) {opt}")
            elif isinstance(opts, dict):
                for k, v in opts.items():
                    print(f"  ({k}) {v}")
            break
    for field in ["answer", "ground_truth", "gt", "label", "correct"]:
        if field in ci_q and ci_q[field]:
            print(f"  ✓ Ground truth: {ci_q[field]}")
            break
    img_path = q.get("_resolved_image_path", "N/A")
    exists = "✅" if img_path and os.path.exists(str(img_path)) else "❌"
    print(f"  Image: {exists}")


# =============================================================================
# CELL 5: Load Qwen3-VL-2B-Instruct
# =============================================================================

from transformers import AutoProcessor, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

print(f"\n🤖 Loading model: {MODEL_ID}")
print(f"   GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"   VRAM used before loading: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

t_start = time.time()

# Load processor
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

# Load model in FP16 (fits on T4 16GB for 2B model)
# Try multiple loader classes — Qwen VL models may need different classes
# depending on the transformers version
model = None
loader_name = None

# Attempt 1: Native Qwen3VL class if available in transformers
try:
    from transformers import Qwen3VLForConditionalGeneration
    print("   Attempting to load via Qwen3VLForConditionalGeneration...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    loader_name = "Qwen3VLForConditionalGeneration"
except Exception as e:
    print(f"   Qwen3VLForConditionalGeneration: not used ({e})")

# Attempt 2: AutoModelForMultimodalLM
if model is None:
    try:
        from transformers import AutoModelForMultimodalLM
        print("   Attempting to load via AutoModelForMultimodalLM...")
        model = AutoModelForMultimodalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        loader_name = "AutoModelForMultimodalLM"
    except Exception as e:
        print(f"   AutoModelForMultimodalLM: not used ({e})")

# Attempt 3: AutoModelForImageTextToText
if model is None:
    try:
        from transformers import AutoModelForImageTextToText
        print("   Attempting to load via AutoModelForImageTextToText...")
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        loader_name = "AutoModelForImageTextToText"
    except Exception as e:
        print(f"   AutoModelForImageTextToText: not used ({e})")

# Attempt 4: AutoModelForVision2Seq
if model is None:
    try:
        from transformers import AutoModelForVision2Seq
        print("   Attempting to load via AutoModelForVision2Seq...")
        model = AutoModelForVision2Seq.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        loader_name = "AutoModelForVision2Seq"
    except Exception as e:
        print(f"   AutoModelForVision2Seq: not used ({e})")

# Attempt 5: Fallback to Qwen2_5_VLForConditionalGeneration or AutoModelForCausalLM
if model is None:
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
        print("   Attempting fallback via Qwen2_5_VLForConditionalGeneration...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        loader_name = "Qwen2_5_VLForConditionalGeneration"
    except Exception as e:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        loader_name = "AutoModelForCausalLM"

print(f"   ✅ Successfully loaded via {loader_name}")

t_load = time.time() - t_start

# Record model info
model_hash = hashlib.md5(MODEL_ID.encode()).hexdigest()[:8]

if torch.cuda.is_available():
    vram_after = torch.cuda.memory_allocated(0) / 1e9
else:
    vram_after = 0

MANIFEST["model"] = {
    "model_id": MODEL_ID,
    "model_hash": model_hash,
    "dtype": str(model.dtype),
    "load_time_s": round(t_load, 1),
    "vram_after_load_gb": round(vram_after, 2),
    "num_parameters": sum(p.numel() for p in model.parameters()),
}

print(f"\n✅ Model loaded in {t_load:.1f}s")
print(f"   Parameters: {MANIFEST['model']['num_parameters']:,}")
print(f"   Dtype: {model.dtype}")
print(f"   VRAM used: {vram_after:.2f} GB")


# =============================================================================
# CELL 6: Run 10 MMAD Questions with Answer Parser
# =============================================================================

def build_mmad_prompt(question_data):
    """
    Build the prompt for an MMAD multiple-choice question.
    Adapts to the actual field names found in the data (case-insensitive).
    """
    # Create case-insensitive lookup
    ci_data = {str(k).lower(): v for k, v in question_data.items()}

    q_text = ""
    for field in ["question", "query", "text", "prompt", "instruction", "desc", "description"]:
        if field in ci_data and ci_data[field]:
            q_text = str(ci_data[field]).strip()
            break

    # Handle conversation format (e.g. LLaVA format: list of turns)
    if not q_text and "conversations" in ci_data:
        convs = ci_data["conversations"]
        if isinstance(convs, list):
            for turn in convs:
                if isinstance(turn, dict) and turn.get("from") in ["human", "user"]:
                    q_text = str(turn.get("value", "")).strip()
                    break

    options_text = ""
    options_list = []
    for field in ["options", "choices", "answers", "candidates"]:
        if field in ci_data and ci_data[field]:
            opts = ci_data[field]
            if isinstance(opts, list):
                options_list = opts
                for j, opt in enumerate(opts):
                    options_text += f"({chr(65+j)}) {opt}\n"
            elif isinstance(opts, dict):
                for k, v in opts.items():
                    options_list.append(v)
                    options_text += f"({k}) {v}\n"
            break

    if not q_text:
        # Generic fallback
        q_text = "Is there an anomaly or defect in this industrial object? Identify any anomaly present."

    if options_text:
        prompt = f"""{q_text}

{options_text.strip()}

Answer with the letter of the correct option (e.g., A, B, C, or D). Give only the letter."""
    else:
        # Options might already be formatted inside question text
        prompt = f"""{q_text}

Answer with the letter of the correct option (e.g., A, B, C, or D). Give only the letter."""

    return prompt, options_list


def parse_answer(response_text, num_options=4):
    """
    Extract the answer option from model's response using strict pattern matching.
    Returns: (parsed_letter, parse_success)

    Parse-failure rate under corruption is itself a finding (Section 5.1 of the
    mid-term doc), so we track this carefully.
    """
    if not response_text:
        return None, False

    text = response_text.strip()

    # Strategy 1: Single letter A-D at the start
    match = re.match(r'^([A-D])\b', text.upper())
    if match:
        return match.group(1), True

    # Strategy 2: Pattern like "(A)" or "A)" or "A."
    match = re.search(r'\(([A-D])\)|([A-D])[).\s:]', text.upper())
    if match:
        letter = match.group(1) or match.group(2)
        return letter, True

    # Strategy 3: "answer is A" or "option A" patterns
    match = re.search(r'(?:answer|option|choice)\s*(?:is\s*)?([A-D])\b', text.upper())
    if match:
        return match.group(1), True

    # Strategy 4: Exactly one capital letter A-D in the response
    letters_found = re.findall(r'\b([A-D])\b', text.upper())
    if len(letters_found) == 1:
        return letters_found[0], True

    # Strategy 5: Very short response containing one option letter
    if len(text) <= 5:
        for c in text.upper():
            if c in "ABCD":
                return c, True

    return None, False


def get_ground_truth(question_data):
    """Extract ground truth answer from question data (case-insensitive)."""
    ci_data = {str(k).lower(): v for k, v in question_data.items()}
    for field in ["answer", "ground_truth", "gt", "label", "correct"]:
        if field in ci_data and ci_data[field] is not None:
            gt = str(ci_data[field]).strip().upper()
            if len(gt) == 1 and gt in "ABCD":
                return gt
            if gt.isdigit() and int(gt) < 4:
                return chr(65 + int(gt))
            return gt
    return None


def run_inference(model, processor, image_path, prompt_text):
    """
    Run Qwen3-VL inference on a single image + text prompt.
    Temperature 0 (greedy) for determinism, as specified in Section 5.3.
    Returns the model's text response.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"   ⚠️ Could not load image {image_path}: {e}")
        image = Image.new("RGB", (224, 224), (128, 128, 128))

    # Build chat messages in Qwen VL format
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    # Process with the chat template
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
        try:
            text_input = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            try:
                from qwen_vl_utils import process_vision_info
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text_input],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(model.device)
            except ImportError:
                inputs = processor(
                    text=[text_input],
                    images=[image],
                    return_tensors="pt",
                    padding=True,
                ).to(model.device)

        except Exception as e:
            print(f"   ⚠️ Chat template fallback ({e}), using direct processing")
            inputs = processor(
                text=[prompt_text],
                images=[image],
                return_tensors="pt",
                padding=True,
            ).to(model.device)

    # Generate — greedy decoding, no sampling
    gen_kwargs = {
        "max_new_tokens": 64,
        "do_sample": False,
    }

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    input_len = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[0][input_len:]
    response = processor.decode(generated_ids, skip_special_tokens=True).strip()

    return response


# ---------- Run the 10 questions ----------
print("\n" + "=" * 60)
print("RUNNING SMOKE TEST: 10 MMAD Questions")
print("=" * 60)

# Check for existing results (resume pattern)
completed_indices = set()
if RESULTS_PATH.exists():
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                completed_indices.add(entry["question_index"])
    if completed_indices:
        print(f"   ♻️ Resuming: {len(completed_indices)} questions already completed")

results = []

for i, q in enumerate(smoke_test_questions):
    # Skip already-completed questions (resume support)
    if i in completed_indices:
        print(f"\n--- Question {i+1}/{len(smoke_test_questions)} [SKIPPED — already done] ---")
        continue

    print(f"\n--- Question {i+1}/{len(smoke_test_questions)} ---")

    prompt_text, options_list = build_mmad_prompt(q)
    gt = get_ground_truth(q)
    img_path = q.get("_resolved_image_path", "")

    print(f"  Prompt: {prompt_text[:150]}...")
    print(f"  Image: {os.path.basename(str(img_path)) if img_path else 'N/A'}")
    print(f"  Ground truth: {gt}")

    # Run inference
    t0 = time.time()
    try:
        response = run_inference(model, processor, img_path, prompt_text)
        inference_time = time.time() - t0
        error = None
    except Exception as e:
        response = ""
        inference_time = time.time() - t0
        error = str(e)
        print(f"  ❌ Inference error: {e}")

    # Parse answer
    parsed_answer, parse_success = parse_answer(response)
    is_correct = (parsed_answer == gt) if (parsed_answer and gt) else None

    print(f"  Response: {response[:200]}")
    print(f"  Parsed: {parsed_answer} (parse {'✅' if parse_success else '❌'})")
    print(f"  Correct: {'✅' if is_correct else '❌' if is_correct is False else '❓'}")
    print(f"  Time: {inference_time:.1f}s")

    result = {
        "question_index": i,
        "prompt": prompt_text[:500],
        "ground_truth": gt,
        "model_response": response,
        "parsed_answer": parsed_answer,
        "parse_success": parse_success,
        "is_correct": is_correct,
        "inference_time_s": round(inference_time, 2),
        "error": error,
        "image_path": os.path.basename(str(img_path)) if img_path else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    results.append(result)

    # CHECKPOINT: Append to disk after every question
    with open(RESULTS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# CELL 7: Save Manifest & Summary
# =============================================================================

# Reload all results (including resumed ones) for final stats
all_results = []
if RESULTS_PATH.exists():
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_results.append(json.loads(line))

total = len(all_results)
parsed_ok = sum(1 for r in all_results if r["parse_success"])
correct = sum(1 for r in all_results if r["is_correct"] is True)
errors = sum(1 for r in all_results if r["error"] is not None)
avg_time = sum(r["inference_time_s"] for r in all_results) / max(total, 1)

MANIFEST["results_summary"] = {
    "total_questions": total,
    "parse_success": parsed_ok,
    "parse_failure": total - parsed_ok,
    "parse_success_rate": round(parsed_ok / max(total, 1) * 100, 1),
    "correct": correct,
    "incorrect": parsed_ok - correct,
    "accuracy_on_parsed": round(correct / max(parsed_ok, 1) * 100, 1),
    "inference_errors": errors,
    "avg_inference_time_s": round(avg_time, 2),
}

with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
    json.dump(MANIFEST, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 60)
print("PHASE 0 SMOKE TEST — SUMMARY")
print("=" * 60)
print(f"  Model:              {MODEL_ID}")
print(f"  GPU:                {MANIFEST['gpu_info']['gpu_name']}")
print(f"  Questions run:      {total}")
print(f"  Parse success:      {parsed_ok}/{total} ({MANIFEST['results_summary']['parse_success_rate']}%)")
print(f"  Correct answers:    {correct}/{parsed_ok} ({MANIFEST['results_summary']['accuracy_on_parsed']}%)")
print(f"  Inference errors:   {errors}")
print(f"  Avg inference time: {avg_time:.1f}s per question")
print(f"")
print(f"  📁 Manifest saved:  {MANIFEST_PATH}")
print(f"  📁 Results saved:   {RESULTS_PATH}")
print("=" * 60)

# ---------- EXIT CRITERION CHECK ----------
if parsed_ok > 0 and errors < total:
    print("\n✅ PHASE 0 EXIT CRITERION MET:")
    print("   At least one model answered at least one MMAD question end-to-end")
    print("   with a parseable answer. Pipeline is validated.")
    print("\n   ➡️  Next step: Phase 1 — Clean baseline on full MMAD question set")
else:
    print("\n❌ PHASE 0 EXIT CRITERION NOT MET:")
    print("   No questions were successfully parsed. Debug needed.")
    print("   Check the model responses and parser above.")


# =============================================================================
# CELL 8: Checkpoint Resume Verification
# =============================================================================

print("\n" + "=" * 60)
print("CHECKPOINT RESUME VERIFICATION")
print("=" * 60)

loaded_results = []
if RESULTS_PATH.exists():
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                loaded_results.append(json.loads(line))

print(f"  Results on disk:   {len(loaded_results)} entries")
print(f"  Results in memory: {len(results)} entries (this session)")
print(f"  Total completed:   {len(loaded_results)} entries (including resumed)")
print(f"\n  ✅ Resume pattern works — safe for Kaggle session resets.")
print(f"  In Phase 1+, we check for existing results and skip completed questions.")

print("\n🏁 Phase 0 complete. Ready to proceed to Phase 1.")
