# Phase 0: Environment Setup & Smoke Test Validation

## 🎯 Purpose & Scope
Phase 0 serves as the operational baseline and pipeline validation phase for the M.Tech Dissertation:
**"Robustness and Reproducibility of Open-Source Small Multimodal LLMs for Industrial Anomaly Detection"**.

The goal was to ensure end-to-end correctness of:
1. Local hardware setup (CUDA 12.4, NVIDIA GeForce RTX 4060 Laptop GPU).
2. Small Multimodal LLM loading (`Qwen/Qwen3-VL-2B-Instruct` in FP16 via `Qwen3VLForConditionalGeneration`).
3. Parsing complex MMAD benchmark JSON structure (8,366 image entries across 4 datasets).
4. Image loading and local path resolution without placeholders or missing files.
5. Deterministic greedy decoding (`temperature=0.0`, `seed=42`) and regex answer extraction.
6. Append-safe JSONL checkpointing and formatted report generation.

---

## 💻 Hardware & Environment Manifest
- **GPU**: NVIDIA GeForce RTX 4060 Laptop GPU (8.19 GB VRAM)
- **CUDA / Driver**: CUDA 12.4 / Driver 595.84.07
- **PyTorch**: 2.6.0+cu124
- **Transformers**: 5.17.0
- **Model Architecture**: `Qwen3VLForConditionalGeneration`
- **VRAM Footprint**: **4.26 GB / 8.19 GB** (~52% utilization, zero OOM risk)
- **Average Inference Latency**: **0.42s per question**

---

## 🔬 What Was Executed & Tested
- Evaluated **10 real industrial test cases** from MMAD `DS-MVTec` (cables, hazelnuts, metal nuts, tiles).
- Tested binary anomaly detection ("Is there any defect?"), defect classification ("What kind of defect is this?"), and surface appearance queries.
- Answer extraction regex achieved **10/10 (100%) parse rate**.
- Prediction accuracy on real smoke test images: **7/10 (70.0%)**.

---

## 📁 Output Artifacts in `phase-0/results/`
- **`results.txt`**: Human-readable evaluation report with per-question cards (Question, Options, Actual Answer, Generated Answer, Status).
- **`phase0_manifest.json`**: Machine-readable environment and execution manifest.
- **`phase0_results.jsonl`**: Append-safe checkpointed test logs.
- **`phase0_metrics_plot.png`**: Bar charts of parse rate, accuracy, and latency.
- **`phase0_sample_predictions.png`**: Visual grid showing evaluated industrial images with predictions.
- **`phase0_predictions_table.png`**: Visual table image of evaluated questions.

---

## 🚀 How This Connects to Other Phases
- **Phase 0 (Smoke Test)**: Validated pipeline integrity on 10 samples.
- **Phase 1 (Clean Baseline)**: Scaled up to 2,500 questions across 4 datasets to establish the clean performance baseline.
- **Phase 2 (Corruption Study)**: Uses the validated pipeline to apply 7 real-world industrial corruptions across 5 severity levels.
