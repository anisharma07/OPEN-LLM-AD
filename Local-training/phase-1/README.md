# Phase 1: Clean Baseline Benchmark Evaluation (MMAD)

## 🎯 Purpose & Scope
Phase 1 establishes the **Clean Benchmark Baseline** for the M.Tech Dissertation:
**"Robustness and Reproducibility of Open-Source Small Multimodal LLMs for Industrial Anomaly Detection"**.

Before measuring degradation under real-world factory corruptions (Phase 2), we must accurately map out the model's clean uncorrupted capabilities, strengths, and failure boundaries across the entire taxonomy of industrial inspection tasks.

---

## 💻 Hardware & Inference Configuration
- **Model**: `Qwen/Qwen3-VL-2B-Instruct` (FP16) via `Qwen3VLForConditionalGeneration`
- **Target GPU**: NVIDIA GeForce RTX 4060 Laptop GPU (8.19 GB VRAM)
- **VRAM Footprint**: **4.26 GB / 8.19 GB** (~52% utilization)
- **Inference Mode**: Deterministic Greedy Decoding (`temperature=0.0`, `seed=42`, `do_sample=False`)
- **Optimization**: `torch.inference_mode()` + `image.thumbnail((512, 512))` + `max_new_tokens=8`
- **Average Latency**: **0.16s per question** (~6.5 minutes for 2,500 questions)

---

## 📊 Benchmark Scale & Results Summary
- **Questions Evaluated**: **2,500 questions** (Stratified across 9 subtasks & 38 product categories)
- **Overall Accuracy**: **71.2%** (1,780 / 2,500 correct)
- **Cohen's Kappa ($\kappa$)**: **0.615** (Substantial Agreement beyond chance)
- **Regex Parse Success Rate**: **100.0%** (2,500 / 2,500 extractable)

---

## 📦 Multi-Dataset Coverage (All 4 MMAD Datasets)

| Dataset | Questions Evaluated | Correct | Accuracy (%) | Key Finding |
| :--- | :---: | :---: | :---: | :--- |
| **DS-MVTec (MVTec-AD)** | 497 | 385 | **77.5%** | Classic surface defects (scratches, cracks, cuts) |
| **VisA (Visual Anomaly)** | 787 | 587 | **74.6%** | High-density electronics (PCBs, chips, capsules) |
| **GoodsAD** | 716 | 500 | **69.8%** | Packaging tears, deformed cans, open food boxes |
| **MVTec-LOCO** | 500 | 308 | **61.6%** | ⚠️ **Hardest Dataset!** Logical & structural layout errors |

---

## 🔬 Subtask Accuracy Breakdown (RQ1 Core Contribution)

| Subtask Name | Correct | Total | Accuracy (%) | Capability Tier |
| :--- | :---: | :---: | :---: | :--- |
| **Object Classification** | 263 | 277 | **94.9%** | Near-perfect category understanding |
| **Object Analysis** | 228 | 277 | **82.3%** | High visual reasoning |
| **Defect Analysis** | 228 | 278 | **82.0%** | Strong macro anomaly identification |
| **Object Structure** | 225 | 277 | **81.2%** | Spatial component layout |
| **Object Details** | 221 | 277 | **79.8%** | Texture and fine details |
| **Defect Description** | 197 | 277 | **71.1%** | Semantic defect properties |
| **Anomaly Detection** | 173 | 282 | **61.3%** | Binary defect presence/absence detection |
| **Defect Localization** | 132 | 278 | **47.5%** | Spatial defect coordinates / quadrant reasoning |
| **Defect Classification** | 113 | 277 | **40.8%** | ⚠️ **Primary Failure Mode!** (Distinguishing defect types) |

---

## 🖼️ Publication Visual Artifacts in `results/`
1. **`phase1_dataset_breakdown.png`**: Multi-dataset comparative bar chart (MVTec-AD vs GoodsAD vs VisA vs MVTec-LOCO).
2. **`phase1_sample_predictions_grid.png`**: 4x4 high-resolution visual inspection grid (16 real industrial images) with color-coded status banners.
3. **`phase1_error_analysis_gallery.png`**: 3x3 diagnostic failure gallery (9 real defect images) highlighting why the model missed or misclassified defects.
4. **`phase1_subtask_accuracy.png`**: Dual horizontal publication bar charts across all 9 subtasks and top 25 product categories.
5. **`phase1_category_subtask_heatmap.png`**: 2D performance heatmap showing subtask vs category nuances.
6. **`phase1_confusion_analysis.png`**: Anomaly detection confusion matrix (TP, FP, TN, FN) and latency histogram on RTX 4060 GPU.
7. **`results.txt`**: Complete human-readable cards for evaluated questions with full QA details, GT, model prediction, and comparison matrix.
8. **`phase1_manifest.json`**: Machine-readable execution manifest.
9. **`phase1_results.jsonl`**: Append-safe evaluation checkpoints.

---

## 🚀 How This Connects to Phase 2
Phase 1 established that the uncorrupted model achieves **71.2% accuracy** (baseline $A_{\text{clean}}$).
In **Phase 2**, we apply 7 real-world industrial corruptions across 5 severity levels to measure:
- **Robustness Degradation Slope (RDS)**: Rate of performance collapse per severity unit.
- **Critical Failure Boundaries (CFB)**: The exact severity point where the inspection system becomes unreliable.
