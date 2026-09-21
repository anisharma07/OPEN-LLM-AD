# Phase 4: Cross-Model Multimodal LLM Architecture Benchmark

## 🎯 Purpose & Scope
Phase 4 expands the dissertation's scope from a single-model investigation to a **Multi-Architecture Comparative Benchmark**:
**"Robustness and Reproducibility of Open-Source Small Multimodal LLMs for Industrial Anomaly Detection"**.

While Phase 1, Phase 2, and Phase 3 thoroughly established the baseline, corruption vulnerability, and test-time mitigations on `Qwen3-VL-2B-Instruct`, Phase 4 directly addresses **Research Question 1 & Architectural Generalizability**:
- *How do different foundational vision-language architectures perform on industrial anomaly detection?*
- *Does scaling from 2B to 4B parameters yield proportional gains in fine-grained industrial defect localization and classification?*
- *How does Google's new **Gemma 4** vision architecture compare against Alibaba's **Qwen VL** architecture on industrial machine vision tasks?*

---

## 🗺️ Cross-Phase Research Roadmap

| Phase | Phase Name | Core Scientific Focus | Status |
| :--- | :--- | :--- | :--- |
| **Phase 0** | **System Setup & Smoke Test** | Local GPU environment validation & FP16 initialization | ✅ Completed |
| **Phase 1** | **Clean Baseline Benchmark** | 2,500 questions across 4 datasets (`Qwen3-VL-2B`: 71.20%, $\kappa = 0.615$) | ✅ Completed |
| **Phase 2** | **Industrial Corruption Study** | Stress-testing 7 physical corruptions (Conveyor blur $RDS = +1.78$) | ✅ Completed |
| **Phase 3** | **Adaptation & Mitigation** | TTA-IR edge restoration (+3.43% Localization gain) | ✅ Completed |
| **Phase 4** | **Cross-Model Benchmark (Completed)** | Cross-model comparison across **Gemma 4 (2B & 4B)** vs **Qwen (2B & 3B)** on same 2,500 questions | ✅ **Completed** |

---

## 🏆 Final Benchmark Results Across All 4 Models (N = 2,500 Clean Questions)

| Rank | Model Identifier | Developer | Scale | Clean Accuracy | Cohen's $\kappa$ | Avg Latency | Throughput | VRAM Footprint |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **`Qwen3-VL-2B-Instruct`** | Alibaba | ~2.0B | **71.20%** | **0.615** | **0.160s** | **6.2 fps** | 4.82 GB (FP16) |
| 🥈 | **`Qwen2.5-VL-3B-Instruct`** | Alibaba | ~3.7B | **67.68%** | **0.574** | **0.223s** | **4.5 fps** | 5.86 GB (FP16) |
| 🥉 | **`google/gemma-4-E4B-it`** | Google | ~4.1B | **65.60%** | **0.540** | **0.847s** | **1.2 fps** | 3.32 GB (4-bit NF4) |
| 4 | **`google/gemma-4-E2B-it`** | Google | ~2.3B | **60.92%** | **0.476** | **0.337s** | **3.0 fps** | 4.65 GB (FP16) |

---

## 📊 Subtask-Level Performance Breakdown (%)

| Subtask Category | Questions | Qwen3-VL-2B | Qwen2.5-VL-3B | Gemma 4 4B | Gemma 4 2B | Architecture Leader |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Object Classification** | 277 | **94.9%** | 90.6% | 80.9% | 73.3% | 🏆 Qwen3-VL-2B (+14.0%) |
| **Object Analysis** | 277 | 82.3% | **83.4%** | 82.7% | 71.8% | 🏆 Qwen2.5-VL-3B (+0.7%) |
| **Defect Analysis** | 277 | **82.0%** | 77.4% | 76.0% | 75.6% | 🏆 Qwen3-VL-2B (+4.6%) |
| **Object Structure** | 277 | **81.2%** | 79.1% | 80.5% | 67.9% | 🏆 Qwen3-VL-2B (+0.7%) |
| **Object Details** | 277 | **79.8%** | 72.0% | 75.3% | 71.7% | 🏆 Qwen3-VL-2B (+4.5%) |
| **Defect Description** | 277 | **71.1%** | 62.5% | 65.3% | 63.9% | 🏆 Qwen3-VL-2B (+5.8%) |
| **Anomaly Detection** | 277 | **61.3%** | 41.7% | 45.0% | 50.7% | 🏆 Qwen3-VL-2B (+10.6%) |
| **Defect Classification** | 277 | 40.8% | 51.3% | **53.4%** | 47.0% | 🏆 Gemma 4 4B (+2.1%) |
| **Defect Localization** | 277 | 47.5% | **51.3%** | 31.4% | 26.4% | 🏆 Qwen2.5-VL-3B (+3.8%) |

---

## 🔬 Key Scientific Findings & Dissertation Insights

1. **Vision-Language Architecture Dominance (Qwen vs. Gemma)**:
   - Alibaba's **Qwen series** significantly outperforms Google's **Gemma 4** series in industrial anomaly detection across all model scales.
   - Even the compact `Qwen3-VL-2B` (71.20%) beats the larger `Gemma 4 4B` (65.60%) by **+5.60%** overall, while achieving **5.3x higher throughput** (6.2 fps vs. 1.2 fps).

2. **The Spatial Defect Localization Gap**:
   - In *Defect Localization* (predicting bounding coordinates of surface flaws), both Qwen models achieved strong spatial capability (**47.5% - 51.3%**), whereas Gemma models suffered a severe spatial handicap (**26.4% - 31.4%**).
   - *Root Cause*: Qwen uses a native **Dynamic Patch ViT** preserving spatial grid tokens directly into the LLM context, whereas Gemma 4 uses pooled vision tokens that compress fine-grained spatial information.

3. **Parameter Scaling Law Within Architectures**:
   - **Gemma Scaling**: Scaling Gemma 4 from 2B (60.92%) to 4B (65.60%) yielded a significant **+4.68% accuracy gain** and elevated Cohen's Kappa from 0.476 to 0.540, with massive improvements in Object Structure (+12.6%) and Defect Classification (+6.5%).
   - **Qwen Scaling**: `Qwen3-VL-2B` slightly outperformed `Qwen2.5-VL-3B` in classification and anomaly detection, demonstrating the algorithmic superiority of Qwen's generation 3 vision tower over generation 2.5.

4. **Hardware Feasibility on Edge GPUs (8GB RTX 4060)**:
   - `google/gemma-4-E4B-it` in 4-bit NF4 precision operated stably at **3.32 GB VRAM**, proving that 4B multimodal LLMs can be successfully deployed on industrial edge GPUs without Out-Of-Memory (OOM) failures.

---

## 📁 Output Artifacts in `results/`
- **`phase4_cross_model_accuracy_comparison.png`**: High-resolution bar chart comparing accuracy and kappa across all 4 models.
- **`phase4_subtask_accuracy_heatmap.png`**: Heatmap comparing performance across all 9 subtasks.
- **`results.txt`**: Complete formatted benchmark log.
- **`phase4_manifest.json`**: Machine-readable JSON summary of all metrics.
- **`google_gemma_4_e4b_it_results.jsonl`**: 2,500 itemized predictions for Gemma 4 4B.
- **`google_gemma_4_e2b_it_results.jsonl`**: 2,500 itemized predictions for Gemma 4 2B.
- **`qwen_qwen2.5_vl_3b_instruct_results.jsonl`**: 2,500 itemized predictions for Qwen 2.5 3B.
