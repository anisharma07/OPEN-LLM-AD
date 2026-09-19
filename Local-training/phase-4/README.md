# Phase 4: Cross-Model Multimodal LLM Architecture Benchmark

## 🎯 Purpose & Scope
Phase 4 expands the dissertation's scope from a single-model investigation to a **Multi-Architecture Comparative Benchmark**:
**"Robustness and Reproducibility of Open-Source Small Multimodal LLMs for Industrial Anomaly Detection"**.

While Phase 1, Phase 2, and Phase 3 thoroughly established the baseline, corruption vulnerability, and test-time mitigations on `Qwen3-VL-2B-Instruct`, Phase 4 addresses **RQ1 & Architectural Generalizability**:
- *How do different foundational vision-language architectures perform on industrial anomaly detection?*
- *Does scaling from 2B to 4B parameters yield proportional gains in fine-grained industrial defect localization and classification?*
- *How does Google's new **Gemma 4** vision architecture compare against Alibaba's **Qwen VL** architecture on industrial machine vision tasks?*

---

## 🗺️ Cross-Phase Context & Research Roadmap

| Phase | Phase Name | Core Scientific Focus | Status |
| :--- | :--- | :--- | :--- |
| **Phase 0** | **System Setup & Smoke Test** | Local GPU environment validation & FP16 initialization | ✅ Completed |
| **Phase 1** | **Clean Baseline Benchmark** | 2,500 questions across 4 datasets (`Qwen3-VL-2B`: 71.20%, $\kappa = 0.615$) | ✅ Completed |
| **Phase 2** | **Industrial Corruption Study** | Stress-testing 7 physical corruptions (Conveyor blur $RDS = +1.78$) | ✅ Completed |
| **Phase 3** | **Adaptation & Mitigation** | TTA-IR edge restoration (+3.43% Localization gain) | ✅ Completed |
| **Phase 4** | **Cross-Model Benchmark (Current)** | Cross-model comparison across **Gemma 4 (2B & 4B)** vs **Qwen (2B & 3B/4B)** on same 2,500 questions | 🚀 **In Progress** |

---

## 🤖 Models Benchmarked (Same 2,500 Stratified Questions)

| # | Model Identifier | Developer | Parameter Scale | Vision Backbone / Architecture | Context Length |
| :---: | :--- | :---: | :---: | :--- | :---: |
| 1 | **`Qwen/Qwen3-VL-2B-Instruct`** | Alibaba | ~2.0 Billion | Native Dynamic Patch ViT | 32K |
| 2 | **`google/gemma-4-E2B-it`** | Google | ~2.3 Billion | Gemma 4 Vision Transformer | 8K |
| 3 | **`google/gemma-4-E4B-it`** | Google | ~4.1 Billion | Gemma 4 Vision Transformer | 8K |
| 4 | **`Qwen/Qwen2.5-VL-3B-Instruct`** | Alibaba | ~3.7 Billion | Window Attention Qwen-VL ViT | 32K |

---

## 📐 Comparative Metrics Computed
1. **Overall Accuracy ($A$)**:
   Percentage of correct answers across the standardized 2,500 questions.
2. **Inter-Rater Agreement (Cohen's Kappa $\kappa$)**:
   $$\kappa = \frac{p_o - p_e}{1 - p_e}$$
3. **Subtask Performance Hierarchy**:
   Comparative accuracy across all 9 industrial subtasks (Defect Localization, Defect Classification, Logical Anomaly, etc.).
4. **Hardware Efficiency Trade-off**:
   - VRAM Memory Footprint (GB) on NVIDIA RTX 4060.
   - Inference Latency (sec / question) and Throughput (FPS).

---

## 📁 Output Artifacts in `results/`
- **`phase4_cross_model_accuracy_comparison.png`**: High-resolution bar chart comparing accuracy across all evaluated models.
- **`phase4_model_subtask_radar.png`**: Polar radar chart comparing subtask strengths of Gemma 4 vs Qwen.
- **`phase4_latency_accuracy_pareto.png`**: Pareto frontier plotting Accuracy vs Inference Latency.
- **`results.txt`**: Detailed formatted comparison log with per-model summary cards and subtask rankings.
- **`phase4_manifest.json`**: Machine-readable JSON summary.
