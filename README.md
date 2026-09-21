# Open-IAD: Robustness and Reproducibility of Open-Source Small Multimodal LLMs for Industrial Anomaly Detection

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0%2Bcu124-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![Hardware](https://img.shields.io/badge/Hardware-RTX%204060%20(8GB)-76B900.svg?logo=nvidia)](https://nvidia.com)
[![Benchmark](https://img.shields.io/badge/Benchmark-MMAD%20(ICLR%202025)-FF6F00.svg)](https://arxiv.org/abs/2410.09453)
[![Evaluations](https://img.shields.io/badge/Evaluations-62%2C500%20Inferences-success.svg)]()

> **M.Tech Dissertation Project**  
> **Author:** Anirudh Sharma  
> **Master Manuscript:** [`Robustness_and_Reproducibility_of_Open_Source_Small_Multimodal_LLMs_for_Industrial_Anomaly_Detection_Final_Thesis_Draft.md`](Robustness_and_Reproducibility_of_Open_Source_Small_Multimodal_LLMs_for_Industrial_Anomaly_Detection_Final_Thesis_Draft.md)

---

## 📌 Executive Summary

This repository hosts the experimental code, model dispatch engines, evaluation manifests, and results for the M.Tech Dissertation: **"Robustness and Reproducibility of Open-Source Small Multimodal LLMs for Industrial Anomaly Detection"**. 

While closed commercial models (e.g. GPT-4o) report headline scores approaching 75% on curated benchmarks, they cannot be deployed on air-gapped factory inspection lines due to data confidentiality, latency, and operational costs. Conversely, small open-weight multimodal models (2B–4B) can run on-premise, but their stability under physical factory environmental corruptions and memory-constrained edge quantization has never been systematically benchmarked.

Across **62,500 model inferences** on the canonical **MMAD** benchmark using a single consumer **NVIDIA GeForce RTX 4060 (8 GB VRAM)**, we investigate:
1. **Clean Laboratory Baseline (Phase 1, N=2,500):** Establishing reference zero-shot accuracy across 9 inspection tasks (**71.20%** on `Qwen3-VL-2B`).
2. **Industrial Corruption Degradation (Phase 2, N=5,000, 25,000 runs):** Measuring performance collapse under Conveyor Motion Blur, Gaussian Sensor Noise, Defocus Blur, and Low-Light Shift ($RDS = +1.78$ for Motion Blur; Object Classification drops by $-19.39\%$).
3. **Test-Time Augmentation & Restoration (TTA-IR, Phase 3, N=5,000, 25,000 runs):** Applying zero-parameter high-boost Laplacian unsharp masking and bilateral filtering, recovering fine-grained Defect Localization by **+3.43%**.
4. **Cross-Model Architectural Benchmark (Phase 4, 10,000 runs):** Head-to-head comparison across 4 open architectures (`Qwen3-VL-2B`, `Qwen2.5-VL-3B`, `google/gemma-4-E2B-it`, and `google/gemma-4-E4B-it`), including a breakthrough 4-bit NF4 + host-RAM embedding hybrid routing pipeline that enables 16 GB Gemma 4 4B to run on 8 GB GPUs at only **3.32 GB peak VRAM**.

---

## 📊 Cross-Model Benchmark Results (Phase 4, N=2,500 Questions / Model)

| Model Architecture | Quantization | Accuracy | Cohen's $\kappa$ | Latency / Query | Throughput | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen3-VL-2B-Instruct** | Native FP16 | **71.20%** | **0.615** | **0.160s** | **6.25 FPS** | 4.82 GB |
| **Qwen2.5-VL-3B-Instruct** | Native FP16 | **67.68%** | **0.574** | **0.223s** | **4.48 FPS** | 5.86 GB |
| **google/gemma-4-E4B-it** | **4-bit NF4** | **65.60%** | **0.540** | **0.847s** | **1.18 FPS** | **3.32 GB** |
| **google/gemma-4-E2B-it** | Native FP16 | **60.92%** | **0.476** | **0.337s** | **2.97 FPS** | 4.65 GB |

### Key Architectural Finding: The Spatial Coordinate Token Gap
- **Qwen Dynamic Patch ViT:** Preserves 2D spatial coordinate patch tokens into the language model context, maintaining **47.65% – 51.26%** localization accuracy.
- **Gemma Token Pooling:** Employs spatial token pooling to compress visual inputs for any-to-any multimodal conversation, discarding precise boundary coordinate tokens and resulting in localization collapse (**26.35% – 31.41%**).

---

## 🗂️ Repository Structure & Phase Progression

```
Open-IAD/
├── Robustness_and_Reproducibility_..._Final_Thesis_Draft.md  # Complete thesis manuscript
├── Robustness and Reproducibility of...Mid-Term D.pdf         # Midterm review document
├── MMAD/                                                     # Dataset hierarchy (38 classes)
│   ├── MVTec-AD/
│   ├── VisA/
│   ├── GoodsAD/
│   └── AeBAD/
└── Local-training/
    ├── requirements.txt                                      # Pinned dependencies
    ├── phase-0/                                              # Environment & smoke tests
    │   └── README.md
    ├── phase-1/                                              # Clean baseline (N=2,500)
    │   ├── run_phase1_clean_baseline.py
    │   ├── README.md
    │   └── results/
    ├── phase-2/                                              # Industrial corruptions (N=5,000)
    │   ├── README.md
    │   └── results/
    ├── phase-3/                                              # TTA-IR test-time mitigations (N=5,000)
    │   ├── README.md
    │   └── results/
    └── phase-4/                                              # Cross-model & 4-bit edge zoo
        ├── run_phase4_cross_model_benchmark.py
        ├── README.md
        └── results/
            ├── google_gemma_4_e4b_it_results.jsonl
            ├── google_gemma_4_e2b_it_results.jsonl
            ├── qwen_qwen2.5_vl_3b_instruct_results.jsonl
            ├── phase4_cross_model_accuracy_comparison.png
            ├── phase4_subtask_accuracy_heatmap.png
            ├── phase4_manifest.json
            └── results.txt
```

---

## 🚀 Quickstart & Reproducibility

### 1. Environment Setup
```bash
git clone https://github.com/anisharma07/OPEN-LLM-AD.git
cd OPEN-LLM-AD/Local-training

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Phase 4 Cross-Model Benchmark
```bash
python3 phase-4/run_phase4_cross_model_benchmark.py
```

### 3. Verify Deterministic Manifests
All evaluation records are written line-by-line to `Local-training/phase-4/results/*.jsonl` with question ID, exact prompt, raw output, parsed choice, ground-truth label, and inference latency.

---

## 📜 Citation
If you utilize this benchmark, codebase, or results in your academic research, please cite:
```bibtex
@mastersthesis{sharma2026robustness,
  title={Robustness and Reproducibility of Open-Source Small Multimodal LLMs for Industrial Anomaly Detection},
  author={Sharma, Anirudh},
  school={Department of Computer Science and Engineering},
  year={2026}
}
```
