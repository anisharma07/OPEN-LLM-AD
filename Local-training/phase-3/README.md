# Phase 3: Robustness Adaptation, Mitigation & Recovery

## 🎯 Purpose & Scope
Phase 3 represents the **Engineering & Methodological Intervention** of the M.Tech Dissertation:
**"Robustness and Reproducibility of Open-Source Small Multimodal LLMs for Industrial Anomaly Detection"**.

In Phase 2, empirical stress-testing revealed that factory floor corruptions cause severe degradation:
- **Conveyor Motion Blur** caused catastrophic performance collapse ($RDS = +1.78$, accuracy falling from 77.8% down to 64.4%).
- **Gaussian Sensor Noise** degraded accuracy to 68.9% ($RDS = +0.89$).
- **Defect Localization** collapsed to **34.3%** under severe corruption.

In real-world manufacturing plants, retraining multi-billion parameter foundation models is computationally prohibitive and latency-sensitive edge devices (e.g. NVIDIA Jetson, RTX 4060) require lightweight, plug-and-play solutions.

**Phase 3 designs, implements, and benchmarks edge-deployable Adaptation and Mitigation Mechanisms** to flatten the Robustness Degradation Slope ($\Delta RDS$) and recover lost accuracy without retraining the backbone weights.

---

## 🗺️ Cross-Phase Context & Research Roadmap

| Phase | Phase Name | Focus & Contributions | Status |
| :--- | :--- | :--- | :--- |
| **Phase 0** | **System Setup & Smoke Test** | Validated RTX 4060 GPU, PyTorch 2.6.0+cu124, FP16 loading of `Qwen/Qwen3-VL-2B-Instruct` | ✅ Completed |
| **Phase 1** | **Clean Baseline Benchmark** | Comprehensive evaluation on 2,500 samples across 4 datasets (`DS-MVTec`, `GoodsAD`, `VisA`, `MVTec-LOCO`), achieving $A_{\text{clean}} = 71.2\%$, $\kappa = 0.615$ | ✅ Completed |
| **Phase 2** | **Industrial Corruption Study** | Stress-tested 7 corruptions across 5 severities (1,620 tests). Discovered Motion Blur as primary failure mode ($RDS = +1.78$) and Defect Localization fragility | ✅ Completed |
| **Phase 3** | **Adaptation & Mitigation (Current)**| Deploying Test-Time Image Restoration (TTA-IR), Defect-Anchored Reasoning (DAR), and Hybrid Adaptation. Measuring Robustness Recovery Rate ($RRR$) | 🚀 **In Progress** |
| **Phase 4** | **Edge Deployment & PEFT/LoRA** | Quantization (INT8/INT4), throughput/latency benchmarking, and parameter-efficient LoRA adapter training for hardened industrial deployment | ⏳ Upcoming |

---

## 🛠️ The 3 Adaptation & Mitigation Strategies

```
                       ┌────────────────────────────────────────────────────────┐
                       │  Corrupted Industrial Sensor Image (e.g., Motion Blur)  │
                       └───────────────────────────┬────────────────────────────┘
                                                   │
                ┌──────────────────────────────────┴──────────────────────────────────┐
                ▼                                                                     ▼
   [Strategy 1: Input-Level TTA-IR]                                   [Strategy 2: Prompt-Level DAR]
   Deterministic Edge & Contrast Restoration                          Industrial Noise-Aware Prompt
   • Motion Blur: Laplacian Unsharp Mask                              • Instructs attention heads to
   • Gaussian Noise: Bilateral Filtering                                ignore blur/noise artifacts
   • Low Light: LAB-Space CLAHE                                       • Anchors reasoning to micro-cracks,
                │                                                       structural tears & missing parts
                └──────────────────────────────────┬──────────────────────────────────┘
                                                   │
                                                   ▼
                                     [Strategy 3: Hybrid Adaptation]
                                    Compound Image Restoration + DAR
                                                   │
                                                   ▼
                               ┌───────────────────────────────────────┐
                               │  Qwen3-VL-2B-Instruct Vision Encoder  │
                               └───────────────────┬───────────────────┘
                                                   │
                                                   ▼
                                ┌─────────────────────────────────────┐
                                │ Restored Industrial Anomaly Decision │
                                └─────────────────────────────────────┘
```

### 1. Strategy 1: Input-Level Test-Time Image Restoration (TTA-IR)
Industrial inspection cameras operate in real-time ($<20$ ms budget). We implement deterministic, ultra-fast pre-tokenization restoration filters:
- **Motion Blur Recovery (Unsharp Masking)**:
  $$I_{\text{sharp}} = I + \alpha \cdot (I - I * G_\sigma)$$
  Boosts smeared high-frequency edge gradients before visual patch projection.
- **Gaussian Sensor Noise Recovery (Edge-Preserving Bilateral Filtering)**:
  $$BF[I]_p = \frac{1}{W_p} \sum_{q \in \Omega} I_q \exp\left(-\frac{\|p-q\|^2}{2\sigma_s^2}\right) \exp\left(-\frac{|I_p - I_q|^2}{2\sigma_r^2}\right)$$
  Suppresses sensor thermal grain in homogeneous regions while strictly maintaining sharp anomaly boundary transitions.
- **Low-Light / Underexposure Recovery (CLAHE)**:
  Contrast Limited Adaptive Histogram Equalization applied to the luminance ($L$) channel in $LAB$ color space to enhance dark defect visibility without over-saturating highlights.

### 2. Strategy 2: Prompt-Level Defect-Anchored Reasoning (DAR)
Standard zero-shot prompts assume clear laboratory imagery. The Defect-Anchored Reasoning prompt injects domain constraints directing the attention heads of `Qwen3-VL-2B`:
> *"Disregard global image noise, motion streaks, or lighting shadows. Focus strictly on intrinsic physical anomalies: surface tears, cracks, micro-scratches, structural deformities, contamination spots, missing components, or dimensional discrepancies."*

### 3. Strategy 3: Hybrid Compound Adaptation (TTA + DAR)
Applies test-time high-frequency edge restoration followed by defect-anchored reasoning prompt for compound recovery.

---

## 📐 Mathematical Evaluation Metrics

### 1. Robustness Recovery Rate (RRR)
Measures the percentage of accuracy loss induced by corruption that is successfully recovered by the mitigation strategy:
$$RRR = \frac{A_{\text{mitigated}} - A_{\text{corrupted}}}{A_{\text{clean}} - A_{\text{corrupted}}} \times 100\%$$
- **$RRR = 100\%$**: Complete restoration back to clean baseline accuracy.
- **$RRR > 0\%$**: Effective mitigation.
- **$RRR \le 0\%$**: Ineffective or detrimental (negative transfer).

### 2. Degradation Slope Flattening ($\Delta RDS$)
Quantifies the reduction in the degradation slope:
$$\Delta RDS = RDS_{\text{unmitigated}} - RDS_{\text{mitigated}}$$

### 3. Subtask Recovery Index
Tracks accuracy delta $\Delta A_{\text{subtask}} = A_{\text{mitigated}} - A_{\text{corrupted}}$ across all 9 subtasks, especially Defect Localization which suffered the greatest vulnerability in Phase 2.

---

## 📁 Output Artifacts in `results/`
- **`phase3_recovery_rate_barchart.png`**: High-resolution publication bar chart showing $RRR\%$ across corruptions for all 3 strategies.
- **`phase3_mitigation_before_after_comparison.png`**: Grouped performance comparison curves (Clean vs Corrupted vs TTA vs DAR vs Hybrid).
- **`phase3_restoration_visual_samples.png`**: Multi-panel visual demonstration showing Original $\to$ Corrupted $\to$ Restored images with defect edge zoom-ins.
- **`phase3_subtask_recovery_heatmap.png`**: 2D Heatmap showing recovery gains across the 9 industrial subtasks.
- **`results.txt`**: Comprehensive formatted log with executive summary, RRR metrics, and question-by-question cards.
- **`phase3_manifest.json`**: Machine-readable JSON summary of all experiments.
- **`phase3_results.jsonl`**: Complete record of all individual model decisions.

---

## 🚀 Execution Commands

Run the full Phase 3 benchmark:
```bash
cd "/home/anirudh-sharma/Desktop/M.tech Dissertation/Open-IAD/Local-training"
.venv/bin/python phase-3/phase3_adaptation.py --sample-size 45 --corruptions motion_blur,gaussian_noise,low_light --severities 3,5
```

Or open and run the interactive notebook:
```bash
jupyter lab phase-3/phase3_adaptation.ipynb
```
