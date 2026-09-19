# Phase 2: Industrial Corruption Robustness Study

## 🎯 Purpose & Scope
Phase 2 represents the **Core Scientific Contribution** of the M.Tech Dissertation:
**"Robustness and Reproducibility of Open-Source Small Multimodal LLMs for Industrial Anomaly Detection"**.

While Phase 1 established the baseline accuracy under clean, laboratory conditions ($A_{\text{clean}} = 71.2\%$), actual manufacturing environments (conveyor lines, sorting robotics, automated optical inspection stations) subject visual sensors to severe physical disturbances.

Phase 2 empirically measures:
1. The **Robustness Degradation Slope (RDS)** across 7 industrial corruptions.
2. The **Critical Failure Boundary (CFB)** where the vision-language model fails below acceptable reliability thresholds ($<50\%$).
3. Which industrial inspection subtasks are **resilient vs fragile** under factory disturbances.

---

## 🛠️ Industrial Corruption Taxonomy (7 Corruptions $\times$ 5 Severities)

| # | Corruption Name | Physical Mechanism in Factory | Implementation |
| :--- | :--- | :--- | :--- |
| 1 | **Gaussian Noise** | Thermal sensor noise in budget cameras / low lighting | Additive Gaussian ($\sigma \in [12, 105]$) |
| 2 | **Motion Blur** | Rapid conveyor belt motion & robotic arm vibrations | Directional linear diagonal blur ($k \in [5, 31]$ px) |
| 3 | **Low Light (Underexposure)** | Inadequate factory lighting / power brownouts | Nonlinear gamma drop ($\gamma \in [1.2, 2.6]$, factor $0.85 \to 0.25$) |
| 4 | **Specular Glare** | Blinding reflections on shiny metal / wet surfaces | 2D Gaussian hotspot intensity addition ($80 \to 240$) |
| 5 | **Defocus Blur** | Lens depth-of-field drift / focal misadjustment | Gaussian disk optical blur ($r \in [1.5, 11.0]$ px) |
| 6 | **Perspective Tilt** | Off-axis camera mounting / vibrating brackets | Quadrilateral perspective projection warp ($4\% \to 25\%$) |
| 7 | **Compression Artifacts** | Edge device bandwidth throttling / lossy transmission | JPEG DCT block compression ($Q \in [65, 8]$) |

---

## 📐 Mathematical Formulations

### 1. Robustness Degradation Slope (RDS)
Measures the accuracy loss per unit increase in corruption severity $s \in \{0, 1, 2, 3, 4, 5\}$:
$$RDS_c = \frac{A_{\text{clean}} - A_{c, s=5}}{5}$$
$$\overline{RDS} = \frac{1}{7} \sum_{c=1}^7 RDS_c$$
- **Interpretation**: A high positive RDS indicates extreme vulnerability (steep performance collapse). An RDS near zero signifies noise invariance.

### 2. Critical Failure Boundary (CFB)
Identifies the exact severity transition point where the model becomes untrustworthy for industrial automation:
$$CFB_c = \min \{ s \in \{1..5\} \mid A_{c, s} < 0.50 \}$$

### 3. Relative Robustness Index (RRI)
Quantifies the normalized area under the robustness curve:
$$RRI_c = \frac{\frac{1}{5} \sum_{s=1}^5 A_{c, s}}{A_{\text{clean}}}$$

---

## 📁 Output Artifacts in `results/`
- **`phase2_corruption_preview_grid.png`**: High-resolution 7x6 visual grid showing an industrial inspection image subjected to all 7 corruptions across all 5 severity levels.
- **`phase2_degradation_curves.png`**: Publication line plot mapping Accuracy vs Corruption Severity for all 7 corruptions.
- **`phase2_rds_radar.png`**: Polar radar chart showing the RDS profile across all corruptions.
- **`phase2_subtask_resilience_matrix.png`**: 2D Heatmap revealing which subtasks collapse first under extreme corruption ($s=5$).
- **`results.txt`**: Detailed report with RDS rankings, CFB table, and question cards.
- **`phase2_manifest.json`**: Machine-readable JSON manifest.
- **`phase2_results.jsonl`**: Full evaluation checkpoints.

---

## 🚀 How This Connects to Phase 3
Phase 2 pinpoints the exact failure modes and most destructive corruptions (e.g. Motion Blur and Specular Glare). 
**Phase 3 (Prompt Tuning, LoRA Adaptation & Test-Time Augmentation)** will design targeted mitigation strategies to flatten the RDS slope and recover lost accuracy without retraining the backbone model.
