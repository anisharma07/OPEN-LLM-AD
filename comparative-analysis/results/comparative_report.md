# Comparative Analysis — This Dissertation vs MMAD (ICLR 2025)

**Reference paper:** Jiang et al., *MMAD: A Comprehensive Benchmark for Multimodal Large
Language Models in Industrial Anomaly Detection*, ICLR 2025.
**This work:** `Qwen3-VL-2B`, `Qwen2.5-VL-3B`, `Gemma-4-E2B`, `Gemma-4-E4B` evaluated
0-shot on a 2,500-question MMAD sample, plus a 25,000-inference corruption study,
all on a single RTX 4060 Laptop GPU (8.19 GB).

Every "this work" number below is recomputed from the raw run artefacts
(`phase4_manifest.json`, `phase1_results.jsonl`, `benchmark_5k_results.jsonl`) by
`scripts/01_build_datasets.py`. Nothing is copied from the manuscript prose.

---

## 1. Making the two sets of numbers comparable

Three adjustments were needed before any comparison is meaningful.

**1.1 — Nine subtasks folded into seven.** `mmad.json` carries nine subtask types.
MMAD's Table 2 prints seven columns: it merges *Object Analysis*, *Object Structure*
and *Object Details* into a single "Object Analysis" column (paper §3.2 defines it as
"composition, position, appearance and function of the object"). This analysis applies
the same fold, weighted by question count.

**1.2 — The paper's "Average" is an unweighted mean of its seven columns.** This was
verified rather than assumed: it reproduces exactly for 20 of the 21 printed rows
(GPT-4o: 524.45 / 7 = 74.92). The one exception is Claude-3.5-sonnet, where the paper
prints 60.14 in both the *Anomaly Discrimination* and *Defect Classification* cells and
the row average (68.36) is 0.26 below the mean of the printed cells — an apparent
duplicated cell in the published table. Its value is carried through unchanged.

**1.3 — Anomaly Discrimination is a balanced accuracy in MMAD, not plain accuracy.**
The paper (§4.1) averages normal-class and abnormal-class accuracy because the split is
imbalanced. This was recomputed from the raw prediction logs, identifying normal samples
by the `/good/` path segment used across MVTec/VisA/GoodsAD:

| Model | plain accuracy | MMAD balanced accuracy | n normal | n abnormal |
|---|---|---|---|---|
| Qwen3-VL-2B | 61.30 | **61.41** | 142 | 140 |
| Qwen2.5-VL-3B | 41.73 | **47.43** | 120 | 158 |
| Gemma-4-E4B | 44.96 | **51.58** | 120 | 158 |
| Gemma-4-E2B | 50.72 | **55.04** | 120 | 158 |

The balanced figures are used throughout. For three of four models this is worth
5–7 points, so using plain accuracy would have understated them against the paper.

---

## 2. Headline result

Under MMAD's own 7-column protocol:

| Model | Params | Setting | MMAD 7-task average |
|---|---|---|---|
| Human (expert) | — | — | 86.65 |
| GPT-4o | — | 1-shot | 74.92 |
| Gemini-1.5-pro | — | 1-shot | 73.09 |
| InternVL2-76B | 76B | 1-shot | 70.75 |
| Gemini-1.5-flash | — | 1-shot | 68.90 |
| **Qwen3-VL-2B (this work)** | **2.2B** | **0-shot** | **68.40** |
| Claude-3.5-sonnet | — | 1-shot | 68.36 |
| LLaVA-NeXT-34B | 34B | 1-shot | 67.16 |
| GPT-4o-mini | — | 1-shot | 66.29 |
| MiniCPM-V2.6 | 8B | 1-shot | 66.25 |
| **Qwen2.5-VL-3B (this work)** | **3.1B** | **0-shot** | **65.51** |
| **Gemma-4-E4B (this work)** | **4.4B** | **0-shot** | **62.58** |
| **Gemma-4-E2B (this work)** | **2.3B** | **0-shot** | **58.80** |

Qwen3-VL-2B at 2.2B parameters scores **68.40** — above every open-source model of
13B or less in the paper, above two of the five commercial APIs, 2.35 points below a
76B model and 6.52 below GPT-4o. See `figures/fig1_overall_leaderboard.png` and
`fig2_accuracy_vs_scale.png`.

### Where the gap to GPT-4o actually sits

| Task | Qwen3-VL-2B | GPT-4o | Δ |
|---|---|---|---|
| Object Classification | 94.90 | 94.98 | **−0.08** |
| Defect Analysis | 82.00 | 83.41 | −1.41 |
| Object Analysis (folded) | 81.10 | 82.80 | −1.70 |
| Defect Description | 71.10 | 73.21 | −2.11 |
| Anomaly Discrimination | 61.41 | 68.63 | −7.22 |
| Defect Localization | 47.50 | 55.62 | −8.12 |
| Defect Classification | 40.80 | 65.80 | **−25.00** |

Four of seven tasks are within 2.2 points. One task — Defect Classification — carries
more than half of the total deficit. This is a sharper and more useful claim than a
single average, and it points fine-tuning effort at one task rather than at the model
as a whole. See `fig3_seven_task_profile.png` and `fig4_delta_vs_gpt4o.png`.

### The vision-encoder effect

Defect Localization separates architectures, not sizes (`fig5_defect_localization_gap.png`):

- Qwen dynamic-patch ViT: **51.3** (2.5-VL-3B), **47.5** (3-VL-2B) — mid-field among the
  paper's 7–76B systems, both above GPT-4o-mini (38.8).
- Gemma pooled-token: **31.4** (E4B), **26.4** (E2B) — in the bottom four of the entire
  table, 6.4 and 1.4 points off random chance (25.0).

The 4.4B Gemma scores below the 2.2B Qwen on this task by 16.1 points, which is the
cleanest evidence in the whole comparison that token pooling, not capacity, is the
binding constraint for spatial grounding.

---

## 3. Two axes MMAD does not measure

**3.1 — Robustness to image corruption** (`fig6_robustness_gap.png`). MMAD scores only
pristine laboratory captures. Recomputed from all 25,000 rows of the 5,000-sample study:

| Condition | Accuracy | Δ vs clean |
|---|---|---|
| Clean | 69.86 | — |
| Gaussian noise + TTA-IR | 66.68 | −3.18 |
| Gaussian noise (sev 4) | 66.64 | −3.22 |
| Motion blur (sev 4) | 66.10 | −3.76 |
| Motion blur + TTA-IR | 65.78 | −4.08 |

The overall 3.8-point drop understates the damage: Object Classification alone falls
89.2 → 69.8 (**−19.4**), while several defect subtasks *rise* under corruption — a
majority-class bias artefact, not a genuine improvement, and it should be described
that way. Note also that at the overall level **TTA-IR does not help**: it is −0.32
points on motion blur and +0.04 on Gaussian noise. Its gains are confined to specific
subtasks (Defect Localization +3.43 under blur).

**3.2 — Deployment envelope** (`fig8_deployment_envelope.png`). MMAD reports accuracy and
parameter count and nothing else. The most accurate model here is also the fastest
(6.2 fps vs 1.2 for Gemma-4-E4B), and 4-bit NF4 with CPU-offloaded embeddings puts the
largest model in the *smallest* memory footprint (3.32 GB vs 4.82 GB for Qwen3-VL-2B).

---

## 4. Threats to this comparison

Stated plainly, because they bound how strongly the headline can be worded.

1. **Shot setting.** MMAD's Table 2 is 1-shot; this work is 0-shot. The paper's own
   Table 3 measures that effect on eight models: it ranges from **−1.29 to +1.90**
   points (`fig7_shot_setting_control.png`). Smaller than the gaps discussed above, but
   it means the 0.04-point ordering against Claude-3.5-sonnet is noise, not a result.
2. **Sample size.** 2,500 questions here vs the full 39,672. At n≈278 per subtask the
   95% binomial half-width is roughly ±6 points, so per-subtask differences under about
   6 points are not separable. The overall average is much tighter (≈±1.8 at n=2,500).
3. **The question sets are not byte-identical across phases.** Phase-1 (Qwen3-VL-2B)
   drew 282 Anomaly Detection questions; the three phase-4 models drew 278. The three
   phase-4 models share one identical set; Qwen3-VL-2B's differs slightly.
4. **Model generation.** The paper's open models are 2023–2024 releases; the models here
   are 2025–2026. The comparison is a fair snapshot of *what is deployable now versus
   what the benchmark recorded*, but it is not an architecture-controlled experiment —
   some of the gain is simply two years of pretraining progress.
5. **Quantization is not held constant.** Gemma-4-E4B ran in 4-bit NF4 because of the
   8 GB ceiling; the other three ran FP16. Its numbers are a *deployable-configuration*
   result, not a clean architectural measurement.
6. **Anomaly Discrimination normal/abnormal split** is inferred from the `/good/` path
   convention. It matched the expected counts on all four runs, but it is an inference
   from file layout rather than an explicit label field.

---

## 5. Discrepancies between the draft manuscript and the run artefacts

`scripts/03_manuscript_audit.py` compares every subtask cell printed in the draft
against the corresponding artefact. **28 of 36 cells disagree by more than 0.5 points.**
Overall accuracies (71.20 / 67.68 / 65.60 / 60.92) and Defect Localization all match;
the per-subtask breakdowns largely do not.

The Phase-1 case has a clear explanation. The draft's §6.2 table, labelled
"Phase 1: Clean Baseline Benchmark (N=2,500)", does not match the Phase-1 2,500-question
run (mean absolute difference **3.62** points) — it matches the **clean column of the
5,000-sample Phase-2 study** almost exactly (mean absolute difference **0.08** points):

| Subtask | Draft §6.2 | Phase-1 run (N=2,500) | 5k clean column |
|---|---|---|---|
| Anomaly Detection | 55.56 | 61.30 | 55.56 |
| Object Classification | 89.21 | 94.90 | 89.23 |
| Object Details | 74.10 | 79.80 | 74.05 |
| Defect Classification | 45.85 | 40.80 | 45.77 |
| Defect Analysis | 76.98 | 82.00 | 76.94 |

So the table appears to be correct data under the wrong label. The fix is either to
relabel it as the 5,000-sample clean baseline, or to replace it with the Phase-1
figures from `phase1_manifest.json`. The §9.4 cross-model table has larger and less
systematic divergences (up to 11.7 points on Qwen2.5-VL-3B's Anomaly Detection) and
should be regenerated from `phase4_manifest.json`.

Full cell-by-cell audit: `results/tables/manuscript_audit.csv` and
`results/tables/manuscript_phase1_table_origin.csv`.

---

## 6. Figure index

| File | Shows |
|---|---|
| `fig1_overall_leaderboard.png` | All 21 MMAD rows + 4 local models on one 7-task average |
| `fig2_accuracy_vs_scale.png` | Accuracy by parameter band — 2–4B vs 7B…76B |
| `fig3_seven_task_profile.png` | Qwen3-VL-2B vs GPT-4o, InternVL2-76B, MiniCPM-V2.6 |
| `fig4_delta_vs_gpt4o.png` | Per-task gap of all four local models to GPT-4o |
| `fig5_defect_localization_gap.png` | Defect Localization ranking across every model |
| `fig6_robustness_gap.png` | Clean vs corrupted vs TTA-IR, overall and per subtask |
| `fig7_shot_setting_control.png` | Validity control: size of the 0-shot/1-shot effect |
| `fig8_deployment_envelope.png` | Accuracy against VRAM and against throughput |

Each figure has a matching CSV in `results/tables/`.

## 7. Reproducing

```bash
python3 comparative-analysis/scripts/01_build_datasets.py    # artefacts -> data/*.csv
python3 comparative-analysis/scripts/02_make_figures.py      # data      -> figures/*.png
python3 comparative-analysis/scripts/03_manuscript_audit.py  # draft     -> audit tables
```
