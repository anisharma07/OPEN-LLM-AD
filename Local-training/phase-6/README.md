# Phase 6: Arm-B Lab — an interactive research lab for detector-guided MLLMs

Phase 5 wired one fixed Arm-B pipeline (PatchCore → red box → Qwen3-VL-2B) and
reported one number. Phase 6 turns that pipeline into a **node graph you can
edit**, in the style of ComfyUI. You can swap the detector (PatchCore ↔ WinCLIP ↔
oracle ↔ null control), the way the box is drawn, the prompt, the MLLM, its
precision and decoding mode, and the answer parser. You can run the graph on one
MMAD question to see every intermediate output, or run it as a **batch
experiment** over a stratified MMAD sample, with grid sweeps and paired
statistics.

```
MMAD Sample ─┬─ image ─► [Corruption] ─► Detector (PatchCore | WinCLIP | Synthetic | Control)
             │                               │ anomaly map + score + τ
             │                               ▼
             │                          Heatmap → Box ──► Visual Prompt ──► MLLM ─► Answer Parser ─► Score "arm_b"
             ├─ question ─► Question Builder ─► Prompt Builder ─┘                         ▲
             │   (paraphrase / option shuffle)   (vanilla / grounded / hedged / custom)   │
             └─ sample ─► Normal References (k normal images for PatchCore / WinCLIP+)   GT
```

Arm A is the same graph without the detector branch. Put both arms in one
graph with different Score labels, and every experiment compares them on
identical questions.

## Launch

```bash
cd Local-training/phase-6
./run_lab.sh                 # → http://localhost:8765   (PORT=… HOST=… to change)
```

The script installs `fastapi`, `uvicorn` and `python-multipart` into
`Local-training/.venv` if they are missing. Everything else (torch,
transformers, sklearn, cv2) is already in that venv. The frontend is plain ES
modules with no build step and no CDN.

The lab uses the GPU automatically when `torch.cuda.is_available()`. The toolbar
shows **CPU only** otherwise. On CPU, Qwen3-VL-2B (bf16) takes roughly 10–30 s per
question, PatchCore about 1–3 s per image, and WinCLIP+ several seconds.

## Using the editor

| Action | How |
| --- | --- |
| Add a node | double-click the canvas, **＋ Node**, or right-click → Add node |
| Connect | drag from an output dot to an input dot (only matching types connect; colours = types) |
| Add a compatible node | drop a wire on empty canvas |
| Swap PatchCore for WinCLIP | right-click the node → **Replace with** (wires and same-named params carry over) |
| Ablate a stage | right-click → **Bypass** (`Ctrl+B`): the input passes straight through |
| Sweep a parameter | right-click a widget label, or node → **Sweep parameter** |
| Step through questions | set MMAD Sample `index` → *after run: increment* |
| Use your own image | drop the file on the canvas, or use template 08 |
| Shortcuts | `Ctrl+Enter` run · `Ctrl+S` save · `Ctrl+Z/Ctrl+Shift+Z` · `Ctrl+C/V/D` · `Del` · `.` fit |

Outputs are cached by *(node type, parameters, upstream fingerprints)*. If you
change only the prompt, only the prompt and the MLLM re-run. In a sweep over a
downstream parameter, the detector runs once per question.

## Nodes

| Category | Node | Notes |
| --- | --- | --- |
| Input | **MMAD Sample** | filter by dataset/category, subtask, defective/normal; becomes the iterator in experiments |
| | Load Image, Custom Question | bring your own image and MCQ |
| | **Normal References** | k normals from MMAD's `similar_templates` / `random_templates`, or the category's good folder; never includes the query |
| Pre-process | **Corruption Engine** | Phase 2's 7 corruptions × 5 severities (imported from `phase-2/corruptions.py`) |
| | Test-Time Restoration | Phase 3 TTA-IR filters (`phase-3/mitigations.py`) |
| Question | **Question Builder** | 3 paraphrase templates + option shuffling with a remapped answer key |
| Detector | **PatchCore** | backbone, layers, resolution, feature dim, greedy coreset ratio, k-NN, σ, score aggregation, τ margin |
| | **WinCLIP** | zero-shot or WinCLIP+ few-shot; CLIP model, window scales, prompt-ensemble size, temperature |
| | Synthetic Detector | built from the GT mask with controllable miss rate, false alarms, shift and dilation (oracle = defaults) |
| | Control Detector | random box / centre box / whole image: null controls |
| Localise | **Heatmap → Box** | *gate* (does it fire?) and *threshold* (which pixels?): calibrated, relative, percentile, Otsu or fixed |
| Prompt | **Visual Prompt** | box, box+wash+label (Phase 5), dashed, contour, heatmap overlay, crop-zoom, side-by-side, blur/darken outside |
| | **Prompt Builder** | vanilla (Arm A), grounded (Phase 5), + score, + location words, hedged hint, crop note, custom; optional domain knowledge |
| | Selective Gate | guides only the chosen subtasks (Phase 5's recommendation) |
| Reason | **Multimodal LLM** | any Qwen3-VL / Qwen2.5-VL / Qwen2-VL / SmolVLM / Gemma model in your HF cache, plus a chance baseline; precision (fp16/bf16/fp32/4-bit/8-bit), `generate` or **`letter-logits`** (per-option probabilities), temperature, top-p, seed, max pixels, extra image |
| Evaluate | **Answer Parser** | strict / lenient / answer-tag; failures are logged (optional Phase-5 “fallback A”) |
| | **Score** | its label names the arm in the results |
| | **Detector Metrics** | pixel AUROC, peak-in-defect, box IoU / hit vs the MMAD mask |
| Display | Preview Image, Show Text, Note | |

To add a node, write a `Node` subclass with `@register` in `lab/nodes/*.py`
(see `lab/registry.py`). It appears in the palette after a server restart.

## Experiments and metrics

**⚗ Experiment** runs the current graph over a stratified sample (equal draw per
subtask, the Phase 1/4 protocol), optionally as a grid over any parameters.
Every variant sees the same questions. Each run is written to
`runs/<run_id>/executions.jsonl`, one line per question × variant, holding
predictions, raw outputs, prompts, detector stats and per-node timings. A run
can be stopped and resumed.

The **📊 Results** page reports:

- **Accuracy** with a Wilson 95% CI, **Cohen's κ**, macro accuracy over subtasks, parse-failure rate and MLLM latency.
- **Accuracy conditioned on the cue**: when the detector fired vs. did not fire.
- **Paired comparisons** between arms (and each variant vs. the first): Δ, fixes/breaks, and an exact **McNemar** p-value, with Δ broken down by subtask.
- **Detector quality** (one row per image): image AUROC, both pooled on the τ-normalised score and as a per-category mean; pixel AUROC; peak-in-defect; box IoU and hit; and the gate's **TPR/FPR**.
- **Sweeps**: accuracy against each swept value, the **spread** (range/std, e.g. across paraphrases), and **RDS(s) = (Acc_clean − Acc_s)/s** when a `severity` is swept.
- **Calibration** in letter-logits mode: ECE and Brier score.
- A **sample browser** filterable by subtask, outcome, and fixes/breaks between arms, with each sample's thumbnails, prompts and raw answers.
- **Exports**: CSV, and *Export figures + manifest*, which writes 300-dpi PNGs, `manifest.json` and `graph.json` to `results/<run_id>/`.

Select two or more runs in the list to compare them side by side.

## Templates (Templates ▾)

1. **Arm B · PatchCore**: the Phase 5 architecture.
2. **Arm B · WinCLIP**: the same, with a CLIP-based detector (downloads `openai/clip-vit-base-patch16`, ~600 MB, on first use).
3. **Arm A vs Arm B**: the paired main comparison.
4. **Detector ceiling & controls**: oracle box vs random box vs no box. Does localisation help, or just the presence of a box?
5. **Robustness**: corruption feeding both arms; the preset sweeps severity 0–5 for RDS.
6. **Selective gating**: always-guided vs subtask-gated.
7. **Detector benchmark**: PatchCore vs WinCLIP, no MLLM (fast).
8. **Custom image playground**.
9. **Prompt sensitivity**: a paraphrase sweep for the spread.

Regenerate them with `python make_templates.py`.

## Implementation notes and caveats

- **PatchCore** (`lab/detectors/patchcore.py`): torchvision ImageNet backbones, 3×3 local aggregation, adaptive pooling to `feat_dim`, k-center greedy coreset on a 128-d random projection. τ is the largest score seen on *held-out* normal references (¼ of k). The bank is never scored against images it contains.
- **WinCLIP** (`lab/detectors/winclip.py`): re-implemented on HF `CLIPModel`, with real window masking (ViT run on [CLS] + each window's tokens), a compositional prompt ensemble and harmonic aggregation. WinCLIP+ adds a memory of reference window embeddings and penultimate-layer patch tokens. The backbone is OpenAI ViT-B/16 @224 rather than the paper's LAION ViT-B/16+ @240, so absolute numbers are lower than the paper's. On a 24-image VisA/DS-MVTec check, pixel AUROC was 0.68 for zero-shot, 0.83 for WinCLIP+ and 0.92 for PatchCore.
- **Letter-logits** reads the softmax over the option-letter tokens at the first answer position. It is a single forward pass, so parse failures are impossible.
- Images larger than `max_side` (default 1024) are downscaled on load. The MLLM additionally caps input at `max_pixels` (default 512·28·28, as in Phase 5).
- The **chance baseline** answers uniformly at random, seeded per question. It is useful as a floor and for testing graphs quickly.
- `runs/`, `cache/` and `uploads/` are git-ignored. `results/<run_id>/` holds the small exported artefacts intended for the thesis.

## Layout

```
phase-6/
├── run_lab.sh, requirements-lab.txt, make_templates.py
├── lab/            server.py (FastAPI + job queue) · executor.py (graph run + cache) · registry.py
│   ├── nodes/      inputs.py · vision.py · language.py · evaluation.py
│   ├── detectors/  patchcore.py · winclip.py
│   └── data.py (MMAD index, refs, GT masks) · mllm.py · metrics.py · experiments.py · figures.py · viz.py
├── web/            index.html · style.css · js/{main,editor,widgets,lab,charts,util}.js
├── workflows/      saved graphs (+ templates/)
├── runs/           experiment logs (git-ignored)
└── results/        exported figures + manifests
```
