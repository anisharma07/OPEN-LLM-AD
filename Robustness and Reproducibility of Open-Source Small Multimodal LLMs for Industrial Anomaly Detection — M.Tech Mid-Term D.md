# Robustness and Reproducibility of Open-Source Small Multimodal LLMs for Industrial Anomaly Detection

## M.Tech Thesis — Mid-Term Review Documentation

2026-09-18 · @Someone

Submitted in partial fulfilment of the requirements for the degree of Master of Technology.

| Field | Detail |
| --- | --- |
| Degree | M.Tech |
| Department | \[Department\] |
| Institute | \[Institute\] |
| Candidate | \[Your name\], Roll No. \[\_\_\_\_\] |
| Supervisor | \[Supervisor name\] |
| Review stage | Mid-term progress review |
| Date of review | 18 September 2026 |
| Hardware available | Kaggle T4 / P100 (16 GB), local RTX 4060 (8 GB) |

## Index

| Section | What it covers |
| --- | --- |
| [1. Problem statement and research questions](#mkrkd5xybdm.166) | Thesis claim, RQ1-RQ4, scope, why it is publishable |
| [2. Literature review](#mkrkd5xybdm.2578) | Four phases of the field, paper status table, open questions |
| [3. Datasets and benchmarks](#mkrkd5xybdm.7797) | MMAD, MVTec-AD, VisA, corruption suite, licensing |
| [4. Limitations and the gap](#mkrkd5xybdm.10597) | L1-L5, the four openings, one-sentence gap statement |
| [5. Proposed architecture](#mkrkd5xybdm.13482) | Pipeline diagram, components, model zoo, determinism controls |
| [6. Implementation in phases](#mkrkd5xybdm.17336) | Phases 0-5, GPU hours, exit criteria, calendar |
| [7. Evaluation protocol and metrics](#mkrkd5xybdm.21710) | Metric definitions, reporting rules, contamination check |
| [8. Risk register and hardware feasibility](#mkrkd5xybdm.23646) | Risks, mitigations, what is not attempted |
| [9. Expected contributions and venues](#mkrkd5xybdm.25506) | Four contributions, publication targets |
| [10. Future work](#mkrkd5xybdm.26838) | Extensions beyond this thesis |
| [11. Conclusion](#mkrkd5xybdm.27813) | Mid-term status and immediate next step |
| [12. References](#mkrkd5xybdm.28789) | Citation list with venue verification notes |
| [Appendix A — Presentation mapping](#mkrkd5xybdm.31710) | 16-slide deck plan mapped to sections |

## 1. Problem statement and research questions

**Thesis claim.** Reported accuracy on industrial anomaly detection (IAD) benchmarks overstates what open-source small multimodal LLMs deliver in a factory, because published numbers are measured on clean images, with one prompt, one seed, and often a closed API model. This work measures the gap and reports what survives.

Industrial inspection runs on edge hardware with imperfect cameras. A model that scores 80% on curated MVTec-AD images may drop sharply under motion blur, poor lighting, JPEG compression, or a change of viewpoint. No published work has measured this for small (2B-8B) open-weight vision-language models, which are the only ones deployable on factory edge devices and the only ones a student or small firm can run.

### Research questions

1. **RQ1 (Robustness).** How much does the anomaly-detection and anomaly-classification accuracy of open small multimodal LLMs degrade under realistic image corruptions, and which corruption types hurt most?
2. **RQ2 (Reproducibility).** How stable are these results across random seeds, prompt phrasings, quantisation levels (FP16 vs 4-bit) and inference libraries? Is a single reported number meaningful?
3. **RQ3 (Open vs closed).** How far behind GPT-4o-class closed models are open small models on the same benchmark and prompt, and does the gap widen or narrow under corruption?
4. **RQ4 (Mitigation).** Can a cheap, training-free intervention (prompt structuring, reference-image conditioning, test-time augmentation, or self-consistency voting) recover a measurable part of the lost accuracy within a 16 GB GPU budget?

### Scope

- **In scope:** image-level anomaly detection and defect-type classification via multimodal question answering; open-weight models of roughly 2B-8B parameters; public industrial datasets; inference-time methods.
- **Out of scope:** training a new foundation model, pixel-level segmentation as the primary task, 3D or point-cloud anomaly detection, video, and any experiment needing more than 16 GB of VRAM.

### Why this is publishable

It is a measurement and benchmarking contribution, not a recombination of existing blocks. The field has converged on a single accuracy number per model on clean data; this work shows that number is fragile and provides a protocol for reporting it honestly. Such papers age well, are cheap to run, and are hard for reviewers to dismiss when the code and corrupted data are released.

## 2. Literature review

The field moved through three phases in four years: memory-bank detectors that saturated MVTec-AD, CLIP-based zero-shot detectors, and multimodal LLMs that explain as well as detect. Robustness evaluation only began in 2026.

### 2.1 Phase 1 — Unsupervised detection (2022-2023)

[PatchCore](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html) (CVPR 2022, Tübingen + Amazon AWS) stores patch features of normal images in a coreset memory bank and scores a test patch by nearest-neighbour distance, reaching about 99.6% image AUROC on MVTec-AD. It defines the ceiling that made MVTec-AD a saturated benchmark. [WinCLIP](https://arxiv.org/abs/2303.14814) (CVPR 2023, KAIST + AWS AI Labs) replaced training with language: window-based CLIP scoring against prompts such as "a photo of a damaged \[object\]" gives zero-shot detection and few-shot extension. Neither model explains its decision, and neither was ever tested under image degradation.

### 2.2 Phase 2 — LLMs enter the loop (2024-2025)

[AnomalyGPT](https://ojs.aaai.org/index.php/AAAI/article/view/27963) (AAAI 2024, CASIA) was the first widely cited work to put a large vision-language model inside the detection loop, removing manual thresholds and adding dialogue. [Myriad](https://arxiv.org/abs/2310.19070) (Science China Information Sciences, 2026; HIT + Pazhou Lab) formalised the "vision expert" pattern: a conventional detector produces a noisy anomaly map, and the LMM calibrates and verbalises it. [Echo](https://arxiv.org/abs/2501.15795) decomposed the same idea into four modules (reference extractor, knowledge guide, reasoning expert, decision maker).

[VELM](https://arxiv.org/abs/2505.02626) (CVPR Workshops 2025, Freiburg/Bonn + Endress+Hauser) shifted the question from detection to classification: given that something is anomalous, what kind of defect is it? Its lasting contribution is benchmark hygiene — it found 36 mislabelled MVTec-AD samples and merged ambiguous defect classes to produce MVTec-AC and VisA-AC. Its oracle experiment (ground-truth masks instead of predicted ones) separates detector error from classifier error, and on VisA-AC accuracy falls from 87.6% with perfect masks to 69.6% with a real detector.

### 2.3 Phase 3 — Specialist assistants and benchmarks (2025-2026)

[MMAD](https://arxiv.org/abs/2410.09453) (ICLR 2025, SUSTech + Tencent YouTu) is the reference benchmark: 39,672 multiple-choice questions over 8,366 images from 38 product classes across four public datasets, organised into seven inspection subtasks. Its headline finding is that the best commercial model, GPT-4o, averaged 74.9%, far below industrial requirements.

[Anomaly-OV](https://arxiv.org/abs/2502.07601) (CVPR 2025, Johns Hopkins + Honda Research Institute) trains a specialist assistant on LLaVA-OneVision with a Look-Twice Feature Matching mechanism, releasing the Anomaly-Instruct-125k tuning set and the VisA-D&R benchmark; it reports 88.6% average image AUROC against 85.3% for AdaCLIP. [AD-Copilot](https://arxiv.org/abs/2603.13779) (preprint, IEEE TIP format, 2026; same group as MMAD) argues that general MLLMs encode each image separately and only compare them in language, missing subtle visual differences, and adds a cross-attention Comparison Encoder, reporting 82.3% on MMAD. It also alleges that some competing methods gain on MMAD partly through training on overlapping data.

### 2.4 Phase 4 — Robustness and evaluation (2026, closest prior work)

[RobustMAD](https://arxiv.org/abs/2607.16243) (TMLR 2026; SUTD, A\*STAR, NTU, Chongqing University) is the nearest neighbour to this thesis and must be positioned against carefully. It is the first deployment-motivated benchmark for multimodal *small* language models in industrial inspection, spanning object understanding, anomaly detection, unanswerable or ill-posed queries, and visual quality degradation. Two findings matter here: top small models can beat larger ones such as Phi-4-Multimodal-Instruct and GPT-5 Nano, and phrasing alone can flip a verdict — a model that correctly flags frayed copper wire under a neutral query returns "defect-free" under a confirmation-seeking rephrasing. Its visual degradation study applies motion blur to a random half of the images and low lighting to the other half.

### 2.5 Paper status table

Venue status decides whether a number can be cited as established. Verified against the listed source page.

| Work | Venue | Year | Peer-reviewed | Code | Role in this thesis |
| --- | --- | --- | --- | --- | --- |
| [RobustMAD](https://arxiv.org/abs/2607.16243) | TMLR | 2026 | Yes | Yes | Closest prior work; baseline and contrast |
| [MMAD](https://arxiv.org/abs/2410.09453) | ICLR | 2025 | Yes | Yes | Primary benchmark |
| [Anomaly-OV](https://arxiv.org/abs/2502.07601) | CVPR | 2025 | Yes | Yes | Specialist model baseline |
| [VELM](https://arxiv.org/abs/2505.02626) | CVPR Workshops | 2025 | Yes (workshop) | Check | Classification task + cleaned labels |
| [PatchCore](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html) | CVPR | 2022 | Yes | Yes | Vision-expert / detector reference |
| [WinCLIP](https://arxiv.org/abs/2303.14814) | CVPR | 2023 | Yes | Unofficial only | Zero-shot reference |
| [AnomalyGPT](https://ojs.aaai.org/index.php/AAAI/article/view/27963) | AAAI | 2024 | Yes | Yes | Related work |
| [Myriad](https://arxiv.org/abs/2310.19070) | Sci. China Inf. Sci. | 2026 | Yes | Yes | Related work |
| [AD-Copilot](https://arxiv.org/abs/2603.13779) | arXiv (TIP format) | 2026 | Not yet | Yes | Cite as preprint; leakage argument |
| [Echo](https://arxiv.org/abs/2501.15795) | arXiv / CIE | 2025 | Partial | Check | Related work |
| MS-CLIP-AD | ICBASE | 2026 | Yes (low-tier) | No | Cite only as trend example |
| LLM Hybrid Framework | IICAIET | 2025 | Yes (low-tier) | No | Cite only as untested proposal |

### 2.6 What the literature does not answer

No published work reports how small open multimodal models behave under *severity-graded* corruption, across seeds, prompts and quantisation levels. RobustMAD establishes that fragility exists; it does not quantify how fast accuracy decays with severity, nor whether a reported number is reproducible.

## 3. Datasets and benchmarks

MMAD is the primary evaluation set; MVTec-AD and VisA supply raw images for the corruption suite; RobustMAD is the external comparison point. All are public and downloadable without institutional access.

| Dataset | Content | Task form | Use in this work |
| --- | --- | --- | --- |
| [MMAD](https://github.com/jam-cc/MMAD) | 39,672 questions over 8,366 images, 38 classes, 4 source datasets | Multiple-choice VQA, 7 subtasks | Primary benchmark; clean and corrupted runs |
| [MVTec-AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) | 15 classes, 5,354 images, pixel masks | Detection / segmentation | Source images for corruption; sanity checks |
| [VisA](https://github.com/amazon-science/spot-diff) | 12 classes, 10,821 images | Detection / segmentation | Second domain, harder than MVTec-AD |
| MVTec-AC / VisA-AC ([VELM](https://arxiv.org/abs/2505.02626)) | Cleaned and merged defect-type labels | Anomaly classification | Classification sub-experiment |
| [RobustMAD](https://github.com/en-research/RobustMAD) | Open-ended queries + motion blur / low-light variants | Open-ended QA | External comparison; cross-check of findings |
| [MVTec AD 2](https://www.mvtec.com/company/research/datasets/mvtec-ad-2) | Advanced scenarios, lighting variation | Detection | Optional real (not synthetic) degradation check |

### 3.1 Why MMAD is the primary benchmark

Multiple-choice answers make scoring deterministic and cheap, which matters when the same question must be run hundreds of times across corruption severities and seeds. Its seven subtasks let a single run report detection, defect classification, defect localisation and object analysis separately, so a robustness drop can be attributed to a specific capability rather than a single blended score.

### 3.2 Corruption suite (constructed, not downloaded)

The corrupted evaluation sets are a contribution of this thesis. Corruptions are applied to MMAD's source images with fixed seeds and five severity levels each, following the established ImageNet-C protocol so that severity is comparable across types.

| Corruption | Why it is industrially realistic |
| --- | --- |
| Motion blur | Moving conveyor, handheld inspection |
| Defocus blur | Autofocus failure, varying part height |
| Low light / brightness shift | Shift changes, shadowing, lamp ageing |
| Gaussian and shot noise | Cheap sensors, short exposure |
| JPEG compression | Bandwidth-limited camera streams |
| Downscaling (resolution loss) | Low-cost edge cameras |
| Rotation and small translation | Imperfect part placement |

Motion blur and low light overlap with RobustMAD by design, which gives a point of agreement to validate the pipeline against; the remaining five types and the severity grading are new.

### 3.3 Licensing and access

MVTec datasets are free for non-commercial research use and require accepting MVTec's licence; VisA is released by Amazon Science; MMAD is on Hugging Face with images and captions. All corrupted derivatives must be released as *generation scripts plus seeds*, not as re-hosted images, to stay within the source licences. This is recorded now because it affects what can be published later.

## 4. Limitations of existing work and the gap

Five limitations recur across the reviewed papers. Together they mean a published accuracy number does not predict field behaviour.

**L1 — Clean-image evaluation only.** MVTec-AD, VisA and MMAD images are captured under controlled lighting with fixed cameras. Every number in Section 2, except RobustMAD's, is measured on such images. A factory camera does not produce them.

**L2 — Dependence on closed models.** VELM and Echo build on GPT-4o; MMAD's strongest result is GPT-4o's 74.9%. Closed models change silently behind the same name, cannot be run on-premises where privacy matters, and cannot be inspected. Results built on them are not reproducible in the strict sense.

**L3 — Single-run reporting.** Papers report one number per model per dataset. Temperature, prompt phrasing, seed, quantisation and inference library are usually unreported. RobustMAD shows that a rephrased query alone can flip a verdict from defect to defect-free, which implies single-run numbers carry an unmeasured error bar.

**L4 — Inconsistent and sometimes wrong metrics.** VELM's "macro accuracy", defined as TP/(TP+FP+FN), is the Jaccard index, not accuracy. MS-CLIP-AD's claim that pixel AUROC exceeds image AUROC on 12 of 15 categories is contradicted by its own table, which supports 7. Cross-paper comparison is therefore unsafe without re-running.

**L5 — Possible benchmark contamination.** AD-Copilot alleges that some methods improve on MMAD partly by training on overlapping data. Public benchmarks also risk appearing in foundation-model pre-training corpora. No reviewed paper tests for memorisation.

### 4.1 The gap this thesis fills

RobustM![image.png](blob/2a97b428-e3c8)part of L3 for small models, using open-ended queries and two degradations applied to disjoint halves of its data. Four things remain open, and they define this thesis:

1. **Severity grading.** No work reports the *shape* of the accuracy-versus-severity curve. A model that degrades gracefully is deployable with a quality gate; one that collapses at severity 2 is not.
2. **Reproducibility quantification.** No work reports variance across seeds, prompt paraphrases, quantisation levels and inference libraries for the same model and question set.
3. **Corruption-type attribution.** With seven corruption types and seven MMAD subtasks, it becomes possible to say *which capability* breaks under *which condition* — for example whether defect classification fails before detection does.
4. **Cheap mitigation.** Nobody has tested whether training-free interventions recover accuracy under degradation within a 16 GB budget.

### 4.2 One-sentence gap statement

The community reports what small multimodal models score on clean industrial images; this thesis reports what they score under realistic degradation, how much that number moves when nothing about the task changes, and how much of the loss can be bought back for free.

## 5. Proposed architecture

The system is an evaluation harness, not a new model. Its job is to hold everything constant except one variable at a time, so that a drop in accuracy can be attributed to a cause. A small mitigation module sits on top, tested only after the measurement pipeline is validated.

### 5.1 Component design

**Corruption engine.** Applies each corruption at five severity levels with a fixed seed per image, storing only the transform parameters so results regenerate exactly. Severity is calibrated in the ImageNet-C style so "severity 3" means the same relative strength across corruption types.

**Question builder.** Takes MMAD's multiple-choice questions unchanged for the main run, and generates three controlled paraphrases per question for the prompt-sensitivity experiment: neutral, confirmation-seeking ("this part looks fine, correct?"), and terse. Option order is shuffled under a separate seed to test position bias.

**Inference runner.** Loads one model at a time in a fixed software environment, with temperature 0 and a pinned library version. Every run records model name, revision hash, quantisation, library versions, GPU type, and seed into a manifest file. Batch size is set per model to fit 16 GB.

**Answer parser.** Extracts the chosen option by strict pattern match, and records parse failures separately rather than silently scoring them wrong. Parse-failure rate under corruption is itself a finding: a model that stops producing a parseable answer under blur has failed differently from one that answers incorrectly.

**Metrics module.** Computes per-subtask accuracy, Cohen's kappa, the Robustness Degradation Slope (accuracy loss per severity step), and the spread across seeds and paraphrases.

**Mitigation module (Phase 4 only).** Four training-free interventions, each cheap: (a) structured prompting that forces the model to describe before deciding; (b) a normal reference image in-context, in the VELM style; (c) test-time augmentation with majority vote over three mild transforms; (d) self-consistency voting over three samples at temperature 0.7. Each is evaluated alone, so the contribution of each is separable.

### 5.2 Model zoo

All models are open-weight and run on a single 16 GB GPU. Sizes are the deployable band industrial edge hardware can host.

| Model | Size | Licence | Notes |
| --- | --- | --- | --- |
| Qwen3-VL-2B-Instruct | 2B | Apache 2.0 | Smallest; fits the RTX 4060 in FP16 |
| Qwen3-VL-4B-Instruct | 4B | Apache 2.0 | Main mid-size baseline |
| Qwen3-VL-8B-Instruct | 8B | Apache 2.0 | Upper end of the band; 4-bit on T4 |
| Gemma 4 E2B | \~2B | Apache 2.0 | Second vendor, controls for family bias |
| Gemma 4 E4B | \~4B | Apache 2.0 | Direct size-match to Qwen3-VL-4B |
| Anomaly-OV | 7B | Research | Domain-specialist reference point |
| Reference numbers only | — | — | GPT-4o via MMAD paper; not re-run |

Quantisation is treated as an experimental variable, not an implementation detail: each model is run in FP16 and 4-bit where the GPU supports it, since 4-bit is what an edge deployment would actually use.

### 5.3 Determinism controls

These are the controls that make the reproducibility claim defensible:

- Temperature 0 and fixed seeds for every run except the deliberate self-consistency experiment
- Pinned versions of transformers, torch, and the quantisation library, recorded per run
- Model revision hashes pinned, not just model names
- Identical image pre-processing path for clean and corrupted runs
- All manifests and raw model outputs stored, not only aggregate scores

## 6. Implementation in phases

Six phases, each with an exit criterion that must be met before the next starts. The GPU budget assumes roughly 30 Kaggle GPU-hours per week with a 12-hour session cap, so every phase is designed to checkpoint and resume.

| Phase | Work | GPU hours | Exit criterion |
| --- | --- | --- | --- |
| 0 | Environment, data download, single-image smoke test | \~5 | One model answers one MMAD question end to end |
| 1 | Clean-baseline reproduction | \~25 | Reproduce a published MMAD number within a stated tolerance |
| 2 | Corruption engine + corrupted evaluation | \~60 | Full accuracy-vs-severity curves for all models |
| 3 | Reproducibility study | \~40 | Variance reported for seeds, paraphrases, quantisation |
| 4 | Mitigation experiments | \~45 | Each intervention measured alone, with ablation |
| 5 | Write-up, code release, paper submission | \~10 | Reproducible repository + submitted manuscript |

### Phase 0 — Environment and smoke test

1. Fix the software stack and record exact versions; build one Kaggle notebook that installs from a pinned requirements file.
2. Download MMAD from Hugging Face and MVTec-AD/VisA source images; verify checksums.
3. Run Qwen3-VL-2B on ten MMAD questions; confirm the answer parser extracts options correctly.
4. Establish the checkpoint-and-resume pattern: results appended to disk after every batch, session-safe.

*Risk handled here:* Kaggle session resets wiping work. Nothing later is attempted until resume works.

### Phase 1 — Clean baseline

1. Run all model-zoo entries on the full MMAD question set, FP16 where memory allows, otherwise 4-bit.
2. Record per-subtask accuracy and Cohen's kappa; compare with MMAD's published table.
3. Document every discrepancy between your number and the published number, with a hypothesis for each.

*This phase is a contribution in itself*: an independent reproduction of MMAD results for small open models, with the differences explained.

**Exit criterion:** at least one model reproduces its published score within a tolerance you state in advance (for example ±2 accuracy points), or the deviation is explained.

### Phase 2 — Corruption study (core contribution)

1. Implement the seven corruptions at five severities; visually verify samples at each severity.
2. Generate corrupted evaluation sets from parameters, not stored images, to keep storage small.
3. Run every model on every corruption-severity combination. This is the largest compute block; prioritise: all models at severities 1, 3, 5 first, then fill 2 and 4 if hours allow.
4. Compute accuracy-vs-severity curves per corruption type, per MMAD subtask.
5. Cross-check the motion-blur and low-light results against RobustMAD's findings; agreement validates the pipeline, disagreement is itself worth reporting.

**Exit criterion:** a complete curve set, plus a ranked list of which corruption breaks which subtask first.

### Phase 3 — Reproducibility study

1. Five seeds per model on a fixed 1,000-question subset; report the spread, not just the mean.
2. Three prompt paraphrases per question, including the confirmation-seeking phrasing RobustMAD showed to be dangerous.
3. Option-order shuffling to measure position bias.
4. FP16 versus 4-bit on the same questions, same hardware.
5. Where possible, the same model under two inference libraries.

**Exit criterion:** a single table showing, for each model, how much its reported accuracy moves when nothing about the task changes.

### Phase 4 — Mitigation

1. Implement the four training-free interventions from Section 5.1.
2. Evaluate each independently on the severity-3 corrupted set, then the best two in combination.
3. Record cost: added latency and memory per intervention, since a mitigation that doubles inference time is not deployable.

**Exit criterion:** a table of accuracy recovered versus cost paid, with an honest statement if nothing works — a negative result here is still publishable within the larger paper.

### Phase 5 — Write-up and release

1. Public repository: corruption scripts, seeds, run manifests, raw outputs, analysis notebooks.
2. Paper draft targeting the venues in Section 9.
3. Thesis chapters mapped from Sections 2-7 of this document.

### Suggested calendar

Assuming a mid-term review now and a final submission window of roughly six months, Phases 0-1 take four weeks, Phase 2 six weeks, Phase 3 three weeks, Phase 4 four weeks, Phase 5 three weeks, leaving about four weeks of slack for failures and re-runs. Slack is not optional; Phase 2 will over-run.

## 7. Evaluation protocol and metrics

Every metric is defined here in advance, because Section 4 (L4) showed that loose metric definitions are a real failure mode in this literature.

| Metric | Definition | What it answers |
| --- | --- | --- |
| Accuracy | Correct answers / total questions, per subtask | Baseline capability |
| Cohen's kappa | Agreement corrected for chance | Guards against inflated multiple-choice scores |
| Robustness Degradation Slope (RDS) | Accuracy points lost per severity step, fitted over severities 1-5 | How fast a model fails |
| Relative Robustness | Accuracy at severity s / accuracy on clean images | Size-independent comparison across models |
| Parse-failure rate | Responses no strict parser can map to an option | Distinguishes wrong answers from broken output |
| Seed spread | Max minus min accuracy across five seeds | Reproducibility under identical conditions |
| Prompt spread | Max minus min accuracy across three paraphrases | Sensitivity to phrasing |
| Quantisation delta | FP16 accuracy minus 4-bit accuracy | Cost of deployment-realistic precision |
| Flip rate | Share of questions whose answer changes between two conditions | Instability that averages hide |

### 7.1 Reporting rules adopted

- Any headline accuracy is reported as mean with spread across seeds, never a bare number.
- Chance level is stated beside every multiple-choice result.
- Statistical comparison between two models uses a paired bootstrap over questions, since the same questions are answered by both; report confidence intervals rather than p-values alone.
- The exact metric formula goes in the paper, to avoid repeating VELM's accuracy/Jaccard confusion.

### 7.2 Contamination check

A short memorisation probe is run on a sample: ask the model to describe an MMAD image without showing it, and ask about a defect label not present in a given image. Strong performance without the image is evidence of leakage. This is a screening test, not proof, and will be reported as such.

## 8. Risk register and hardware feasibility

Available hardware: Kaggle Tesla T4 (16 GB) and P100 (16 GB) at roughly 30 GPU-hours per week with a 12-hour session cap, plus a local RTX 4060 (8 GB). No cluster access.

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| Phase 2 compute over-runs the weekly quota | High | Schedule slip | Prioritise severities 1/3/5; subsample questions per class; results are valid on a documented subset |
| Kaggle session reset loses a long run | High | Lost hours | Checkpoint after every batch; results appended to persistent storage; resume-from-manifest built in Phase 0 |
| 4-bit quantisation unsupported on P100 | Medium | Halves usable GPU pool | Route quantised runs to T4; use P100 for FP16 small models and image processing |
| 8B model does not fit 16 GB even in 4-bit with long prompts | Medium | Model dropped | Cap image resolution and context; if still infeasible, report the 2B-4B band and state the limit |
| RobustMAD scoops part of the contribution | Medium | Reduced novelty | Positioning already separates severity grading, reproducibility and mitigation; cite RobustMAD as baseline, not competitor |
| Open-ended answers hard to score | Medium | Noisy results | Multiple-choice MMAD is the primary task; open-ended is a secondary analysis only |
| Published baselines cannot be reproduced | Medium | Phase 1 stalls | Discrepancy documented as a finding; thesis continues on your own measured baseline |
| Local 8 GB GPU insufficient for anything useful | Low | Minor | Use it for corruption generation, parsing and analysis, not inference |

### 8.1 What is explicitly not attempted

Training Anomaly-OV or AD-Copilot from scratch is infeasible on this hardware and is not attempted; their released weights are used for inference and their published numbers are cited. No experiment in this plan requires more than 16 GB of VRAM or multi-day continuous training.

## 9. Expected contributions and publication targets

Four contributions, each independently defensible so that a weak result in one does not sink the thesis.

1. **A severity-graded corruption benchmark** for industrial anomaly VQA, released as scripts and seeds over public datasets.
2. **The first accuracy-versus-severity characterisation** of open small multimodal models on MMAD, attributed by corruption type and inspection subtask.
3. **A reproducibility audit** quantifying how much a reported number moves across seeds, paraphrases, option order and quantisation.
4. **An evaluated set of training-free mitigations**, with accuracy recovered measured against latency and memory cost.

### 9.1 Venue options

| Target | Fit | Realism |
| --- | --- | --- |
| CVPR / ICCV VAND workshop | Exactly this community; workshops accept benchmark and analysis papers | Strong first target |
| ICIP / ICPR / BMVC | Full peer-reviewed conference, benchmark papers welcome | Realistic with complete Phases 0-3 |
| TMLR | Where RobustMAD appeared; values thorough evaluation over novelty | Ambitious but well matched |
| IEEE Access / Journal of Intelligent Manufacturing | Journal route, longer format | Fallback with full results |

A workshop submission after Phase 3 and a journal or conference submission after Phase 4 is the sensible two-step plan; the workshop paper becomes a thesis chapter either way.

## 10. Future work

These extend the thesis but are out of scope for the current hardware and timeline.

- **Robustness-aware fine-tuning.** QLoRA on corrupted images to test whether small models can be hardened cheaply, once the measurement baseline exists.
- **Real degradation instead of synthetic.** Photographing a physical object under genuine motion and lighting variation, or using MVTec AD 2, to check that synthetic corruption predicts real-world failure.
- **Pixel-level robustness.** Extending the study from answer accuracy to localisation quality under degradation.
- **Uncertainty and abstention.** Teaching the model to say "image quality insufficient" rather than guessing — arguably the most valuable industrial behaviour, and a natural follow-up paper.
- **On-device latency study.** Measuring throughput on real edge hardware such as a Jetson-class device.
- **Cross-domain transfer.** Testing whether robustness rankings measured on MVTec-AD hold on medical or agricultural anomaly data.

## 11. Conclusion

The literature has established that multimodal LLMs can detect and describe industrial anomalies, and that the best reported scores still fall short of industrial requirements. What it has not established is whether those scores survive the conditions of an actual production line, or whether they are stable enough to be trusted at all.

This thesis measures both, for the class of models that can actually be deployed on-site: open-weight models of 2B to 8B parameters. The design is deliberately matched to available hardware — inference-heavy, training-light, checkpointed against session limits — so that every claim in it can be produced on a single 16 GB GPU and independently reproduced by anyone else with the same.

At mid-term, the literature review, gap analysis, benchmark selection, architecture and phased plan are complete. The immediate next step is Phase 0: pin the environment, download MMAD, and put one model through one question end to end.

## 12. References

Verify each venue against its official proceedings page before the final citation list; community paper lists carry venue errors. Sources opened for this document are linked.

1. Roth, K. et al. *Towards Total Recall in Industrial Anomaly Detection.* [CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html). Code: github.com/amazon-science/patchcore-inspection
2. Jeong, J. et al. *WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation.* [CVPR 2023](https://arxiv.org/abs/2303.14814).
3. Gu, Z. et al. *AnomalyGPT: Detecting Industrial Anomalies Using Large Vision-Language Models.* [AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/27963). Code: github.com/CASIA-IVA-Lab/AnomalyGPT
4. Li, Y. et al. *Myriad: Large Multimodal Model by Applying Vision Experts for Industrial Anomaly Detection.* [Science China Information Sciences 69(9), 2026](https://arxiv.org/abs/2310.19070).
5. Jiang, X. et al. *MMAD: A Comprehensive Benchmark for Multimodal Large Language Models in Industrial Anomaly Detection.* [ICLR 2025](https://arxiv.org/abs/2410.09453). Code and data: [github.com/jam-cc/MMAD](https://github.com/jam-cc/MMAD)
6. Xu, J. et al. *Towards Zero-Shot Anomaly Detection and Reasoning with Multimodal Large Language Models.* [CVPR 2025](https://arxiv.org/abs/2502.07601). Code: [github.com/honda-research-institute/Anomaly-OneVision](https://github.com/honda-research-institute/Anomaly-OneVision)
7. Mokhtar, S. et al. *Detect, Classify, Act: Categorizing Industrial Anomalies with Multi-Modal Large Language Models.* [CVPR Workshops 2025](https://arxiv.org/abs/2505.02626).
8. Chen, S. et al. *Can Multimodal Large Language Models be Guided to Improve Industrial Anomaly Detection?* [arXiv 2501.15795](https://arxiv.org/abs/2501.15795).
9. Jiang, X. et al. *AD-Copilot: A Vision-Language Assistant for Industrial Anomaly Detection via Visual In-context Comparison.* [arXiv 2603.13779](https://arxiv.org/abs/2603.13779) (preprint; not peer-reviewed at time of writing). Code: [github.com/jam-cc/AD-Copilot](https://github.com/jam-cc/AD-Copilot)
10. Arunan, A. et al. *RobustMAD: Evaluating Real-World Robustness of Multimodal Small Language Models for Deployable Anomaly Detection Assistants.* [TMLR 2026](https://arxiv.org/abs/2607.16243). Project page: [robustmad.github.io](https://robustmad.github.io/). Code: [github.com/en-research/RobustMAD](https://github.com/en-research/RobustMAD)
11. Gu, Z. et al. *UniVAD: A Training-free Unified Model for Few-shot Visual Anomaly Detection.* [CVPR 2025](https://arxiv.org/abs/2412.03342). Code: [github.com/FantasticGNU/UniVAD](https://github.com/FantasticGNU/UniVAD)
12. Bergmann, P. et al. *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection.* CVPR 2019 / IJCV 2021.
13. Zou, Y. et al. *SPot-the-Difference Self-supervised Pre-training for Anomaly Detection and Segmentation (VisA).* ECCV 2022.
14. Hendrycks, D. and Dietterich, T. *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations.* ICLR 2019 — source of the severity protocol.
15. Jiang, X. et al. *A Survey of Recent Advances in Industrial Anomaly Detection: From Normal-Only Training to Foundation-Model Priors.* Preprints, 2026 — used for taxonomy.
16. Community paper list: [M-3LAB/awesome-industrial-anomaly-detection](https://github.com/M-3LAB/awesome-industrial-anomaly-detection) — index only; venue labels require verification.

### Verification note

Two venue errors were found in the community list while preparing this document: MMAD is labelled ICLR 2024 in one table when it is ICLR 2025, and AnomalyMoE appears as both AAAI 2025 and AAAI 2026. At least one listed repository URL is a placeholder. Cite from proceedings pages only.

## Appendix A — Mid-term presentation mapping

A 16-slide deck for a 15-minute talk with 5 minutes of questions. Each slide draws from one section above, so the deck can be built without rewriting content.

| # | Slide | Source | Content |
| --- | --- | --- | --- |
| 1 | Title | — | Title, name, roll number, supervisor, date |
| 2 | Motivation | §1 | One factory photo vs one benchmark photo; the deployment gap |
| 3 | Problem statement | §1 | The thesis claim in one sentence |
| 4 | Research questions | §1 | RQ1-RQ4, one line each |
| 5 | Literature: detection era | §2.1 | PatchCore, WinCLIP; MVTec-AD saturated |
| 6 | Literature: MLLM era | §2.2-2.3 | AnomalyGPT → Myriad → VELM → MMAD → Anomaly-OV → AD-Copilot timeline |
| 7 | Literature: robustness | §2.4 | RobustMAD, and exactly where it stops |
| 8 | Gap analysis | §4 | L1-L5 as five short bullets |
| 9 | Datasets | §3 | The dataset table, trimmed to four rows |
| 10 | Corruption suite | §3.2 | Sample images at severities 1, 3, 5 — the most persuasive slide in the deck |
| 11 | Proposed architecture | §5 | The pipeline diagram |
| 12 | Model zoo and hardware | §5.2, §8 | Models, sizes, and the 16 GB constraint stated openly |
| 13 | Metrics | §7 | RDS, relative robustness, seed spread — four rows only |
| 14 | Phased plan | §6 | The six-phase table with GPU hours |
| 15 | Expected contributions | §9 | Four contributions, target venues |
| 16 | Timeline and next steps | §6 | Calendar bar; Phase 0 starting now |
