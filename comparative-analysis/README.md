# comparative-analysis

Comparison of this dissertation's measured results against the MMAD benchmark paper
(Jiang et al., ICLR 2025), with the figures generated from the raw run artefacts.

**Start here:** [`results/comparative_report.md`](results/comparative_report.md)

```
data/       mmad_paper_table2.csv              transcribed from the paper (Table 2)
            mmad_paper_table3_shot_setting.csv transcribed from the paper (Table 3)
            thiswork_*.csv                     generated from Local-training/ artefacts
            merged_leaderboard.csv             the two sources on one protocol
scripts/    01_build_datasets.py               artefacts -> data/
            02_make_figures.py                 data      -> figures/
            03_manuscript_audit.py             draft vs artefacts -> results/tables/
figures/    fig1..fig8 .png                    200 dpi, thesis-ready
results/    comparative_report.md              the written analysis
            tables/                            the table view behind each figure
```

Rebuild everything:

```bash
python3 scripts/01_build_datasets.py
python3 scripts/02_make_figures.py
python3 scripts/03_manuscript_audit.py
```

Source artefacts consumed (read-only, none modified):
`Local-training/phase-1/results/phase1_{manifest.json,results.jsonl}`,
`Local-training/phase-4/results/phase4_manifest.json` + the three per-model JSONL runs,
`Local-training/benchmark_5k_results.jsonl`.
