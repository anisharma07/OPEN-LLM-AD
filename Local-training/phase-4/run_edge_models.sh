#!/bin/bash
set -e
cd "/home/anirudh-sharma/Desktop/M.tech Dissertation/Open-IAD"

echo "=========================================================================="
echo "STARTING 3-MODEL COMPACT & EDGE BENCHMARK SUITE (2,500 SAMPLES EACH)"
echo "1. HuggingFaceTB/SmolVLM-256M-Instruct (~256M params)"
echo "2. vikhyatk/moondream2 (~1.8B params)"
echo "3. google/paligemma2-3b-pt-224 (~3.0B params)"
echo "=========================================================================="

echo ""
echo ">>> [1/3] Launching HuggingFaceTB/SmolVLM-256M-Instruct..."
Local-training/.venv/bin/python Local-training/phase-4/phase4_cross_model_benchmark.py --model-id HuggingFaceTB/SmolVLM-256M-Instruct --sample-size 2500

echo ""
echo ">>> [2/3] Launching vikhyatk/moondream2..."
Local-training/.venv/bin/python Local-training/phase-4/phase4_cross_model_benchmark.py --model-id vikhyatk/moondream2 --sample-size 2500

echo ""
echo ">>> [3/3] Launching google/paligemma2-3b-pt-224..."
Local-training/.venv/bin/python Local-training/phase-4/phase4_cross_model_benchmark.py --model-id google/paligemma2-3b-pt-224 --sample-size 2500

echo ""
echo ">>> Rebuilding comparative analysis datasets & figures..."
Local-training/.venv/bin/python comparative-analysis/scripts/01_build_datasets.py
Local-training/.venv/bin/python comparative-analysis/scripts/02_make_figures.py

echo ""
echo "=========================================================================="
echo "✅ COMPLETED ALL 3 MODELS AND REBUILT ALL DATASETS & FIGURES!"
echo "=========================================================================="
