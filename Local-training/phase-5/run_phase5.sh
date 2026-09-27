#!/bin/bash
set -e
cd "/home/anirudh-sharma/Desktop/M.tech Dissertation/Open-IAD"

echo "=========================================================================="
echo "LAUNCHING PHASE 5: HYBRID PATCHCORE VISION-EXPERT + MLLM BENCHMARK"
echo "=========================================================================="

Local-training/.venv/bin/python Local-training/phase-5/phase5_hybrid_patchcore_mllm.py --sample-size 500 --reset

echo ""
echo "=========================================================================="
echo "✅ PHASE 5 COMPLETE! RESULTS SAVED IN Local-training/phase-5/results/"
echo "=========================================================================="
