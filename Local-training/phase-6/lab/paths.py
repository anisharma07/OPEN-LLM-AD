"""Filesystem layout shared by every lab module."""

from pathlib import Path

PHASE_DIR = Path(__file__).resolve().parent.parent          # Local-training/phase-6
LOCAL_TRAINING_DIR = PHASE_DIR.parent                        # Local-training
REPO_DIR = LOCAL_TRAINING_DIR.parent                         # Open-IAD
MMAD_DIR = REPO_DIR / "MMAD"

WEB_DIR = PHASE_DIR / "web"
WORKFLOW_DIR = PHASE_DIR / "workflows"
TEMPLATE_DIR = WORKFLOW_DIR / "templates"
RUNS_DIR = PHASE_DIR / "runs"
UPLOAD_DIR = PHASE_DIR / "uploads"
CACHE_DIR = PHASE_DIR / "cache"
BLOB_DIR = CACHE_DIR / "blobs"
RESULTS_DIR = PHASE_DIR / "results"

# Earlier phases whose modules the lab reuses verbatim.
PHASE2_DIR = LOCAL_TRAINING_DIR / "phase-2"   # corruptions.py
PHASE3_DIR = LOCAL_TRAINING_DIR / "phase-3"   # mitigations.py

for _d in (WORKFLOW_DIR, TEMPLATE_DIR, RUNS_DIR, UPLOAD_DIR, BLOB_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
