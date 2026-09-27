#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
==============================================================================
Phase 5: Visual Prompting & Attention Guidance Module
==============================================================================
Translates PatchCore spatial anomaly cues into visual prompts:
1. High-visibility red bounding box + tag label
2. Semi-transparent attention overlay
3. Grounded inspection prompt directing MLLM attention to the defect region.
==============================================================================
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

def apply_visual_prompt_overlay(pil_img, bbox, anomaly_score=1.0, label="DEFECT CANDIDATE"):
    """
    Overlays a red bounding box and attention highlight onto the image.
    bbox: (ymin, xmin, ymax, xmax)
    """
    if bbox is None:
        return pil_img.copy()

    ymin, xmin, ymax, xmax = bbox
    orig_w, orig_h = pil_img.size

    # Base copy
    annotated = pil_img.copy().convert("RGBA")
    overlay = Image.new("RGBA", annotated.size, (255, 255, 255, 0))
    draw_overlay = ImageDraw.Draw(overlay)

    # Line thickness scaled to resolution
    lw = max(3, int(min(orig_w, orig_h) / 180))

    # 1. Semi-transparent red wash over defect region
    draw_overlay.rectangle([xmin, ymin, xmax, ymax], fill=(255, 0, 0, 45))

    # 2. Solid bounding box outline
    for i in range(lw):
        draw_overlay.rectangle([xmin - i, ymin - i, xmax + i, ymax + i], outline=(255, 20, 20, 230))

    # 3. Label tag
    badge_h = max(24, int(orig_h * 0.035))
    badge_w = int(len(label) * badge_h * 0.65)
    badge_ymin = max(0, ymin - badge_h)
    draw_overlay.rectangle([xmin, badge_ymin, xmin + badge_w, ymin], fill=(220, 10, 10, 240))
    draw_overlay.text((xmin + 8, badge_ymin + 3), label, fill=(255, 255, 255, 255))

    # Composite
    result = Image.alpha_composite(annotated, overlay).convert("RGB")
    return result

def format_hybrid_prompt(question_text, options_dict, has_visual_cue=True):
    """
    Constructs the grounded multimodal prompt referencing the highlighted visual cue.
    """
    opt_str = "\n".join([f"({k}) {v}" for k, v in sorted(options_dict.items())])

    if has_visual_cue:
        prompt = (
            "A vision-expert anomaly detection model has localized the potential flaw area inside the RED bounding box.\n"
            f"Focus your visual inspection on the highlighted red region:\n\n"
            f"{question_text}\n\n"
            f"{opt_str}\n\n"
            "Answer with the letter of the correct option (e.g., A, B, C, or D). Give only the letter."
        )
    else:
        prompt = (
            f"{question_text}\n\n"
            f"{opt_str}\n\n"
            "Answer with the letter of the correct option (e.g., A, B, C, or D). Give only the letter."
        )

    return prompt
