#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
==============================================================================
Industrial Robustness Mitigation Engine (Phase 3)
==============================================================================
M.Tech Dissertation: Robustness and Reproducibility of Open-Source Small
Multimodal LLMs for Industrial Anomaly Detection

Implements 3 core adaptation & mitigation strategies:
1. Input-Level Test-Time Image Restoration (TTA-IR):
   - Unsharp Masking & Laplacian Edge Boost (Motion Blur)
   - Edge-Preserving Bilateral Filtering (Gaussian Noise)
   - CLAHE Contrast-Limited Adaptive Histogram Equalization (Low-Light)
2. Prompt-Level Defect-Anchored Reasoning (DAR):
   - Noise-Aware Domain Constraints guiding model attention
3. Hybrid Compound Adaptation:
   - Simultaneous TTA-IR + DAR prompting
==============================================================================
"""

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance


# =============================================================================
# STRATEGY 1: INPUT-LEVEL TEST-TIME IMAGE RESTORATION (TTA-IR)
# =============================================================================

def restore_motion_blur(img: Image.Image, strength: float = 1.6) -> Image.Image:
    """
    Recovers high-frequency edge gradients smeared by conveyor motion blur
    using Laplacian high-frequency edge boosting and unsharp masking.
    """
    arr = np.array(img)
    # Convert to YCrCb or LAB to sharpen luminance without distorting color balance
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l_channel, a, b = cv2.split(lab)
    
    # Gaussian blur for unsharp mask
    blurred_l = cv2.GaussianBlur(l_channel, (0, 0), sigmaX=2.0)
    # Unsharp mask formula: L_sharp = L + alpha * (L - blurred_L)
    sharp_l = cv2.addWeighted(l_channel, 1.0 + strength, blurred_l, -strength, 0)
    sharp_l = np.clip(sharp_l, 0, 255).astype(np.uint8)
    
    restored_lab = cv2.merge([sharp_l, a, b])
    restored_rgb = cv2.cvtColor(restored_lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(restored_rgb)


def restore_gaussian_noise(img: Image.Image, d: int = 7, sigma_color: float = 75, sigma_space: float = 75) -> Image.Image:
    """
    Removes thermal Gaussian sensor noise while strictly preserving sharp
    defect boundaries using Edge-Preserving Bilateral Filtering.
    """
    arr = np.array(img)
    # Bilateral filter smooths flat noisy regions while keeping sharp defect edges
    filtered = cv2.bilateralFilter(arr, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    return Image.fromarray(filtered)


def restore_low_light(img: Image.Image, clip_limit: float = 3.0, tile_size: int = 8) -> Image.Image:
    """
    Enhances underexposed factory images using Contrast Limited Adaptive
    Histogram Equalization (CLAHE) on the luminance (L) channel in LAB space.
    """
    arr = np.array(img)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l_channel, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    cl = clahe.apply(l_channel)
    
    enhanced_lab = cv2.merge([cl, a, b])
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(enhanced_rgb)


def restore_specular_glare(img: Image.Image) -> Image.Image:
    """
    Softens severe specular glare hotspots via adaptive highlight compression.
    """
    arr = np.array(img, dtype=np.float32)
    # Compress extreme highlight values
    threshold = 210.0
    mask = arr > threshold
    arr[mask] = threshold + (arr[mask] - threshold) * 0.35
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def restore_defocus_blur(img: Image.Image) -> Image.Image:
    """Sharpen optical defocus blur using high-boost kernel."""
    return restore_motion_blur(img, strength=1.2)


def apply_mitigation_filter(img: Image.Image, corruption_type: str, severity: int) -> Image.Image:
    """
    Applies the optimal Test-Time Image Restoration filter based on the corruption mode.
    """
    if corruption_type == "motion_blur":
        strength = 1.2 if severity <= 2 else (1.6 if severity <= 4 else 2.0)
        return restore_motion_blur(img, strength=strength)
    elif corruption_type == "gaussian_noise":
        sc = 50 if severity <= 2 else (75 if severity <= 4 else 100)
        return restore_gaussian_noise(img, d=7, sigma_color=sc, sigma_space=sc)
    elif corruption_type == "low_light":
        clip = 2.0 if severity <= 2 else (3.0 if severity <= 4 else 4.0)
        return restore_low_light(img, clip_limit=clip)
    elif corruption_type == "specular_glare":
        return restore_specular_glare(img)
    elif corruption_type == "defocus_blur":
        return restore_defocus_blur(img)
    else:
        # Generic mild unsharp mask
        return img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=130, threshold=3))


# =============================================================================
# STRATEGY 2: PROMPT-LEVEL DEFECT-ANCHORED REASONING (DAR)
# =============================================================================

def build_standard_prompt(question_data: dict):
    """Builds standard baseline prompt without noise awareness."""
    ci_data = {str(k).lower(): v for k, v in question_data.items()}
    q_text = ""
    for f in ["question", "query", "text", "prompt"]:
        if f in ci_data and ci_data[f]:
            q_text = str(ci_data[f]).strip()
            break
    if not q_text:
        q_text = "Identify any defect or anomaly present in this industrial object."

    opts_dict = {}
    options_text = ""
    for f in ["options", "choices", "answers"]:
        if f in ci_data and ci_data[f]:
            raw_opts = ci_data[f]
            if isinstance(raw_opts, dict):
                opts_dict = raw_opts
                for k, v in sorted(raw_opts.items()):
                    options_text += f"({k}) {v}\n"
            elif isinstance(raw_opts, list):
                for idx, v in enumerate(raw_opts):
                    k = chr(65 + idx)
                    opts_dict[k] = v
                    options_text += f"({k}) {v}\n"
            break

    prompt = f"{q_text}\n\n{options_text.strip()}\n\nAnswer with the letter of the correct option (e.g., A, B, C, or D). Give only the letter."
    return prompt, q_text, opts_dict


def build_defect_anchored_prompt(question_data: dict):
    """
    Builds an Industrial Noise-Aware Defect-Anchored prompt that guides the
    model's visual attention mechanism to ignore sensor disturbances and focus
    on structural anomalies, cracks, scratches, and missing parts.
    """
    ci_data = {str(k).lower(): v for k, v in question_data.items()}
    q_text = ""
    for f in ["question", "query", "text", "prompt"]:
        if f in ci_data and ci_data[f]:
            q_text = str(ci_data[f]).strip()
            break
    if not q_text:
        q_text = "Identify any defect or anomaly present in this industrial object."

    opts_dict = {}
    options_text = ""
    for f in ["options", "choices", "answers"]:
        if f in ci_data and ci_data[f]:
            raw_opts = ci_data[f]
            if isinstance(raw_opts, dict):
                opts_dict = raw_opts
                for k, v in sorted(raw_opts.items()):
                    options_text += f"({k}) {v}\n"
            elif isinstance(raw_opts, list):
                for idx, v in enumerate(raw_opts):
                    k = chr(65 + idx)
                    opts_dict[k] = v
                    options_text += f"({k}) {v}\n"
            break

    dar_directive = (
        "[INDUSTRIAL QUALITY INSPECTION DIRECTIVE - ROBUST DEFECT ANCHORING]\n"
        "You are an automated industrial quality inspection expert operating on factory sensor feeds.\n"
        "ATTENTION: The sensor image may exhibit industrial disturbances such as conveyor motion blur, "
        "sensor thermal grain, or uneven illumination.\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. Disregard global image noise, motion streaks, or lighting shadows.\n"
        "2. Focus strictly on intrinsic physical anomalies: surface tears, cracks, micro-scratches, "
        "structural deformities, contamination spots, missing components, or dimensional discrepancies.\n"
        "3. Carefully cross-reference the defect patterns described in the question."
    )

    prompt = (
        f"{dar_directive}\n\n"
        f"INSPECTION QUERY:\n{q_text}\n\n"
        f"CANDIDATE OPTIONS:\n{options_text.strip()}\n\n"
        f"Select the letter corresponding to the correct anomaly condition. Give ONLY the letter (e.g. A, B, C, or D)."
    )
    return prompt, q_text, opts_dict
