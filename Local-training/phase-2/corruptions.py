#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
==============================================================================
Industrial Image Corruption Engine (Phase 2)
==============================================================================
M.Tech Dissertation: Robustness and Reproducibility of Open-Source Small
Multimodal LLMs for Industrial Anomaly Detection

Implements 7 realistic industrial imaging corruptions across 5 severity levels:
1. Gaussian Noise (Sensor thermal noise in low-cost machine vision cameras)
2. Motion Blur (Conveyor belt motion / robotic arm vibrations)
3. Low-Light / Underexposure (Poor factory illumination / power dips)
4. Specular Glare (Harsh directional reflections on metallic / glossy parts)
5. Defocus Blur (Camera depth-of-field drift or optical misfocus)
6. Perspective Tilt (Camera mounting misalignment / off-axis angle)
7. Compression Artifacts (Edge bandwidth throttling / lossy JPEG transmission)
==============================================================================
"""

import io
import os
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from scipy.ndimage import convolve
import matplotlib.pyplot as plt

CORRUPTION_TYPES = [
    "gaussian_noise",
    "motion_blur",
    "low_light",
    "specular_glare",
    "defocus_blur",
    "perspective_tilt",
    "compression",
]

CORRUPTION_DISPLAY_NAMES = {
    "gaussian_noise": "Gaussian Sensor Noise",
    "motion_blur": "Conveyor Motion Blur",
    "low_light": "Low-Light / Underexposure",
    "specular_glare": "Specular Glare / Reflection",
    "defocus_blur": "Defocus Blur (Optical Drift)",
    "perspective_tilt": "Perspective Off-Axis Tilt",
    "compression": "JPEG Edge Compression",
}


def apply_gaussian_noise(img: Image.Image, severity: int) -> Image.Image:
    """Additive zero-mean Gaussian sensor noise."""
    assert 1 <= severity <= 5
    stds = [12, 25, 45, 70, 105]
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, stds[severity - 1], arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def apply_motion_blur(img: Image.Image, severity: int) -> Image.Image:
    """Linear diagonal motion blur simulating conveyor belt movement."""
    assert 1 <= severity <= 5
    ks = [5, 9, 15, 23, 31]
    k = ks[severity - 1]
    kernel = np.zeros((k, k), dtype=np.float32)
    np.fill_diagonal(kernel, 1.0 / k)
    arr = np.array(img, dtype=np.float32)
    for c in range(3):
        arr[:, :, c] = convolve(arr[:, :, c], kernel)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def apply_low_light(img: Image.Image, severity: int) -> Image.Image:
    """Nonlinear gamma attenuation and brightness drop."""
    assert 1 <= severity <= 5
    gammas = [1.2, 1.4, 1.7, 2.1, 2.6]
    factors = [0.85, 0.70, 0.55, 0.40, 0.25]
    g = gammas[severity - 1]
    f = factors[severity - 1]
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.clip((arr ** g) * f * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def apply_specular_glare(img: Image.Image, severity: int) -> Image.Image:
    """Localized high-intensity Gaussian specular glare hotspot."""
    assert 1 <= severity <= 5
    radii = [40, 60, 85, 115, 150]
    intensities = [80, 120, 160, 200, 240]
    w, h = img.size
    cx, cy = w // 2, h // 2
    r = radii[severity - 1]
    intensity = intensities[severity - 1]
    y, x = np.ogrid[:h, :w]
    dist_sq = (x - cx) ** 2 + (y - cy) ** 2
    glare = intensity * np.exp(-dist_sq / (2 * (r ** 2)))
    arr = np.array(img, dtype=np.float32)
    for c in range(3):
        arr[:, :, c] += glare
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def apply_defocus_blur(img: Image.Image, severity: int) -> Image.Image:
    """Camera depth-of-field misfocus."""
    assert 1 <= severity <= 5
    radii = [1.5, 3.0, 5.0, 7.5, 11.0]
    return img.filter(ImageFilter.GaussianBlur(radius=radii[severity - 1]))


def apply_perspective_tilt(img: Image.Image, severity: int) -> Image.Image:
    """Projective quadrilateral warp simulating camera off-axis tilt."""
    assert 1 <= severity <= 5
    tilt_fracs = [0.04, 0.08, 0.13, 0.18, 0.25]
    w, h = img.size
    d = tilt_fracs[severity - 1] * w
    quad = (d, 0, 0, h, w, h, w - d, 0)
    return img.transform((w, h), Image.Transform.QUAD, quad, resample=Image.Resampling.BICUBIC)


def apply_compression(img: Image.Image, severity: int) -> Image.Image:
    """Lossy JPEG DCT compression artifacts."""
    assert 1 <= severity <= 5
    qualities = [65, 45, 30, 18, 8]
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=qualities[severity - 1])
    buf.seek(0)
    return Image.open(buf).convert("RGB")


DISPATCHER = {
    "gaussian_noise": apply_gaussian_noise,
    "motion_blur": apply_motion_blur,
    "low_light": apply_low_light,
    "specular_glare": apply_specular_glare,
    "defocus_blur": apply_defocus_blur,
    "perspective_tilt": apply_perspective_tilt,
    "compression": apply_compression,
}


def apply_corruption(img: Image.Image, corruption_type: str, severity: int) -> Image.Image:
    """
    Apply specified industrial corruption to PIL image.
    If severity == 0, returns the unmodified clean image.
    """
    if severity == 0:
        return img.copy()
    if corruption_type not in DISPATCHER:
        raise ValueError(f"Unknown corruption '{corruption_type}'. Must be one of {list(DISPATCHER.keys())}")
    return DISPATCHER[corruption_type](img, severity)


def generate_corruption_preview_grid(sample_image_path: str, output_path: str):
    """
    Generate high-resolution 7x6 publication preview grid:
    Rows: 7 Corruptions
    Columns: Clean (Sev 0), Sev 1, Sev 2, Sev 3, Sev 4, Sev 5
    """
    clean_img = Image.open(sample_image_path).convert("RGB")
    clean_img.thumbnail((400, 400), Image.Resampling.LANCZOS)

    n_rows = len(CORRUPTION_TYPES)
    n_cols = 6  # Clean, S1, S2, S3, S4, S5

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 21))
    fig.suptitle("Industrial Corruption Taxonomy & Severity Calibration (MMAD Industrial Benchmark)", fontsize=16, fontweight="bold", y=0.995)

    col_titles = ["Clean (Sev 0)", "Severity 1 (Mild)", "Severity 2", "Severity 3 (Moderate)", "Severity 4", "Severity 5 (Severe)"]

    for i, c_type in enumerate(CORRUPTION_TYPES):
        c_name = CORRUPTION_DISPLAY_NAMES[c_type]
        for j in range(n_cols):
            ax = axes[i, j]
            sev = j
            corrupted_img = apply_corruption(clean_img, c_type, sev)
            ax.imshow(corrupted_img)
            ax.axis("off")

            if i == 0:
                ax.set_title(col_titles[j], fontsize=11, fontweight="bold", pad=8)

            if j == 0:
                ax.text(-0.15, 0.5, c_name, transform=ax.transAxes, fontsize=11, fontweight="bold",
                        ha="right", va="center", rotation=0,
                        bbox=dict(boxstyle="round,pad=0.4", facecolor="#ecf0f1", edgecolor="gray"))

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"✅ Corruption preview grid saved: {output_path}")


if __name__ == "__main__":
    import sys
    test_img_path = "/home/anirudh-sharma/Desktop/M.tech Dissertation/Open-IAD/MMAD/DS-MVTec/DS-MVTec/metal_nut/image/bent/007.png"
    out = "corruption_preview_test.png"
    if len(sys.argv) > 1:
        test_img_path = sys.argv[1]
    if len(sys.argv) > 2:
        out = sys.argv[2]
    generate_corruption_preview_grid(test_img_path, out)
