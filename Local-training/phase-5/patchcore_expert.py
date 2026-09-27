#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
==============================================================================
Phase 5: PatchCore Vision Expert Module
==============================================================================
Extracts patch-level feature representations from intermediate CNN layers
(Layer 2 + Layer 3) and compares them against normal coreset memory banks
to generate high-resolution pixel-level anomaly heatmaps and defect bounding boxes.
==============================================================================
"""

import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from scipy.ndimage import gaussian_filter

class PatchCoreExpert:
    def __init__(self, device=None, backbone="resnet50"):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if backbone == "resnet50":
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif backbone == "wide_resnet50_2":
            self.model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.DEFAULT)
        else:
            self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        self.model.to(self.device).eval()

        self.features = []
        def hook(module, input, output):
            self.features.append(output)

        self.model.layer2[-1].register_forward_hook(hook)
        self.model.layer3[-1].register_forward_hook(hook)

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.memory_banks = {}

    def extract_patches(self, img_path):
        self.features.clear()
        if isinstance(img_path, Image.Image):
            img = img_path.convert("RGB")
        else:
            img = Image.open(img_path).convert("RGB")

        orig_w, orig_h = img.size
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            _ = self.model(tensor)

        l2 = self.features[0]
        l3 = F.interpolate(self.features[1], size=l2.shape[-2:], mode="bilinear", align_corners=False)
        concat = torch.cat([l2, l3], dim=1)
        pooled = F.avg_pool2d(concat, 3, stride=1, padding=1)
        b, c, h, w = pooled.shape
        patches = pooled.permute(0, 2, 3, 1).reshape(-1, c)

        return patches, (h, w), (orig_w, orig_h)

    def build_memory_bank(self, category_key, normal_image_paths, max_patches=2000):
        if category_key in self.memory_banks:
            return self.memory_banks[category_key]

        all_patches = []
        # Sample up to 8 normal images to keep memory bank lean & ultra-fast
        sample_paths = normal_image_paths[:8]
        for p in sample_paths:
            patches, _, _ = self.extract_patches(p)
            all_patches.append(patches)

        if not all_patches:
            return None

        combined = torch.cat(all_patches, dim=0)
        # Subsample to max_patches
        if combined.shape[0] > max_patches:
            idx = torch.randperm(combined.shape[0])[:max_patches]
            combined = combined[idx]

        self.memory_banks[category_key] = combined

        # Calibrate category normal threshold from normal validation samples
        good_maxes = []
        for p in sample_paths[:4]:
            patches, _, _ = self.extract_patches(p)
            dists = torch.cdist(patches, combined)
            min_dists, _ = torch.min(dists, dim=1)
            good_maxes.append(min_dists.max().item())

        if not hasattr(self, "category_thresholds"):
            self.category_thresholds = {}
        self.category_thresholds[category_key] = max(good_maxes) * 1.12 if good_maxes else 28.0

        return combined

    def detect_anomaly(self, img_path, normal_image_paths, category_key=None, threshold=None):
        """
        Computes PatchCore anomaly heatmap and returns bounding box and anomaly score.
        """
        if category_key is None:
            category_key = "default"

        bank = self.build_memory_bank(category_key, normal_image_paths)
        if bank is None:
            return None, 0.0, None

        tau = getattr(self, "category_thresholds", {}).get(category_key, 28.0)
        test_patches, (gh, gw), (orig_w, orig_h) = self.extract_patches(img_path)

        # Nearest neighbor L2 distance
        dists = torch.cdist(test_patches, bank)
        min_dists, _ = torch.min(dists, dim=1)
        score_map = min_dists.reshape(1, 1, gh, gw)
        # Interpolate to 256x256 for blazing fast Gaussian smoothing and clustering
        score_map = F.interpolate(score_map, size=(256, 256), mode="bilinear", align_corners=False)
        score_map = score_map.squeeze().cpu().numpy()
        score_map = gaussian_filter(score_map, sigma=2)

        max_val = float(score_map.max())
        image_anomaly_score = max_val / (tau + 1e-6)

        # Threshold against calibrated normal threshold
        mask = score_map > tau
        y_indices, x_indices = np.where(mask)
        bbox = None
        if len(y_indices) > 10:  # Minimum cluster size at 256x256
            ymin_256, ymax_256 = int(y_indices.min()), int(y_indices.max())
            xmin_256, xmax_256 = int(x_indices.min()), int(x_indices.max())
            # Add padding
            pad_h = int((ymax_256 - ymin_256) * 0.1)
            pad_w = int((xmax_256 - xmin_256) * 0.1)
            ymin_256 = max(0, ymin_256 - pad_h)
            ymax_256 = min(256, ymax_256 + pad_h)
            xmin_256 = max(0, xmin_256 - pad_w)
            xmax_256 = min(256, xmax_256 + pad_w)

            # Scale to original image coordinates
            scale_y = orig_h / 256.0
            scale_x = orig_w / 256.0
            bbox = (
                int(ymin_256 * scale_y),
                int(xmin_256 * scale_x),
                int(ymax_256 * scale_y),
                int(xmax_256 * scale_x),
            )

        return score_map, image_anomaly_score, bbox
