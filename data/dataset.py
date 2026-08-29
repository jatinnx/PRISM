"""Dataset for the point-only framework.

Contract: the **training** manifest must contain no ``mask`` key — enforced in
the constructor so a leak fails fast. Dense masks appear only in validation
manifests (already remapped to 0..16 at write time by make_manifests.py).
"""
import json
import random
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class PairAugment:
    """Weak/strong augmentation pair for teacher (weak) / student (strong).

    With ``multi_scale=True``, a random crop is taken at a random scale
    (0.5–1.0 of the image) and resized back to ``image_size``, so the model
    sees objects at multiple resolutions.  The crop is guaranteed to contain
    at least one labelled point when possible, so the sparse supervision
    stays within the crop.
    """

    def __init__(self, image_size: int, weak_brightness: float, strong_brightness: float,
                 strong_contrast: float, strong_noise_std: float,
                 multi_scale: bool = False, crop_scale_range: tuple = (0.5, 1.0)):
        self.image_size = image_size
        self.weak_brightness = weak_brightness
        self.strong_brightness = strong_brightness
        self.strong_contrast = strong_contrast
        self.strong_noise_std = strong_noise_std
        self.multi_scale = multi_scale
        self.crop_scale_range = crop_scale_range

    def _geom(self, image: np.ndarray, points: np.ndarray, hflip: bool, vflip: bool, rot: int):
        h, w = image.shape[:2]
        if hflip:
            image = image[:, ::-1].copy()
            if len(points):
                points[:, 0] = (w - 1) - points[:, 0]
        if vflip:
            image = image[::-1, :].copy()
            if len(points):
                points[:, 1] = (h - 1) - points[:, 1]
        for _ in range(int(rot) % 4):
            image = np.rot90(image, 1).copy()
            if len(points):
                old_x = points[:, 0].copy()
                old_y = points[:, 1].copy()
                points[:, 0] = old_y
                points[:, 1] = (w - 1) - old_x
                h, w = w, h
        return image, points

    def _color(self, image: np.ndarray, brightness: float, contrast: float, noise_std: float):
        x = image.astype(np.float32) / 255.0
        if brightness > 0:
            x = x + random.uniform(-brightness, brightness)
        if contrast > 0:
            c = random.uniform(1.0 - contrast, 1.0 + contrast)
            x = (x - 0.5) * c + 0.5
        if noise_std > 0:
            x = x + np.random.normal(0.0, noise_std, x.shape).astype(np.float32)
        x = np.clip(x, 0.0, 1.0)
        return (x * 255.0).astype(np.uint8)

    def _multi_scale_crop(self, image: np.ndarray, points: np.ndarray):
        """Random crop at a random scale, then resize back to image_size.

        Ensures the crop contains at least one labelled point when possible,
        so supervision isn't lost.  Returns (cropped_resized_image, transformed_points).
        """
        h, w = image.shape[:2]
        # Pick random scale in [crop_scale_range[0], crop_scale_range[1]]
        scale = random.uniform(*self.crop_scale_range)
        crop_h = max(1, int(h * scale))
        crop_w = max(1, int(w * scale))

        # Try to place crop so it contains at least one point
        if len(points) > 0:
            # Pick a random point to include
            px, py = points[random.randint(0, len(points) - 1), :2]
            # Random offset such that (px, py) is inside the crop
            min_y = max(0, int(py) - crop_h + 1)
            max_y = min(h - crop_h, int(py))
            min_x = max(0, int(px) - crop_w + 1)
            max_x = min(w - crop_w, int(px))
            if min_y <= max_y and min_x <= max_x:
                y0 = random.randint(min_y, max_y)
                x0 = random.randint(min_x, max_x)
            else:
                y0 = random.randint(0, max(0, h - crop_h))
                x0 = random.randint(0, max(0, w - crop_w))
        else:
            y0 = random.randint(0, max(0, h - crop_h))
            x0 = random.randint(0, max(0, w - crop_w))

        y1 = y0 + crop_h
        x1 = x0 + crop_w
        crop = image[y0:y1, x0:x1].copy()

        # Resize crop back to image_size
        resized = cv2.resize(crop, (self.image_size, self.image_size),
                            interpolation=cv2.INTER_LINEAR)

        # Transform points: shift by crop origin, then scale by resize factor
        new_points = points.copy()
        if len(new_points) > 0:
            new_points[:, 0] = (new_points[:, 0] - x0) * (self.image_size / crop_w)
            new_points[:, 1] = (new_points[:, 1] - y0) * (self.image_size / crop_h)
            # Filter out points that fell outside the crop
            mask = ((new_points[:, 0] >= 0) & (new_points[:, 0] < self.image_size) &
                    (new_points[:, 1] >= 0) & (new_points[:, 1] < self.image_size))
            new_points = new_points[mask]

        return resized, new_points

    def __call__(self, image: np.ndarray, points: np.ndarray):
        # Multi-scale crop FIRST (before geometric + color augmentations)
        if self.multi_scale and random.random() < 0.5:
            image, points = self._multi_scale_crop(image, points)
        else:
            # Resize to target size if not cropping (input may be > image_size)
            h, w = image.shape[:2]
            if h != self.image_size or w != self.image_size:
                image = cv2.resize(image, (self.image_size, self.image_size),
                                   interpolation=cv2.INTER_LINEAR)

        hflip = random.random() < 0.5
        vflip = random.random() < 0.5
        rot = random.randint(0, 3)
        p = points.copy().astype(np.float32)
        weak, p = self._geom(image.copy(), p, hflip, vflip, rot)
        strong = weak.copy()
        weak = self._color(weak, self.weak_brightness, 0.0, 0.0)
        strong = self._color(strong, self.strong_brightness, self.strong_contrast, self.strong_noise_std)
        return weak, strong, p


def _read_image(path: str, size: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    return image


class PointOnlyDataset(Dataset):
    def __init__(self, manifest: str, image_size: int, training: bool, augment: PairAugment = None):
        self.items = json.loads(Path(manifest).read_text())
        self.image_size = image_size
        self.training = training
        self.augment = augment
        if training:
            leaked = [it.get("id") for it in self.items if "mask" in it]
            if leaked:
                raise ValueError(
                    f"training manifest {manifest} contains dense masks for {len(leaked)} "
                    f"items (e.g. {leaked[0]}). Point-only contract violated.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        item = self.items[idx]
        image = _read_image(item["image"], self.image_size)
        points = np.asarray(item.get("points", []), dtype=np.float32).reshape(-1, 3)
        if len(points):
            points[:, 0] = points[:, 0] * (self.image_size / item.get("width", self.image_size))
            points[:, 1] = points[:, 1] * (self.image_size / item.get("height", self.image_size))
        if self.training and self.augment is not None:
            weak, strong, points = self.augment(image, points)
        else:
            weak, strong = image, image.copy()

        out = {
            "image_weak": torch.from_numpy(weak).permute(2, 0, 1).float() / 255.0,
            "image_strong": torch.from_numpy(strong).permute(2, 0, 1).float() / 255.0,
            "points": torch.from_numpy(points).float(),
            "image_id": item.get("id", str(idx)),
        }
        if not self.training and item.get("mask"):
            mask = cv2.imread(item["mask"], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(item["mask"])
            mask = cv2.resize(mask, (self.image_size, self.image_size),
                              interpolation=cv2.INTER_NEAREST)
            out["mask"] = torch.from_numpy(mask).long()
        return out


def collate_points(batch: List[Dict]) -> Dict:
    out = {
        "image_weak": torch.stack([x["image_weak"] for x in batch]),
        "image_strong": torch.stack([x["image_strong"] for x in batch]),
        "points": [x["points"] for x in batch],
        "image_id": [x["image_id"] for x in batch],
    }
    if "mask" in batch[0]:
        out["mask"] = torch.stack([x["mask"] for x in batch])
    return out
