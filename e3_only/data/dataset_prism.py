"""Dataset for PRISM: image + points + the frozen region partition, kept aligned.

The partition is cached in the *original* image frame, so every geometric
augmentation applied to the image must be applied to it identically, with nearest
interpolation. Getting that wrong is silent -- the loss still decreases, it just
enforces homogeneity over the wrong pixels -- so the transforms are shared by
construction here rather than reimplemented per tensor.

Propagation is computed *after* augmentation, from the transformed region map and
the transformed points, which keeps it correct under flips, rotations and crops
without needing an inverse transform anywhere.

The point-only contract is unchanged and still enforced: a training manifest
carrying a ``mask`` key raises.
"""
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..core.regions import propagate_points_np


class PairAugmentR:
    """Weak/strong pair with an auxiliary integer map carried along.

    Weak and strong differ only photometrically -- the geometric transform is
    drawn once and applied to both -- so teacher and student outputs are pixel
    aligned and no target ever needs to be warped between them.
    """

    def __init__(self, image_size: int, weak_brightness: float = 0.05,
                 strong_brightness: float = 0.25, strong_contrast: float = 0.25,
                 strong_noise_std: float = 0.03, multi_scale: bool = False,
                 crop_scale_range=(0.5, 1.0)):
        self.image_size = image_size
        self.weak_brightness = weak_brightness
        self.strong_brightness = strong_brightness
        self.strong_contrast = strong_contrast
        self.strong_noise_std = strong_noise_std
        self.multi_scale = multi_scale
        self.crop_scale_range = crop_scale_range

    @staticmethod
    def _geom(image, aux, points, hflip, vflip, rot):
        h, w = image.shape[:2]
        if hflip:
            image = image[:, ::-1].copy()
            aux = aux[:, ::-1].copy()
            if len(points):
                points[:, 0] = (w - 1) - points[:, 0]
        if vflip:
            image = image[::-1, :].copy()
            aux = aux[::-1, :].copy()
            if len(points):
                points[:, 1] = (h - 1) - points[:, 1]
        for _ in range(int(rot) % 4):
            image = np.rot90(image, 1).copy()
            aux = np.rot90(aux, 1).copy()
            if len(points):
                ox = points[:, 0].copy()
                oy = points[:, 1].copy()
                points[:, 0] = oy
                points[:, 1] = (w - 1) - ox
            h, w = w, h
        return image, aux, points

    def _color(self, image, brightness, contrast, noise_std):
        x = image.astype(np.float32) / 255.0
        if brightness > 0:
            x = x + random.uniform(-brightness, brightness)
        if contrast > 0:
            c = random.uniform(1.0 - contrast, 1.0 + contrast)
            x = (x - 0.5) * c + 0.5
        if noise_std > 0:
            x = x + np.random.normal(0.0, noise_std, x.shape).astype(np.float32)
        return (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)

    def _crop(self, image, aux, points):
        h, w = image.shape[:2]
        s = random.uniform(*self.crop_scale_range)
        ch, cw = max(1, int(h * s)), max(1, int(w * s))
        if len(points):
            px, py = points[random.randrange(len(points)), :2]
            y0 = random.randint(max(0, int(py) - ch + 1), min(h - ch, max(0, int(py))))
            x0 = random.randint(max(0, int(px) - cw + 1), min(w - cw, max(0, int(px))))
        else:
            y0 = random.randint(0, max(0, h - ch))
            x0 = random.randint(0, max(0, w - cw))
        img = cv2.resize(image[y0:y0 + ch, x0:x0 + cw], (self.image_size, self.image_size),
                         interpolation=cv2.INTER_LINEAR)
        a = cv2.resize(aux[y0:y0 + ch, x0:x0 + cw], (self.image_size, self.image_size),
                       interpolation=cv2.INTER_NEAREST)
        p = points.copy()
        if len(p):
            p[:, 0] = (p[:, 0] - x0) * (self.image_size / cw)
            p[:, 1] = (p[:, 1] - y0) * (self.image_size / ch)
            keep = ((p[:, 0] >= 0) & (p[:, 0] < self.image_size) &
                    (p[:, 1] >= 0) & (p[:, 1] < self.image_size))
            p = p[keep]
        return img, a, p

    def __call__(self, image: np.ndarray, aux: np.ndarray, points: np.ndarray):
        if self.multi_scale and random.random() < 0.5:
            image, aux, points = self._crop(image, aux, points)
        hflip = random.random() < 0.5
        vflip = random.random() < 0.5
        rot = random.randint(0, 3)
        p = points.copy().astype(np.float32)
        weak, aux, p = self._geom(image.copy(), aux.copy(), p, hflip, vflip, rot)
        strong = self._color(weak, self.strong_brightness, self.strong_contrast,
                             self.strong_noise_std)
        weak = self._color(weak, self.weak_brightness, 0.0, 0.0)
        return weak, strong, aux, p


def _read_image(path: str, size: int) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[:2] != (size, size):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    return img


class PrismDataset(Dataset):
    def __init__(self, manifest: str, image_size: int, training: bool,
                 region_npz: Optional[str] = None, augment: Optional[PairAugmentR] = None,
                 num_classes: int = 17):
        self.items = json.loads(Path(manifest).read_text())
        self.image_size = image_size
        self.training = training
        self.augment = augment
        self.num_classes = num_classes
        if training:
            leaked = [it.get("id") for it in self.items if "mask" in it]
            if leaked:
                raise ValueError(
                    f"training manifest {manifest} carries dense masks for "
                    f"{len(leaked)} items (e.g. {leaked[0]}). Point-only contract violated.")

        self.regions = None
        if region_npz:
            z = np.load(region_npz, allow_pickle=True)
            self.regions = {str(k): i for i, k in enumerate(z["ids"])}
            self._region_data = z["regions"]
            # n_sam splits the partition: ids < n_sam are SAM masks, ids >= n_sam are
            # connected components of whatever SAM left over. Training does not need
            # the split (the point-conflict exclusion already drops the bad filler),
            # but inference does -- there are no points at test time, so the
            # region vote has no other way to tell the two apart.
            if "n_sam" not in z.files:
                raise KeyError(
                    f"region cache {region_npz} has a partition but no 'n_sam' "
                    f"array, so SAM masks cannot be told apart from filler "
                    f"components. This is not benign: _region_vote gates on "
                    f"`region < n_sam`, which is false everywhere at n_sam=0, so "
                    f"--region-vote would degrade to plain argmax while still "
                    f"printing region_vote=True. Rebuild with "
                    f"tools/build_region_cache.py.")
            self._n_sam = z["n_sam"]
            missing = [it.get("id") for it in self.items
                       if str(it.get("id")) not in self.regions]
            if missing:
                raise ValueError(
                    f"region cache {region_npz} is missing {len(missing)} manifest ids "
                    f"(e.g. {missing[0]}). Rebuild it with tools/build_region_cache.py.")

    def __len__(self):
        return len(self.items)

    def _region_for(self, item, h, w) -> np.ndarray:
        if self.regions is None:
            return np.zeros((h, w), np.int32)
        r = self._region_data[self.regions[str(item.get("id"))]].astype(np.int32)
        if r.shape != (h, w):
            r = cv2.resize(r, (w, h), interpolation=cv2.INTER_NEAREST)
        return r

    def _n_sam_for(self, item) -> int:
        """SAM/filler split point for this image's partition.

        0 only when there is no partition at all -- a cache that HAS one is
        rejected at construction if it lacks n_sam, because n_sam=0 makes
        ``_region_vote``'s ``region < n_sam`` gate false everywhere and turns
        --region-vote into plain argmax without saying so.
        """
        if self.regions is None:
            return 0                      # no partition at all: nothing to split
        return int(self._n_sam[self.regions[str(item.get("id"))]])

    def __getitem__(self, idx: int) -> Dict:
        item = self.items[idx]
        s = self.image_size
        image = _read_image(item["image"], s)
        region = self._region_for(item, s, s)

        points = np.asarray(item.get("points", []), dtype=np.float32).reshape(-1, 3)
        if len(points):
            points[:, 0] *= s / item.get("width", s)
            points[:, 1] *= s / item.get("height", s)

        if self.training and self.augment is not None:
            weak, strong, region, points = self.augment(image, region, points)
        else:
            weak, strong = image, image

        prop, conflict = propagate_points_np(region, points, self.num_classes)

        out = {
            "image_weak": torch.from_numpy(np.ascontiguousarray(weak)).permute(2, 0, 1).float() / 255.0,
            "image_strong": torch.from_numpy(np.ascontiguousarray(strong)).permute(2, 0, 1).float() / 255.0,
            "points": torch.from_numpy(points).float(),
            "region": torch.from_numpy(region.astype(np.int64)),
            "prop": torch.from_numpy(prop.astype(np.int64)),
            "conflict": torch.from_numpy(conflict),
            "n_sam": int(self._n_sam_for(item)),
            "image_id": str(item.get("id", idx)),
        }
        if not self.training and item.get("mask"):
            m = cv2.imread(item["mask"], cv2.IMREAD_GRAYSCALE)
            if m is None:
                raise FileNotFoundError(item["mask"])
            if m.shape != (s, s):
                m = cv2.resize(m, (s, s), interpolation=cv2.INTER_NEAREST)
            out["mask"] = torch.from_numpy(m).long()
        return out


def collate_prism(batch: List[Dict]) -> Dict:
    out = {
        "image_weak": torch.stack([x["image_weak"] for x in batch]),
        "image_strong": torch.stack([x["image_strong"] for x in batch]),
        "region": torch.stack([x["region"] for x in batch]),
        "prop": torch.stack([x["prop"] for x in batch]),
        "conflict": torch.stack([x["conflict"] for x in batch]),
        "points": [x["points"] for x in batch],
        "n_sam": torch.tensor([x["n_sam"] for x in batch], dtype=torch.long),
        "image_id": [x["image_id"] for x in batch],
    }
    if "mask" in batch[0]:
        out["mask"] = torch.stack([x["mask"] for x in batch])
    return out
