"""Paired panorama/BEV and overhead images for single-image SkyDiffusion."""

import csv
import os
import pickle
import random
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset

from cruvedbev import CruvedBEV


class SkyDataset(Dataset):
    """Return HWC RGB ``hint`` in [0, 1] and ``jpg`` in [-1, 1].

    Headered CSV/TSV files use ``name,source,target,prompt`` (prompt is optional).
    Legacy CVACT rows use name/source/target[/prompt]. Headerless CVUSA rows
    retain their original name/target/source order when they have three fields,
    or name/source/target/prompt when they have four fields.

    Relative paths use data_root when provided. Otherwise, paths are resolved
    relative to the manifest, with a current-directory fallback for old lists.
    Pickled conditions must already be BEV arrays: HWC RGB uint8 [0, 255] or
    floating point [0, 1]. ``source_mode='bev'`` also accepts ordinary BEV images.
    image_size follows OpenCV's existing (width, height) convention.
    """

    def __init__(self, data_file, is_cvusa=False, prompt="", image_size=(512, 512),
                 drop_prompt_ratio=0, dataset_name="CVACT", data_root=None,
                 source_mode="panorama", rotation_degrees=90.0):
        if source_mode not in ("panorama", "bev"):
            raise ValueError("source_mode must be 'panorama' or 'bev'")
        if dataset_name not in ("CVACT", "CVUSA"):
            raise ValueError("dataset_name must be CVACT or CVUSA")
        if len(image_size) != 2 or any(int(size) != size or size <= 0 for size in image_size):
            raise ValueError("image_size must contain positive integer width and height")
        if not 0 <= drop_prompt_ratio <= 1:
            raise ValueError("drop_prompt_ratio must be in [0, 1]")
        if not np.isfinite(rotation_degrees):
            raise ValueError("rotation_degrees must be finite")
        self.image_size = tuple(int(size) for size in image_size)
        self.drop_prompt_ratio = drop_prompt_ratio
        self.dataset_name = dataset_name
        self.source_mode = source_mode
        self.rotation_degrees = float(rotation_degrees)
        self.data_file = Path(data_file).expanduser().resolve()
        self.data_root = Path(data_root).expanduser().resolve() if data_root else None
        self.data = self._read_manifest(bool(is_cvusa), str(prompt).strip())

    def _resolve_image_path(self, value):
        path = Path(value).expanduser()
        if path.is_absolute():
            return str(path)
        if self.data_root is not None:
            return str(self.data_root / path)
        manifest_path = self.data_file.parent / path
        if manifest_path.exists():
            return str(manifest_path)
        # Original examples use ./examples/... relative to the project root.
        if path.exists():
            return str(path.resolve())
        return str(manifest_path)

    def _read_manifest(self, is_cvusa, default_prompt):
        with self.data_file.open("r", encoding="utf-8-sig", newline="") as handle:
            lines = handle.readlines()
        first_line = next((line for line in lines if line.strip()), "")
        if not first_line:
            raise ValueError("Manifest is empty: {}".format(self.data_file))
        delimiter = "\t" if "\t" in first_line else ","
        reader = csv.reader(lines, delimiter=delimiter)
        records = [(reader.line_num, row) for row in reader if any(field.strip() for field in row)]
        fields = [field.strip().lower() for field in records[0][1]]
        has_header = "source" in fields or "target" in fields
        if has_header:
            required = {"name", "source", "target"}
            if not required.issubset(fields) or len(fields) != len(set(fields)):
                raise ValueError("Manifest header must contain distinct name, source, target columns: {}".format(self.data_file))
            records = records[1:]
        data = []
        for line_number, row in records:
            row = [field.strip() for field in row]
            if has_header:
                if len(row) != len(fields):
                    raise ValueError("{}:{}: expected {} columns, found {}".format(self.data_file, line_number, len(fields), len(row)))
                item = dict(zip(fields, row))
                item["prompt"] = item.get("prompt", default_prompt)
            else:
                if len(row) not in (3, 4):
                    raise ValueError("{}:{}: expected 3 or 4 columns, found {}".format(self.data_file, line_number, len(row)))
                name, source, target = row[:3]
                if is_cvusa:
                    name = os.path.splitext(os.path.basename(name))[0]
                    if len(row) == 3:
                        source, target = target, source
                item = dict(name=name, source=source, target=target,
                            prompt=row[3] if len(row) == 4 else default_prompt)
            if not all(item[key] for key in ("name", "source", "target")):
                raise ValueError("{}:{}: name, source and target must be nonempty".format(self.data_file, line_number))
            item["source"] = self._resolve_image_path(item["source"])
            item["target"] = self._resolve_image_path(item["target"])
            data.append(item)
        if not data:
            raise ValueError("Manifest contains no samples: {}".format(self.data_file))
        return data

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _read_rgb(filename):
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        if image is None:
            raise OSError("Cannot read RGB image: {}".format(filename))
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _normalize_condition(source):
        source = np.asarray(source)
        if source.ndim != 3 or source.shape[2] != 3 or min(source.shape[:2]) < 1:
            raise ValueError("BEV condition must have nonempty HWC RGB shape, got {}".format(source.shape))
        if source.dtype == np.uint8:
            return source.astype(np.float32) / 255.0
        if not np.issubdtype(source.dtype, np.floating):
            raise ValueError("BEV condition must be uint8 [0, 255] or floating point [0, 1], got {}".format(source.dtype))
        if not np.isfinite(source).all():
            raise ValueError("BEV condition contains NaN or infinity")
        if source.min() < 0 or source.max() > 1:
            raise ValueError("Floating BEV condition must be in [0, 1]; uint8 is required for [0, 255]")
        return source.astype(np.float32)

    def __getitem__(self, idx):
        item = self.data[idx]
        try:
            if Path(item["source"]).suffix.lower() == ".pkl":
                with open(item["source"], "rb") as handle:
                    source = pickle.load(handle)
            elif self.source_mode == "bev":
                source = self._read_rgb(item["source"])
            else:
                # CruvedBEV already returns float RGB in [0, 1].
                source = CruvedBEV(item["source"], self.dataset_name,
                                   rotation_degrees=self.rotation_degrees)
            source = self._normalize_condition(source)
            source = cv2.resize(source, self.image_size)
            target = cv2.resize(self._read_rgb(item["target"]), self.image_size)
        except (OSError, ValueError, TypeError, pickle.UnpicklingError, EOFError, cv2.error) as error:
            raise ValueError("Sample {!r} failed (source={!r}, target={!r}): {}".format(
                item["name"], item["source"], item["target"], error)) from error
        prompt = item["prompt"]
        if self.drop_prompt_ratio > 0 and random.random() < self.drop_prompt_ratio:
            prompt = ""
        target = target.astype(np.float32) / 127.5 - 1.0
        return dict(jpg=target, txt=prompt, hint=source, name=item["name"])
