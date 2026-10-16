"""Small local data checks; no model, checkpoint, GPU or downloads are needed."""

import csv
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cruvedbev import BEV_transform, CruvedBEV, Transformation_SVI
from skydiffusion_dataset import SkyDataset


class SingleImageDataTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.rgb = np.full((4, 6, 3), [255, 64, 0], dtype=np.uint8)
        Image.fromarray(self.rgb).save(str(self.root / "source.png"))
        Image.fromarray(self.rgb).save(str(self.root / "target.png"))

    def tearDown(self):
        self.temp_dir.cleanup()

    def manifest(self, rows, delimiter=",", name="pairs.csv"):
        filename = self.root / name
        with filename.open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle, delimiter=delimiter).writerows(rows)
        return filename

    def dataset(self, filename, **kwargs):
        kwargs.setdefault("image_size", (6, 4))
        kwargs.setdefault("source_mode", "bev")
        return SkyDataset(filename, **kwargs)

    def test_headered_csv_preserves_quoted_prompt_and_rgb(self):
        filename = self.manifest([
            ["name", "source", "target", "prompt"],
            ["sample", "source.png", "target.png", "an overhead view, with trees"],
        ])
        sample = self.dataset(filename)[0]
        np.testing.assert_allclose(sample["hint"][0, 0], [1, 64 / 255, 0])
        np.testing.assert_allclose(sample["jpg"][0, 0], [1, 64 / 127.5 - 1, -1])
        self.assertEqual(sample["txt"], "an overhead view, with trees")
        self.assertEqual(sample["hint"].dtype, np.float32)
        self.assertEqual(sample["jpg"].dtype, np.float32)

    def test_headered_tsv_uses_default_prompt(self):
        filename = self.manifest([
            ["target", "name", "source"], ["target.png", "sample", "source.png"],
        ], delimiter="\t")
        dataset = self.dataset(filename, prompt="default")
        self.assertEqual(dataset[0]["txt"], "default")
        self.assertEqual(Path(dataset.data[0]["source"]), self.root / "source.png")

    def test_legacy_cvact_tsv_and_cvusa_column_order(self):
        filename = self.manifest([["sample", "source.png", "target.png"]], delimiter="\t")
        self.assertEqual(self.dataset(filename)[0]["name"], "sample")
        filename = self.manifest([["folder/sample.jpg", "target.png", "source.png"]])
        dataset = self.dataset(filename, is_cvusa=True, dataset_name="CVUSA")
        self.assertEqual(dataset[0]["name"], "sample")
        self.assertEqual(Path(dataset.data[0]["source"]).name, "source.png")
        filename = self.manifest([["folder/sample.jpg", "source.png", "target.png", "text"]])
        self.assertEqual(self.dataset(filename, is_cvusa=True)[0]["txt"], "text")

    def test_explicit_data_root_overrides_manifest_directory(self):
        image_dir = self.root / "images"
        image_dir.mkdir()
        Image.fromarray(np.zeros_like(self.rgb)).save(str(image_dir / "source.png"))
        Image.fromarray(self.rgb).save(str(image_dir / "target.png"))
        filename = self.manifest([["sample", "source.png", "target.png"]])
        sample = self.dataset(filename, data_root=image_dir)[0]
        self.assertEqual(float(sample["hint"].max()), 0)

    def test_float_panorama_condition_is_not_divided_by_255(self):
        filename = self.manifest([["sample", "source.png", "target.png"]])
        condition = np.full((4, 6, 3), 0.5, dtype=np.float32)
        with mock.patch("skydiffusion_dataset.CruvedBEV", return_value=condition) as transform:
            sample = self.dataset(filename, source_mode="panorama", rotation_degrees=45)[0]
        np.testing.assert_array_equal(sample["hint"], condition)
        self.assertEqual(transform.call_args[1]["rotation_degrees"], 45)

    def test_pickled_uint8_and_unit_float_conditions_match(self):
        for array in (self.rgb, self.rgb.astype(np.float32) / 255):
            with (self.root / "source.pkl").open("wb") as handle:
                pickle.dump(array, handle)
            filename = self.manifest([["sample", "source.pkl", "target.png"]])
            sample = self.dataset(filename, source_mode="panorama")[0]
            np.testing.assert_allclose(sample["hint"], self.rgb.astype(np.float32) / 255)

    def test_invalid_pickled_conditions_report_sample_and_range(self):
        invalid = [
            (np.full((4, 6, 3), 128, dtype=np.float32), "must be in"),
            (np.full((4, 6, 3), np.nan, dtype=np.float32), "NaN"),
            (np.ones((3, 4, 6), dtype=np.float32), "HWC RGB"),
            (np.ones((4, 6, 3), dtype=np.int16), "uint8"),
        ]
        for array, message in invalid:
            with self.subTest(message=message):
                with (self.root / "source.pkl").open("wb") as handle:
                    pickle.dump(array, handle)
                filename = self.manifest([["bad_sample", "source.pkl", "target.png"]])
                with self.assertRaisesRegex(ValueError, "bad_sample.*" + message):
                    self.dataset(filename)[0]

    def test_unreadable_images_report_paths(self):
        filename = self.manifest([["sample", "missing.png", "target.png"]])
        with self.assertRaisesRegex(ValueError, "sample.*missing.png"):
            self.dataset(filename)[0]

    def test_empty_or_malformed_manifest_fails_early(self):
        for rows in ([], [["name", "source", "target"]], [["too", "short"]]):
            with self.subTest(rows=rows):
                with self.assertRaises(ValueError):
                    self.dataset(self.manifest(rows))

    def test_prompt_dropout_and_opencv_size_order(self):
        filename = self.manifest([["sample", "source.png", "target.png", "text"]])
        sample = self.dataset(filename, drop_prompt_ratio=1, image_size=(8, 5))[0]
        self.assertEqual(sample["hint"].shape, (5, 8, 3))
        self.assertEqual(sample["jpg"].shape, (5, 8, 3))
        self.assertEqual(sample["txt"], "")

    def test_rotation_is_degrees_with_quarter_panorama_shift(self):
        uv0 = BEV_transform(torch.tensor([0.]), 8, 40, 80, 0.06, -1.5)
        uv90 = BEV_transform(torch.tensor([90.]), 8, 40, 80, 0.06, -1.5)
        shift = torch.remainder(uv90[..., 0] - uv0[..., 0], 80)
        np.testing.assert_allclose(shift.numpy(), 20, atol=2e-5)
        np.testing.assert_allclose(uv90[..., 1].numpy(), uv0[..., 1].numpy())

    def test_panorama_loader_converts_grayscale_to_rgb(self):
        filename = self.root / "gray.png"
        Image.fromarray(np.full((4, 8), 128, dtype=np.uint8)).save(str(filename))
        uv = torch.tensor([[[[1.5, 1.5], [2.5, 1.5]]]], dtype=torch.float32)
        condition = Transformation_SVI(filename, uv, 4, 8)
        self.assertEqual(condition.shape, (1, 2, 3))
        np.testing.assert_allclose(condition, 128 / 255)

    def test_only_supported_panorama_datasets_are_advertised(self):
        with self.assertRaisesRegex(ValueError, "CVACT, CVUSA"):
            CruvedBEV(self.root / "source.png", dataset="VIGOR")


if __name__ == "__main__":
    unittest.main()
