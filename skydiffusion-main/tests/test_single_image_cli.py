"""Exercise the complete CLI using tiny real modules and temporary RGB pairs."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_single_image_training import tiny_model
from skydiffusion_train import main
from skydiffusion_train import create_argparser, load_settings


class SingleImageCLITests(unittest.TestCase):
    def test_ddim_schedule_rejects_out_of_range_last_index(self):
        args = create_argparser().parse_args(["--train-data", "unused.csv", "--ddim-steps", "1000"])
        with self.assertRaisesRegex(ValueError, "below 1000"):
            load_settings(args)

    def test_resume_smoke_combination_fails_before_loading_weights(self):
        with patch.object(sys, "argv", ["skydiffusion_train.py", "--resume", "unused.ckpt", "--smoke-test"]):
            with self.assertRaises(SystemExit) as error:
                main()
        self.assertEqual(error.exception.code, 2)

    def test_cli_trains_saves_and_resumes_with_rgb_manifest(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8)).save(str(root / "bev.png"))
            Image.fromarray(np.full((64, 64, 3), 160, dtype=np.uint8)).save(str(root / "gt.png"))
            manifest = root / "pairs.csv"
            manifest.write_text(
                "name,source,target,prompt\none,bev.png,gt.png,\ntwo,bev.png,gt.png,\n",
                encoding="utf-8")
            initial = root / "init.ckpt"
            torch.save(tiny_model().state_dict(), str(initial))
            output = root / "run"
            arguments = ["skydiffusion_train.py", "--train-data", str(manifest),
                         "--val-data", str(manifest), "--source-mode", "bev",
                         "--image-size", "64", "--accelerator", "cpu",
                         "--output-dir", str(output), "--init-checkpoint", str(initial),
                         "--smoke-test"]
            # Only replace full model construction. The actual training, file
            # parsing, optimizers, callbacks and checkpoint restore run normally.
            with patch.object(sys, "argv", arguments), patch(
                    "ldm.util.instantiate_from_config", return_value=tiny_model()):
                main()
            saved_path = output / "checkpoints/last.ckpt"
            saved = torch.load(str(saved_path), map_location="cpu")
            self.assertEqual(saved["global_step"], 2)
            self.assertTrue(saved["optimizer_states"][0]["state"])
            self.assertTrue((output / "training_config.yaml").is_file())
            self.assertTrue((output / "model_config.yaml").is_file())
            self.assertTrue((output / "data_preview/train/000_bev_gt.png").is_file())
            self.assertTrue(list((output / "metrics").glob("version_*/metrics.csv")))
            arguments = ["skydiffusion_train.py", "--config", str(output / "training_config.yaml"),
                         "--resume", str(saved_path), "--max-steps", "3",
                         "--image-log-every", "0"]
            with patch.object(sys, "argv", arguments), patch(
                    "ldm.util.instantiate_from_config", return_value=tiny_model()):
                main()
            saved = torch.load(str(saved_path), map_location="cpu")
            self.assertEqual(saved["global_step"], 3)
            self.assertTrue(any(float(state["step"]) == 3
                                for state in saved["optimizer_states"][0]["state"].values()))


if __name__ == "__main__":
    unittest.main()
