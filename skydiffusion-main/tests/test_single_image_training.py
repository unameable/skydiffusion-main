"""CPU integration checks with the real LAControlLDM, without external weights.

Run with the repository's environment:
    python -m unittest discover -s tests -p test_single_image_training.py -v
"""

import gc
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cldm.cldm import LAControlLDM
from ldm.modules.diffusionmodules.util import checkpoint
from skydiffusion_training import (configure_trainable_parameters,
                                  initialize_from_sd,
                                  initialize_from_skydiffusion, load_weights,
                                  make_completed_checkpoint_callback)


class DummyTextEncoder(nn.Module):
    """A local stand-in for CLIP; all diffusion and VAE modules remain real."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(2, 8)

    def encode(self, text):
        indices = torch.zeros(len(text), 2, dtype=torch.long,
                              device=self.embedding.weight.device)
        return self.embedding(indices)


class TinyPairs(Dataset):
    def __init__(self):
        generator = torch.Generator().manual_seed(456)
        self.hint = torch.rand(2, 16, 16, 3, generator=generator)
        self.target = torch.rand(2, 16, 16, 3, generator=generator) * 2 - 1

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {"hint": self.hint[index], "jpg": self.target[index],
                "txt": "", "name": str(index)}


def tiny_model():
    torch.manual_seed(123)
    common = dict(image_size=8, in_channels=4, model_channels=32,
                  attention_resolutions=[], num_res_blocks=1,
                  channel_mult=[1, 2], num_heads=4,
                  use_spatial_transformer=True, transformer_depth=1,
                  context_dim=8, use_checkpoint=True, legacy=False)
    model = LAControlLDM(
        control_stage_config={"target": "cldm.cldm.LAControlNet",
                              "params": dict(common, hint_channels=4)},
        control_key="hint", only_mid_control=False,
        unet_config={"target": "cldm.cldm.ControlledUnetModel",
                     "params": dict(common, out_channels=4)},
        first_stage_config={
            "target": "ldm.models.autoencoder.AutoencoderKL",
            "params": {
                "embed_dim": 4, "lossconfig": {"target": "torch.nn.Identity"},
                "ddconfig": dict(double_z=True, z_channels=4, resolution=16,
                                 in_channels=3, out_ch=3, ch=32, ch_mult=[1, 2],
                                 num_res_blocks=1, attn_resolutions=[], dropout=0.)}},
        cond_stage_config={"target": __name__ + ".DummyTextEncoder"},
        timesteps=10, linear_start=.00085, linear_end=.012,
        first_stage_key="jpg", cond_stage_key="txt", image_size=8,
        channels=4, cond_stage_trainable=False, conditioning_key="crossattn",
        scale_factor=.18215, use_ema=False)
    # Pretrained SD has a nonzero output convolution. Fresh UNet initializes it
    # to zero, which would initially conceal the control branch's gradients.
    with torch.no_grad():
        model.model.diffusion_model.out[-1].weight.normal_(std=.01)
    source = {key: value.detach().clone() for key, value in model.state_dict().items()
              if key.startswith(("model.diffusion_model.", "first_stage_model.",
                                 "cond_stage_model."))}
    initialize_from_sd(model, source)
    configure_trainable_parameters(model, learning_rate=1e-3, sd_locked=True)
    return model


class SingleImageTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def tearDown(self):
        gc.collect()

    def test_checkpoint_preserves_input_gradients_through_frozen_layer(self):
        layer = nn.Linear(3, 2)
        for parameter in layer.parameters():
            parameter.requires_grad_(False)
        data = torch.randn(4, 3, requires_grad=True)
        expected = layer.weight.sum(dim=0).expand_as(data)
        checkpoint(layer, (data,), layer.parameters(), True).sum().backward()
        torch.testing.assert_close(data.grad, expected)
        self.assertTrue(all(parameter.grad is None for parameter in layer.parameters()))

    def test_sd_initialization_and_strict_checkpoint_validation(self):
        model = tiny_model()
        source = {key: value.detach().clone()
                  for key, value in model.state_dict().items()
                  if key.startswith(("model.diffusion_model.", "first_stage_model.",
                                     "cond_stage_model."))}
        # Change source tensors after cond_encoder was constructed: this catches
        # accidentally retaining the VAE copy made before loading SD weights.
        source["first_stage_model.encoder.conv_in.weight"].fill_(.21)
        source["first_stage_model.quant_conv.weight"].fill_(.14)
        initialize_from_sd(model, source)
        state = model.state_dict()
        source_input = state["model.diffusion_model.input_blocks.0.0.weight"]
        control_input = state["control_model.input_blocks.0.0.weight"]
        torch.testing.assert_close(control_input[:, :4], source_input)
        self.assertEqual(torch.count_nonzero(control_input[:, 4:]).item(), 0)
        for key, value in state.items():
            if key.startswith("cond_encoder."):
                source_key = "first_stage_model." + key[len("cond_encoder."):]
                torch.testing.assert_close(value, state[source_key])
            if key.startswith(("control_model.zero_convs.",
                               "control_model.middle_block_out.")):
                self.assertEqual(torch.count_nonzero(value).item(), 0)
        invalid = dict(state)
        invalid.pop("cond_encoder.quant_conv.weight")
        with self.assertRaisesRegex(ValueError, "Missing keys"):
            initialize_from_skydiffusion(model, invalid)
        invalid = dict(state)
        invalid["control_model.input_blocks.0.0.weight"] = torch.zeros(1)
        with self.assertRaisesRegex(ValueError, "shape mismatches"):
            initialize_from_skydiffusion(model, invalid)
        source.pop("first_stage_model.encoder.conv_in.weight")
        with self.assertRaisesRegex(ValueError, "Missing keys"):
            initialize_from_sd(model, source)

    def test_two_optimizer_steps_train_control_and_preserve_frozen_weights(self):
        model = tiny_model()
        model.train()
        self.assertFalse(model.first_stage_model.training)
        self.assertFalse(model.cond_encoder.training)
        frozen = {key: value.detach().clone() for key, value in model.named_parameters()
                  if not value.requires_grad}
        batch = next(iter(DataLoader(TinyPairs(), batch_size=2)))
        latent, conditioning = model.get_input(batch, "jpg", bs=1)
        self.assertEqual(latent.shape[0], 1)
        self.assertEqual(conditioning["c_concat"][0].shape[0], 1)
        self.assertEqual(conditioning["c_crossattn"][0].shape[0], 1)
        optimizer = model.configure_optimizers()
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = model.shared_step(batch)
            self.assertTrue(torch.isfinite(loss).item())
            loss.backward()
            grad = model.control_model.zero_convs[0][0].weight.grad
            self.assertIsNotNone(grad)
            self.assertGreater(grad.abs().sum().item(), 0)
            optimizer.step()
        grad = model.control_model.input_blocks[0][0].weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(grad.abs().sum().item(), 0)
        for key, value in model.named_parameters():
            if key in frozen:
                self.assertIsNone(value.grad, key)
                torch.testing.assert_close(value, frozen[key], rtol=0, atol=0)

    def test_checkpoint_callback_preserves_running_save_step(self):
        callback = make_completed_checkpoint_callback()
        trainer = SimpleNamespace(state=SimpleNamespace(finished=False), global_step=1)
        saved = {"global_step": 2}
        callback.on_save_checkpoint(trainer, None, saved)
        self.assertEqual(saved["global_step"], 2)

    def test_cpu_preview_samples_and_logs_rgb_hint(self):
        model = tiny_model().eval()
        batch = next(iter(DataLoader(TinyPairs(), batch_size=2)))
        images = model.log_images(batch, N=1, ddim_steps=5,
                                  unconditional_guidance_scale=1.0)
        self.assertEqual(tuple(images["control"].shape), (1, 3, 16, 16))
        torch.testing.assert_close(images["control"], batch["hint"][:1].permute(0, 3, 1, 2) * 2 - 1)
        samples = images["samples_cfg_scale_1.00"]
        self.assertEqual(tuple(samples.shape), (1, 3, 16, 16))
        self.assertEqual(samples.device.type, "cpu")
        self.assertTrue(torch.isfinite(samples).all().item())

    def test_lightning_saves_and_resumes_optimizer_and_global_step(self):
        with tempfile.TemporaryDirectory() as directory:
            loader = DataLoader(TinyPairs(), batch_size=1)
            model = tiny_model()
            common = dict(gpus=0, precision=32, max_epochs=10,
                          enable_checkpointing=False, enable_progress_bar=False,
                          enable_model_summary=False, num_sanity_val_steps=0,
                          log_every_n_steps=1, limit_val_batches=1,
                          callbacks=[make_completed_checkpoint_callback()])
            trainer = pl.Trainer(max_steps=2, default_root_dir=directory,
                                 logger=CSVLogger(directory, name="first"), **common)
            trainer.fit(model, train_dataloaders=loader, val_dataloaders=loader)
            self.assertEqual(trainer.global_step, 2)
            checkpoint_path = str(Path(directory) / "resume.ckpt")
            trainer.save_checkpoint(checkpoint_path)
            stored = torch.load(checkpoint_path, map_location="cpu")
            saved_step = stored["global_step"]
            self.assertEqual(saved_step, 2)
            self.assertTrue(stored["optimizer_states"][0]["state"])
            resumed = tiny_model()
            initialize_from_skydiffusion(resumed, load_weights(checkpoint_path))
            trainer2 = pl.Trainer(max_steps=saved_step + 1, default_root_dir=directory,
                                  logger=CSVLogger(directory, name="resumed"), **common)
            trainer2.fit(resumed, train_dataloaders=loader,
                         val_dataloaders=loader, ckpt_path=checkpoint_path)
            self.assertEqual(trainer2.global_step, saved_step + 1)
            states = trainer2.optimizers[0].state.values()
            self.assertTrue(any(float(state["step"]) == 3 for state in states))
            self.assertIn("val/loss", trainer.callback_metrics)


if __name__ == "__main__":
    unittest.main()
