"""Checkpoint preparation and parameter selection for single-image training."""

from pathlib import Path

import torch


def load_weights(path):
    """Read a local LDM/Lightning checkpoint or a .safetensors state dictionary."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Checkpoint does not exist: {}".format(path))
    if path.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file
        weights = load_file(str(path), device="cpu")
    else:
        # The repository targets torch 1.12; Lightning resumes contain more than tensors.
        weights = torch.load(str(path), map_location="cpu")
    if isinstance(weights, dict) and "state_dict" in weights:
        weights = weights["state_dict"]
    if not isinstance(weights, dict):
        raise ValueError("Expected an LDM/Lightning state dictionary: {}".format(path))
    return weights


def _check_shapes(target, weights, keys):
    missing = [key for key in keys if key not in weights]
    mismatched = [key for key in keys if key in weights and
                  (not isinstance(weights[key], torch.Tensor) or
                   target[key].shape != weights[key].shape)]
    if missing or mismatched:
        raise ValueError(
            "Incompatible checkpoint. Missing keys ({}): {}; shape mismatches ({}): {}. "
            "Use an original LDM SD 1.5 checkpoint, or a complete LAControlLDM checkpoint."
            .format(len(missing), missing[:8], len(mismatched), mismatched[:8]))


def initialize_from_sd(model, weights):
    """Copy SD 1.5 weights, pad the control input, and retain zero convolutions."""
    target = model.state_dict()
    prefixes = ("model.diffusion_model.", "first_stage_model.", "cond_stage_model.")
    required = [key for key in target if key.startswith(prefixes)]
    _check_shapes(target, weights, required)
    initialized = {}
    copied = {prefix: 0 for prefix in prefixes}
    control_copied = 0
    for key, value in target.items():
        if key.startswith(prefixes):
            initialized[key] = weights[key]
            for prefix in prefixes:
                copied[prefix] += 1
        elif key.startswith("cond_encoder."):
            source_key = "first_stage_model." + key[len("cond_encoder."):]
            _check_shapes(target, {key: weights.get(source_key)}, [key])
            initialized[key] = weights[source_key]
        elif key.startswith("control_model."):
            source_key = "model.diffusion_model." + key[len("control_model."):]
            if source_key not in weights:
                # zero_convs and middle_block_out are intentionally zero initialized.
                if not key.startswith(("control_model.zero_convs.",
                                       "control_model.middle_block_out.")):
                    raise ValueError("No SD source for control parameter: " + key)
                initialized[key] = torch.zeros_like(value)
            elif value.shape == weights[source_key].shape:
                initialized[key] = weights[source_key]
                control_copied += 1
            elif key == "control_model.input_blocks.0.0.weight":
                source = weights[source_key]
                if (value.ndim != 4 or source.ndim != 4 or
                        value.shape[0] != source.shape[0] or
                        value.shape[2:] != source.shape[2:] or
                        value.shape[1] != 2 * source.shape[1]):
                    raise ValueError("Unexpected control input convolution shape")
                expanded = torch.zeros_like(value)
                expanded[:, :source.shape[1]] = source
                initialized[key] = expanded
                control_copied += 1
            else:
                raise ValueError("Control/SD parameter shape mismatch: " + key)
        else:
            # Schedule and logvar come from the chosen training configuration.
            initialized[key] = value
    model.load_state_dict(initialized, strict=True)
    return {"mode": "sd15", "copied": copied,
            "control_parameters_copied": control_copied,
            "condition_encoder": "copied from loaded VAE"}


def initialize_from_skydiffusion(model, weights):
    """Require every model key, avoiding silent random frozen modules."""
    target = model.state_dict()
    _check_shapes(target, weights, list(target))
    unexpected = [key for key in weights if key not in target]
    model.load_state_dict({key: weights[key] for key in target}, strict=True)
    return {"mode": "skydiffusion", "loaded_keys": len(target),
            "ignored_extra_keys": unexpected[:20], "ignored_extra_count": len(unexpected)}


def configure_trainable_parameters(model, learning_rate=1e-5, sd_locked=True):
    """Freeze the SD backbone while retaining autograd through its decoder."""
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.control_model.parameters():
        parameter.requires_grad_(True)
    if not sd_locked:
        for module in (model.model.diffusion_model.output_blocks,
                       model.model.diffusion_model.out):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"trainable_parameters": trainable,
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "sd_locked": sd_locked, "learning_rate": learning_rate}


def make_completed_checkpoint_callback():
    """Keep PL 1.5's final explicit save aligned with completed optimizer steps."""
    from pytorch_lightning.callbacks import Callback

    class CompletedCheckpointStep(Callback):
        def on_save_checkpoint(self, trainer, pl_module, checkpoint):
            # During training PL 1.5 saves before incrementing global_step, so
            # its built-in +1 is correct. After fit returns it has incremented
            # already, but dump_checkpoint still adds 1. Fix only that case.
            if trainer.state.finished:
                checkpoint["global_step"] = trainer.global_step

    return CompletedCheckpointStep()
