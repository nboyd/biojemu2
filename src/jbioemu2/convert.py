"""Torch -> JAX weight conversion. The only module that imports torch (``reference`` extra)."""

from __future__ import annotations

import importlib
from pathlib import Path

import jax.numpy as jnp
import torch

import jbioemu2.atom  # noqa: F401  (registers classes)
import jbioemu2.model  # noqa: F401
import jbioemu2.nn  # noqa: F401
import jbioemu2.structure  # noqa: F401
import jbioemu2.transformer  # noqa: F401
import jbioemu2.trunk  # noqa: F401
from jbioemu2.backend import REGISTRY, from_torch
from jbioemu2.model import BioEmu2


def _resolve(path: str):
    module, _, name = path.rpartition(".")
    return getattr(importlib.import_module(module), name)


for _path, _convert in REGISTRY:
    from_torch.register(_resolve(_path), _convert)
from_torch.register(torch.Tensor, lambda t: jnp.array(t.detach().cpu().float().numpy()))


def load_reference_model(checkpoint_path: Path, config_path: Path) -> torch.nn.Module:
    """The reference ``BioMoaCModel`` with EMA weights, on CPU in eval mode."""
    from bioemu2.model import _load_score_model

    model, _ = _load_score_model(checkpoint_path, config_path, fingerprint_required=False)
    return model.eval()


def convert_checkpoint(checkpoint_path: Path, config_path: Path) -> BioEmu2:
    """Load a BioEmu2 release checkpoint (``checkpoint.ckpt`` + ``config.yaml``) into JAX."""
    with torch.no_grad():
        return from_torch(load_reference_model(checkpoint_path, config_path))


def convert_released(name: str = "bioemu2-preview") -> BioEmu2:
    """Download (if needed) and convert a released checkpoint from ``microsoft/bioemu2``."""
    from bioemu2.checkpoints import get_checkpoint_files

    files = get_checkpoint_files(name)
    return convert_checkpoint(files.checkpoint_path, files.config_path)
