"""Torch -> JAX conversion machinery.

Equinox classes register the dotted path of the torch class they mirror with
``@register_from_torch("pkg.module.Class")``. Registration is bookkeeping only: this
module never imports torch. ``biojemu2.convert`` (the only torch importer) resolves
the paths and dispatches through :func:`from_torch`.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

REGISTRY: list[tuple[str, Any]] = []


def register_from_torch(torch_path: str):
    """Class decorator: convert torch modules of class ``torch_path`` with ``cls.from_torch``."""

    def wrap(cls):
        REGISTRY.append((torch_path, cls.from_torch))
        return cls

    return wrap


def register_unwrap(torch_path: str, child: str) -> None:
    """Convert torch modules of class ``torch_path`` as their submodule ``child``: for torch
    wrappers that only reshape batch dimensions, which unbatched JAX code does not need."""
    REGISTRY.append((torch_path, lambda m: from_torch(getattr(m, child))))


@functools.singledispatch
def from_torch(obj: Any) -> Any:
    raise TypeError(f"no from_torch conversion for {type(obj).__module__}.{type(obj).__name__}")


for _scalar in (int, float, bool, str, type(None)):
    from_torch.register(_scalar, lambda x: x)


class AbstractFromTorch(eqx.Module):
    """Convert a torch module by matching child-module, parameter and buffer names to fields.

    Fields absent from the torch module fall back to their dataclass defaults; torch
    children/parameters with no matching field are an error.
    """

    @classmethod
    def from_torch(cls, module: Any):
        names = {f.name for f in dataclasses.fields(cls)}
        kwargs: dict[str, Any] = {}
        for name, child in module.named_children():
            if name not in names:
                raise ValueError(f"{cls.__name__}: unmatched child module {name!r}")
            kwargs[name] = from_torch(child)
        for name, param in module.named_parameters(recurse=False):
            if name not in names:
                raise ValueError(f"{cls.__name__}: unmatched parameter {name!r}")
            kwargs[name] = from_torch(param)
        for name, buf in module.named_buffers(recurse=False):
            if name in names:
                kwargs[name] = from_torch(buf)
        for f in dataclasses.fields(cls):
            if f.name in kwargs:
                continue
            if hasattr(module, f.name):
                kwargs[f.name] = from_torch(getattr(module, f.name))
            elif f.default is not dataclasses.MISSING:
                kwargs[f.name] = f.default
            else:
                raise ValueError(f"{cls.__name__}: no value for field {f.name!r}")
        return cls(**kwargs)


def stack_modules(modules: list[eqx.Module]) -> eqx.Module:
    """Identical modules as one module whose arrays gain a leading layer axis (for
    :func:`biojemu2.nn.scan_layers`). ``jax.tree.map`` rejects differing static structure."""
    return jax.tree.map(lambda *xs: jnp.stack(xs), *modules)
