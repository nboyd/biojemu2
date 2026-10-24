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
    children/parameters with no matching field are an error. :func:`stacked` fields
    become one module with a leading layer axis.
    """

    @classmethod
    def from_torch(cls, module: Any):
        fields = dataclasses.fields(cls)
        names = {f.name for f in fields}
        torch_names = [n for n, _ in module.named_children()]
        torch_names += [n for n, _ in module.named_parameters(recurse=False)]
        if unmatched := [n for n in torch_names if n not in names]:
            raise ValueError(f"{cls.__name__}: unmatched torch children/parameters {unmatched}")
        kwargs: dict[str, Any] = {}
        for f in fields:
            if hasattr(module, f.name):
                value = from_torch(getattr(module, f.name))
                if f.metadata.get("stacked") and value is not None:
                    value = stack_modules(value)
                kwargs[f.name] = value
            elif f.default is not dataclasses.MISSING:
                kwargs[f.name] = f.default
            else:
                raise ValueError(f"{cls.__name__}: no value for field {f.name!r}")
        return cls(**kwargs)


def stacked(**kwargs):
    """Field converted from a torch ``ModuleList`` with :func:`stack_modules`."""
    return eqx.field(metadata={"stacked": True}, **kwargs)


def stack_modules(modules: tuple[eqx.Module, ...]) -> eqx.Module:
    """Identical modules as one module whose arrays gain a leading layer axis (for
    :func:`biojemu2.nn.scan_layers`). ``jax.tree.map`` rejects differing static structure."""
    return jax.tree.map(lambda *xs: jnp.stack(xs), *modules)
