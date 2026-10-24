"""Leaf layers mirroring the torch modules BioEmu2 is built from."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from biojemu2.backend import AbstractFromTorch, from_torch, register_from_torch


@register_from_torch("torch.nn.Linear")
class Linear(AbstractFromTorch):
    weight: Float[Array, "Out In"]
    bias: Float[Array, " Out"] | None

    def __call__(self, x: Float[Array, "... In"]) -> Float[Array, "... Out"]:
        y = jnp.einsum("...i,oi->...o", x, self.weight)
        return y if self.bias is None else y + self.bias


@register_from_torch("torch.nn.LayerNorm")
class LayerNorm(AbstractFromTorch):
    weight: Float[Array, " D"] | None
    bias: Float[Array, " D"] | None
    eps: float = eqx.field(static=True)

    def __call__(self, x: Float[Array, "... D"]) -> Float[Array, "... D"]:
        x = jax.nn.standardize(x, epsilon=self.eps, algorithm="stable")
        if self.weight is not None:
            x = x * self.weight
        if self.bias is not None:
            x = x + self.bias
        return x


@register_from_torch("torch.nn.Embedding")
class Embedding(AbstractFromTorch):
    weight: Float[Array, "V D"]

    def __call__(self, idx: Int[Array, "..."]) -> Float[Array, "... D"]:
        return self.weight[idx]


@register_from_torch("torch.nn.ReLU")
class ReLU(AbstractFromTorch):
    def __call__(self, x: Array) -> Array:
        return jax.nn.relu(x)


@register_from_torch("torch.nn.SiLU")
class SiLU(AbstractFromTorch):
    def __call__(self, x: Array) -> Array:
        return jax.nn.silu(x)


@register_from_torch("torch.nn.modules.container.Sequential")
class Sequential(eqx.Module):
    layers: tuple[eqx.Module, ...]

    def __call__(self, x: Array) -> Array:
        for layer in self.layers:
            x = layer(x)
        return x

    @classmethod
    def from_torch(cls, m):
        return cls(layers=tuple(from_torch(child) for child in m))


@register_from_torch("bioemu2.biomoa.ops.transition.Transition")
class Transition(AbstractFromTorch):
    """SwiGLU MLP ``fc3(silu(fc1(x)) * fc2(x))`` with an optional pre-LayerNorm."""

    fc1: Linear
    fc2: Linear
    fc3: Linear
    norm: LayerNorm | None = None

    def __call__(self, x: Float[Array, "... D"]) -> Float[Array, "... O"]:
        if self.norm is not None:
            x = self.norm(x)
        return self.fc3(jax.nn.silu(self.fc1(x)) * self.fc2(x))


@register_from_torch("bioemu2.biomoa.modules.encoders.FourierEmbedding")
class FourierEmbedding(AbstractFromTorch):
    """Fixed random Fourier features ``cos(2π (w t + b))``."""

    proj: Linear

    def __call__(self, t: Float[Array, "*B"]) -> Float[Array, "*B D"]:
        return jnp.cos(2 * math.pi * self.proj(t[..., None]))


def scan_layers(step: Callable[[Any, Any], Any], carry: Any, layers: eqx.Module) -> Any:
    """``carry = step(layer, carry)`` for each layer of a stacked module (see
    ``backend.stack_modules``): one rematerialised ``lax.scan`` body for all layers."""
    params, static = eqx.partition(layers, eqx.is_array)

    @jax.checkpoint
    def body(carry, params):
        return step(eqx.combine(params, static), carry), None

    return jax.lax.scan(body, carry, params)[0]
