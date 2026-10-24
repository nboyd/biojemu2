"""Leaf layers mirroring the torch modules BioEmu2 is built from."""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from jbioemu2.backend import AbstractFromTorch, from_torch, register_from_torch


@register_from_torch("torch.nn.Linear")
class Linear(eqx.Module):
    weight: Float[Array, "Out In"]
    bias: Float[Array, " Out"] | None

    def __call__(self, x: Float[Array, "... In"]) -> Float[Array, "... Out"]:
        y = jnp.einsum("...i,oi->...o", x, self.weight)
        return y if self.bias is None else y + self.bias

    @classmethod
    def from_torch(cls, m):
        return cls(weight=from_torch(m.weight), bias=None if m.bias is None else from_torch(m.bias))


def layer_norm(x: Float[Array, "... D"], eps: float) -> Float[Array, "... D"]:
    mean = x.mean(-1, keepdims=True)
    var = jnp.square(x - mean).mean(-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(var + eps)


@register_from_torch("torch.nn.LayerNorm")
class LayerNorm(eqx.Module):
    weight: Float[Array, " D"] | None
    bias: Float[Array, " D"] | None
    eps: float = eqx.field(static=True)

    def __call__(self, x: Float[Array, "... D"]) -> Float[Array, "... D"]:
        x = layer_norm(x, self.eps)
        if self.weight is not None:
            x = x * self.weight
        if self.bias is not None:
            x = x + self.bias
        return x

    @classmethod
    def from_torch(cls, m):
        return cls(
            weight=None if m.weight is None else from_torch(m.weight),
            bias=None if m.bias is None else from_torch(m.bias),
            eps=float(m.eps),
        )


@register_from_torch("torch.nn.Embedding")
class Embedding(AbstractFromTorch):
    weight: Float[Array, "V D"]

    def __call__(self, idx: Int[Array, "..."]) -> Float[Array, "... D"]:
        return self.weight[idx]


@register_from_torch("torch.nn.ReLU")
class ReLU(eqx.Module):
    def __call__(self, x: Array) -> Array:
        return jax.nn.relu(x)

    @classmethod
    def from_torch(cls, m):
        return cls()


@register_from_torch("torch.nn.SiLU")
class SiLU(eqx.Module):
    def __call__(self, x: Array) -> Array:
        return jax.nn.silu(x)

    @classmethod
    def from_torch(cls, m):
        return cls()


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
class Transition(eqx.Module):
    """SwiGLU MLP ``fc3(silu(fc1(x)) * fc2(x))`` with an optional pre-LayerNorm."""

    norm: LayerNorm | None
    fc1: Linear
    fc2: Linear
    fc3: Linear

    def __call__(self, x: Float[Array, "... D"]) -> Float[Array, "... O"]:
        if self.norm is not None:
            x = self.norm(x)
        return self.fc3(jax.nn.silu(self.fc1(x)) * self.fc2(x))

    @classmethod
    def from_torch(cls, m):
        return cls(
            norm=from_torch(m.norm) if m.use_norm else None,
            fc1=from_torch(m.fc1),
            fc2=from_torch(m.fc2),
            fc3=from_torch(m.fc3),
        )


@register_from_torch("bioemu2.biomoa.modules.encoders.FourierEmbedding")
class FourierEmbedding(AbstractFromTorch):
    """Fixed random Fourier features ``cos(2π (w t + b))``."""

    proj: Linear

    def __call__(self, t: Float[Array, "*B"]) -> Float[Array, "*B D"]:
        return jnp.cos(2 * math.pi * self.proj(t[..., None]))
