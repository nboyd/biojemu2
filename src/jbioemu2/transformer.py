"""Attention with pair bias (full token / sliding-window atom) and adaLN transformer layers."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int

from jbioemu2.backend import from_torch, register_from_torch, register_unwrap
from jbioemu2.nn import LayerNorm, Linear, Sequential, Transition, layer_norm

_INF = 1e6  # the reference masks logits with -1e6


def ada_ln(x: Float[Array, "... D"], alpha: Float[Array, "... D"], beta: Float[Array, "... D"]):
    """``LN(x) * (1 + alpha) + beta`` with a parameter-free LayerNorm (eps 1e-5)."""
    return layer_norm(x, 1e-5) * (1 + alpha) + beta


def key_window_index(num_atoms: int, queries: int, keys: int) -> tuple[np.ndarray, np.ndarray]:
    """Static gather table for sliding-window atom attention.

    Query window ``k`` covers atoms ``[k*W, (k+1)*W)`` and attends to the ``H`` atoms
    ``k*W - (H-W)/2 + j``. Returns ``(index [NW, H], in_range [NW, H])``; out-of-range
    slots are clamped to 0 and must be treated as zero padding.
    """
    if num_atoms % queries or (keys - queries) % 2 or keys < queries:
        raise ValueError(f"bad atom window geometry: N={num_atoms}, W={queries}, H={keys}")
    context = (keys - queries) // 2
    index = np.arange(num_atoms // queries)[:, None] * queries - context + np.arange(keys)[None]
    in_range = (index >= 0) & (index < num_atoms)
    return np.where(in_range, index, 0), in_range


def window_keys(x: Float[Array, "N ..."], queries: int, keys: int) -> Float[Array, "NW H ..."]:
    """Key-window view of per-atom values, zero outside the atom range (torch pad + unfold)."""
    index, in_range = key_window_index(x.shape[0], queries, keys)
    gathered = x[index]
    return jnp.where(in_range.reshape(in_range.shape + (1,) * (x.ndim - 1)), gathered, 0)


class _PairBiasAttention(eqx.Module):
    """Shared projections of ``AttentionPairBias`` / ``AttentionPairBiasSlidingWindow``."""

    proj_qkv: Linear
    proj_q_bias: Float[Array, " C"]
    proj_g: Linear
    norm_z: LayerNorm
    proj_z: Linear
    proj_o: Linear
    num_heads: int = eqx.field(static=True)

    def _attend(self, q, k, v, bias, key_mask):
        """q: [..., Q, H, d]; k, v: [..., K, H, d]; bias: [..., H, Q, K]; key_mask: [..., K]."""
        logits = jnp.einsum("...qhd,...khd->...hqk", q, k) / np.sqrt(q.shape[-1]) + bias
        logits = jnp.where(key_mask[..., None, None, :], logits, -_INF)
        return jnp.einsum("...hqk,...khd->...qhd", jax.nn.softmax(logits, axis=-1), v)

    @classmethod
    def from_torch(cls, m):
        if getattr(m, "initial_norm", False):
            raise NotImplementedError("initial_norm=True is not used by BioEmu2")
        return cls(
            **{
                n: from_torch(getattr(m, n))
                for n in ("proj_qkv", "proj_q_bias", "proj_g", "norm_z", "proj_z", "proj_o")
            },
            num_heads=m.num_heads,
        )


@register_from_torch("bioemu2.biomoa.ops.attention_pair_bias.AttentionPairBias")
class AttentionPairBias(_PairBiasAttention):
    """Full self-attention over tokens with an additive per-head pair bias."""

    def __call__(
        self, s: Float[Array, "T C"], z: Float[Array, "T T Z"], mask: Bool[Array, " T"]
    ) -> Float[Array, "T C"]:
        T, C = s.shape
        q, k, v = jnp.split(self.proj_qkv(s), 3, axis=-1)
        heads = (T, self.num_heads, -1)
        q = (q + self.proj_q_bias).reshape(heads)
        bias = jnp.moveaxis(self.proj_z(self.norm_z(z)), -1, 0)  # [H, T, T]
        o = self._attend(q, k.reshape(heads), v.reshape(heads), bias, mask)
        return self.proj_o(jax.nn.sigmoid(self.proj_g(s)) * o.reshape(T, C))


@register_from_torch("bioemu2.biomoa.ops.attention_pair_bias.AttentionPairBiasSlidingWindow")
class AttentionPairBiasSlidingWindow(_PairBiasAttention):
    """Each window of ``W`` query atoms attends to a centred window of ``H`` key atoms."""

    def __call__(
        self, s: Float[Array, "N C"], p: Float[Array, "NW W H P"], mask: Bool[Array, " N"]
    ) -> Float[Array, "N C"]:
        N, C = s.shape
        NW, W, H = p.shape[:3]
        qkv = self.proj_qkv(s)
        q, kv = qkv[:, :C], qkv[:, C:]
        q = (q + self.proj_q_bias).reshape(NW, W, self.num_heads, -1)
        kv = window_keys(kv, W, H).reshape(NW, H, 2, self.num_heads, -1)  # zero pad (no bias)
        key_mask = window_keys(mask, W, H)  # out-of-range keys are False
        bias = jnp.moveaxis(self.proj_z(self.norm_z(p)), -1, 1)  # [NW, heads, W, H]
        o = self._attend(q, kv[:, :, 0], kv[:, :, 1], bias, key_mask)
        return self.proj_o(jax.nn.sigmoid(self.proj_g(s)) * o.reshape(N, C))


_TRANSFORMER = "bioemu2.biomoa.modules.transformer"
register_unwrap(f"{_TRANSFORMER}.AttentionLayer", "pair_bias_attn")
register_unwrap(f"{_TRANSFORMER}.SlidingWindowAttentionLayer", "attention_layer")
register_unwrap(f"{_TRANSFORMER}.ConditionedSlidingWindowTransformerLayer", "transformer_layer")


@register_from_torch("bioemu2.biomoa.modules.transformer.ConditionedTransformerLayer")
class ConditionedTransformerLayer(eqx.Module):
    """``x += gamma * op(lns * ((1 + alpha) LN(x) + beta))`` for attention, then a SwiGLU MLP;
    alpha/beta/gamma come from the conditioning ``c`` (adaLN-Zero). Token layers use full
    attention over ``pairs [T, T, Z]``; atom layers windowed attention over ``[NW, W, H, P]``."""

    cond: Sequential
    attention: AttentionPairBias | AttentionPairBiasSlidingWindow
    transition: Transition
    lns_scale: Float[Array, ""]

    def __call__(
        self,
        x: Float[Array, "L D"],
        pairs: Float[Array, "..."],
        c: Float[Array, "L Dc"],
        mask: Bool[Array, " L"],
    ) -> Float[Array, "L D"]:
        attn_a, attn_b, attn_g, mlp_a, mlp_b, mlp_g = jnp.split(self.cond(c), 6, axis=-1)
        x = x + attn_g * self.attention(self.lns_scale * ada_ln(x, attn_a, attn_b), pairs, mask)
        return x + mlp_g * self.transition(self.lns_scale * ada_ln(x, mlp_a, mlp_b))

    @classmethod
    def from_torch(cls, m):
        return cls(
            cond=from_torch(m.cond),
            attention=from_torch(m.attention),
            transition=from_torch(m.transition),
            lns_scale=jnp.float32(m.lns_scale),
        )


def broadcast_tokens_to_atoms(
    per_token: Float[Array, "T D"], atom_to_token: Int[Array, " N"], atom_valid: Bool[Array, " N"]
) -> Float[Array, "N D"]:
    """Per-token features gathered onto atoms, zero on padding atoms."""
    return jnp.where(atom_valid[:, None], per_token[atom_to_token], 0.0)


def gather_center_atoms(
    per_atom: Float[Array, "N D"], center_index: Int[Array, " T"], token_mask: Bool[Array, " T"]
) -> Float[Array, "T D"]:
    """Each token's centre-atom feature, zero on padding tokens."""
    return jnp.where(token_mask[:, None], per_atom[center_index], 0.0)


def scatter_mean_atoms_to_tokens(
    per_atom: Float[Array, "N D"],
    atom_to_token: Int[Array, " N"],
    atom_valid: Bool[Array, " N"],
    num_tokens: int,
) -> Float[Array, "T D"]:
    """Mean of per-atom features per token over non-padding atoms (``+1e-6`` in the count)."""
    seg = jnp.where(atom_valid, atom_to_token, num_tokens)  # padding -> dump slot
    total = jax.ops.segment_sum(per_atom, seg, num_segments=num_tokens + 1)[:num_tokens]
    count = jax.ops.segment_sum(jnp.ones_like(seg, jnp.float32), seg, num_segments=num_tokens + 1)
    return total / (count[:num_tokens, None] + 1e-6)
