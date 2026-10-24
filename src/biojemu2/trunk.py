"""Sequence trunk: MSA module, PairMixer stack, conditioning, and recycling.

Everything is unbatched: ``T`` tokens, ``S`` MSA rows.
"""

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from biojemu2.backend import AbstractFromTorch, from_torch, register_from_torch, stack_modules
from biojemu2.nn import Embedding, FourierEmbedding, LayerNorm, Linear, Transition, scan_layers

_INF = 1e6  # torch reference masks attention logits with -1e6, not -inf


@register_from_torch("bioemu2.biomoa.ops.triangle_multiplication.TriangleMultiplicationFull")
@register_from_torch("bioemu2.biomoa.ops.triangle_multiplication.TriangleMultiplicationCueq")
@register_from_torch("bioemu2.biomoa.ops.triangle_multiplication.TriangleMultiplicationCueqTiled")
@register_from_torch("bioemu2.biomoa.ops.triangle_multiplication.TriangleMultiplicationOutgoing")
@register_from_torch("bioemu2.biomoa.ops.triangle_multiplication.TriangleMultiplicationIncoming")
class TriangleMultiplication(AbstractFromTorch):
    """Gated triangle multiplicative update (the "full" variant's math; all variants share it)."""

    norm_in: LayerNorm
    p_in: Linear
    g_in: Linear
    norm_out: LayerNorm
    p_out: Linear
    g_out: Linear
    mode: Literal["outgoing", "incoming"] = eqx.field(static=True)

    def __call__(self, z: Float[Array, "T T D"], mask: Bool[Array, "T T"]) -> Float[Array, "T T D"]:
        zn = self.norm_in(z)
        proj = self.p_in(zn) * jax.nn.sigmoid(self.g_in(zn))
        proj = jnp.where(mask[..., None], proj, 0.0)
        a, b = jnp.split(proj, 2, axis=-1)
        spec = "ikd,jkd->ijd" if self.mode == "outgoing" else "kid,kjd->ijd"
        out = jnp.einsum(spec, a, b)
        return self.p_out(self.norm_out(out)) * jax.nn.sigmoid(self.g_out(zn))


@register_from_torch("bioemu2.biomoa.ops.outer_product_mean.OuterProductMeanVanilla")
@register_from_torch("bioemu2.biomoa.ops.outer_product_mean.OuterProductMeanTritonMemorySaving")
class OuterProductMean(AbstractFromTorch):
    norm: LayerNorm
    proj_ab: Linear
    proj_o: Linear

    def __call__(self, m: Float[Array, "S T C"], mask: Bool[Array, "S T"]) -> Float[Array, "T T Z"]:
        num_valid = jnp.maximum(mask.any(-1).sum(), 1)
        proj = jnp.where(mask[..., None], self.proj_ab(self.norm(m)), 0.0)
        a, b = jnp.split(proj, 2, axis=-1)
        outer = jnp.einsum("sic,sjd->ijcd", a, b)
        outer = outer.reshape(*outer.shape[:2], -1) / num_valid
        return self.proj_o(outer)


@register_from_torch("bioemu2.biomoa.ops.pair_weighted_averaging.PairWeightedAveraging")
class PairWeightedAveraging(AbstractFromTorch):
    norm_m: LayerNorm
    norm_z: LayerNorm
    proj_m: Linear
    proj_g: Linear
    proj_z: Linear
    proj_o: Linear
    num_heads: int = eqx.field(static=True)

    def __call__(
        self, m: Float[Array, "S T C"], z: Float[Array, "T T Z"], mask: Bool[Array, "T T"]
    ) -> Float[Array, "S T C"]:
        m = self.norm_m(m)
        v = self.proj_m(m).reshape(*m.shape[:2], self.num_heads, -1)  # [S, T, H, Ch]
        bias = jnp.moveaxis(self.proj_z(self.norm_z(z)), -1, 0)  # [H, T, T]
        weights = jax.nn.softmax(jnp.where(mask[None], bias, -_INF), axis=-1)
        o = jnp.einsum("hij,sjhd->sihd", weights, v).reshape(*m.shape[:2], -1)
        return self.proj_o(jax.nn.sigmoid(self.proj_g(m)) * o)


@register_from_torch("bioemu2.biomoa.modules.conditioning.ContactConditioning")
class ContactConditioning(AbstractFromTorch):
    fourier_embedding: FourierEmbedding
    encoder: Linear
    encoding_unspecified: Float[Array, " Z"]
    encoding_unselected: Float[Array, " Z"]
    cutoff_min: float = eqx.field(static=True)
    cutoff_max: float = eqx.field(static=True)

    def __call__(
        self,
        contact_conditioning: Float[Array, "T T 5"],
        contact_threshold: Float[Array, "T T"],
        pair_mask: Bool[Array, "T T"],
    ) -> Float[Array, "T T Z"]:
        thr = (contact_threshold - self.cutoff_min) / (self.cutoff_max - self.cutoff_min)
        feats = jnp.concatenate(
            [contact_conditioning[..., 2:], thr[..., None], self.fourier_embedding(thr)], axis=-1
        )
        unspecified, unselected = contact_conditioning[..., 0:1], contact_conditioning[..., 1:2]
        out = (
            self.encoder(feats) * (1 - unspecified - unselected)
            + self.encoding_unspecified * unspecified
            + self.encoding_unselected * unselected
        )
        return jnp.where(pair_mask[..., None], out, 0.0)


@register_from_torch("bioemu2.biomoa.modules.conditioning.MethodConditioning")
class MethodConditioning(AbstractFromTorch):
    embedding: Embedding
    pair_embedding: Embedding | None

    def __call__(
        self, method_feature: Int[Array, " T"], token_mask: Bool[Array, " T"]
    ) -> tuple[Float[Array, "T S"], Float[Array, "T T Z"] | None]:
        single = jnp.where(token_mask[:, None], self.embedding(method_feature), 0.0)
        if self.pair_embedding is None:
            return single, None
        e = self.pair_embedding(method_feature)
        pair_mask = token_mask[:, None] & token_mask[None, :]
        return single, jnp.where(pair_mask[..., None], e[:, None] + e[None, :], 0.0)


@register_from_torch(
    "bioemu2.biomoa.modules.relative_position_encoder.FusedRelativePositionEncoder"
)
class RelativePositionEncoder(AbstractFromTorch):
    embed_rel_pos: Embedding
    embed_rel_token: Embedding
    embed_same_entity: Linear
    embed_rel_chain: Embedding
    r_max: int = eqx.field(static=True)
    s_max: int = eqx.field(static=True)

    def __call__(
        self,
        asym_id: Int[Array, " T"],
        entity_id: Int[Array, " T"],
        residue_index: Int[Array, " T"],
        sym_id: Int[Array, " T"],
    ) -> Float[Array, "T T Z"]:
        def pairwise(x):
            return x[:, None], x[None, :]

        token_index = jnp.arange(asym_id.shape[0])
        same_chain = jnp.equal(*pairwise(asym_id))
        same_residue = jnp.equal(*pairwise(residue_index))
        same_entity = jnp.equal(*pairwise(entity_id))
        r_far, s_far = 2 * self.r_max + 1, 2 * self.s_max + 1
        d_residue = jnp.clip(jnp.subtract(*pairwise(residue_index)) + self.r_max, 0, 2 * self.r_max)
        d_residue = jnp.where(same_chain, d_residue, r_far)
        d_token = jnp.clip(jnp.subtract(*pairwise(token_index)) + self.r_max, 0, 2 * self.r_max)
        d_token = jnp.where(same_chain & same_residue, d_token, r_far)
        d_chain = jnp.clip(jnp.subtract(*pairwise(sym_id)) + self.s_max, 0, 2 * self.s_max)
        d_chain = jnp.where(same_entity, d_chain, s_far)
        return (
            self.embed_rel_pos(d_residue)
            + self.embed_rel_token(d_token)
            + self.embed_same_entity(same_entity[..., None].astype(jnp.float32))
            + self.embed_rel_chain(d_chain)
        )


@register_from_torch("bioemu2.biomoa.modules.pairmixer.PairMixerLayer")
class PairMixerLayer(AbstractFromTorch):
    tri_mul_out: TriangleMultiplication
    tri_mul_in: TriangleMultiplication
    transition_z: Transition

    def __call__(
        self, z: Float[Array, "T T Z"], pair_mask: Bool[Array, "T T"]
    ) -> Float[Array, "T T Z"]:
        z = z + self.tri_mul_out(z, pair_mask)
        z = z + self.tri_mul_in(z, pair_mask)
        return z + self.transition_z(z)


@register_from_torch("bioemu2.biomoa.modules.pairmixer.PairMixerMSALayer")
class PairMixerMSALayer(AbstractFromTorch):
    pair_weighted_averaging: PairWeightedAveraging
    msa_transition: Transition
    outer_product_mean: OuterProductMean
    tri_mul_out: TriangleMultiplication
    tri_mul_in: TriangleMultiplication
    z_transition: Transition

    def __call__(
        self,
        z: Float[Array, "T T Z"],
        m: Float[Array, "S T C"],
        pair_mask: Bool[Array, "T T"],
        msa_mask: Bool[Array, "S T"],
    ) -> tuple[Float[Array, "T T Z"], Float[Array, "S T C"]]:
        m = m + self.pair_weighted_averaging(m, z, pair_mask)
        m = m + self.msa_transition(m)
        z = z + self.outer_product_mean(m, msa_mask)
        z = z + self.tri_mul_out(z, pair_mask)
        z = z + self.tri_mul_in(z, pair_mask)
        return z + self.z_transition(z), m


class TrunkInputs(eqx.Module):
    """Token-level trunk features (see ``biojemu2.features.Features``)."""

    method_feature: Int[Array, " T"]
    asym_id: Int[Array, " T"]
    entity_id: Int[Array, " T"]
    residue_index: Int[Array, " T"]
    sym_id: Int[Array, " T"]
    token_bonds: Float[Array, "T T 1"]
    contact_conditioning: Float[Array, "T T 5"]
    contact_threshold: Float[Array, "T T"]
    token_pad_mask: Bool[Array, " T"]
    msa: Float[Array, "S T 33"]
    msa_mask: Bool[Array, "S T"]
    has_deletion: Float[Array, "S T"]
    deletion_value: Float[Array, "S T"]
    msa_paired: Float[Array, "S T"]


@register_from_torch("bioemu2.biomoa.modules.pairmixer.PairMixerMSAModule")
class PairMixerMSAModule(eqx.Module):
    s_proj: Linear
    msa_proj: Linear
    layers: PairMixerMSALayer  # stacked: leading layer axis
    use_paired_feature: bool = eqx.field(static=True)

    def __call__(
        self, z: Float[Array, "T T Z"], emb: Float[Array, "T Ds"], f: TrunkInputs
    ) -> Float[Array, "T T Z"]:
        extra = [f.has_deletion, f.deletion_value] + (
            [f.msa_paired] if self.use_paired_feature else []
        )
        m = self.msa_proj(jnp.concatenate([f.msa] + [x[..., None] for x in extra], axis=-1))
        m = m + self.s_proj(emb)[None]
        pair_mask = f.token_pad_mask[:, None] & f.token_pad_mask[None, :]

        def step(layer, carry):
            return layer(*carry, pair_mask, f.msa_mask)

        return scan_layers(step, (z, m), self.layers)[0]

    @classmethod
    def from_torch(cls, m):
        return cls(
            s_proj=from_torch(m.s_proj),
            msa_proj=from_torch(m.msa_proj),
            layers=stack_modules([from_torch(layer) for layer in m.layers]),
            use_paired_feature=m.use_paired_feature,
        )


@register_from_torch("bioemu2.biomoa.modules.pairmixer.PairMixerModule")
class PairMixerModule(eqx.Module):
    layers: PairMixerLayer  # stacked: leading layer axis

    def __call__(
        self, z: Float[Array, "T T Z"], pair_mask: Bool[Array, "T T"]
    ) -> Float[Array, "T T Z"]:
        return scan_layers(lambda layer, z: layer(z, pair_mask), z, self.layers)

    @classmethod
    def from_torch(cls, m):
        return cls(layers=stack_modules([from_torch(layer) for layer in m.layers]))


@register_from_torch("bioemu2.biomoa.modules.pairmixer.PairMixerSequenceEmbedder")
class PairMixerSequenceEmbedder(AbstractFromTorch):
    s_init: Linear
    z_init_1: Linear
    z_init_2: Linear
    token_bonds: Linear
    rel_pos: RelativePositionEncoder
    msa_module: PairMixerMSAModule
    pairmixer_module: PairMixerModule
    s_recycle: Linear
    z_recycle: Linear
    s_norm: LayerNorm
    z_norm: LayerNorm
    # Optional (absent from the torch module when disabled).
    contact_conditioning: ContactConditioning | None = None
    method_conditioning: MethodConditioning | None = None
    uncond_transition: Transition | None = None

    def _initial(
        self, s_inputs: Float[Array, "T Ds"], f: TrunkInputs
    ) -> tuple[Float[Array, "T S"], Float[Array, "T T Z"]]:
        pair_mask = f.token_pad_mask[:, None] & f.token_pad_mask[None, :]
        s_init = self.s_init(s_inputs)
        z_init = self.z_init_1(s_inputs)[:, None] + self.z_init_2(s_inputs)[None, :]
        if self.method_conditioning is not None:
            single, pair = self.method_conditioning(f.method_feature, f.token_pad_mask)
            s_init = s_init + single
            if pair is not None:
                z_init = z_init + pair
        z_init = z_init + self.rel_pos(f.asym_id, f.entity_id, f.residue_index, f.sym_id)
        z_init = z_init + self.token_bonds(f.token_bonds)
        if self.contact_conditioning is not None:
            z_init = z_init + self.contact_conditioning(
                f.contact_conditioning, f.contact_threshold, pair_mask
            )
        return s_init, z_init

    def __call__(
        self, s_inputs: Float[Array, "T Ds"], f: TrunkInputs, *, num_recycles: int
    ) -> tuple[Float[Array, "T S"], Float[Array, "T T Z"]]:
        """Run ``num_recycles + 1`` trunk iterations. As in the reference, only the last
        iteration is differentiated; earlier iterations' outputs are detached."""
        s_init, z_init = self._initial(s_inputs, f)
        pair_mask = f.token_pad_mask[:, None] & f.token_pad_mask[None, :]

        def iteration(carry):
            s, z = carry
            s = s_init + self.s_recycle(self.s_norm(s))
            z = z_init + self.z_recycle(self.z_norm(z))
            z = self.msa_module(z, s_inputs, f)
            return s, self.pairmixer_module(z, pair_mask)

        carry = (jnp.zeros_like(s_init), jnp.zeros_like(z_init))
        carry = jax.lax.fori_loop(
            0, num_recycles, lambda _, c: jax.lax.stop_gradient(iteration(c)), carry
        )
        return iteration(jax.lax.stop_gradient(carry))

    def unconditional(
        self, s_inputs: Float[Array, "T Ds"], f: TrunkInputs
    ) -> tuple[Float[Array, "T S"], Float[Array, "T T Z"]]:
        """Lightweight unconditional branch (skips the MSA module and PairMixer stack)."""
        if self.uncond_transition is None:
            raise ValueError("this checkpoint has no unconditional branch")
        s_init, z_init = self._initial(s_inputs, f)
        return s_init, z_init + self.uncond_transition(self.z_norm(z_init))


@register_from_torch("bioemu2.biomoa.modules.embedders.DistogramModule")
class DistogramModule(AbstractFromTorch):
    distogram: Linear

    def __call__(self, z: Float[Array, "T T Z"]) -> Float[Array, "T T Bins"]:
        return self.distogram(z + jnp.swapaxes(z, 0, 1))
