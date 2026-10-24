"""Dual-stream (token + atom) diffusion structure module.

``StructureModule.static_context`` computes everything that does not depend on the noisy
coordinates once per system; ``StructureModule.denoise`` is the per-step network for one
sample.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from jbioemu2.atom import (
    AtomAttentionEncoder,
    AtomInputs,
    AtomStaticContext,
    DampedHarmonicTimeEmbedding,
)
from jbioemu2.backend import AbstractFromTorch, from_torch, register_from_torch, stack_modules
from jbioemu2.nn import FourierEmbedding, LayerNorm, Linear, Sequential, Transition, scan_layers
from jbioemu2.transformer import (
    ConditionedTransformerLayer,
    ada_ln,
    broadcast_tokens_to_atoms,
    gather_center_atoms,
)


@register_from_torch("bioemu2.biomoa.modules.diffusion.SingleConditioning")
class SingleConditioning(AbstractFromTorch):
    norm_single: LayerNorm
    single_embed: Linear
    fourier_embed: FourierEmbedding
    norm_fourier: LayerNorm
    fourier_to_single: Linear
    transitions: tuple[Transition, ...]
    dt_embed: DampedHarmonicTimeEmbedding | None = None
    dt_embed_to_single: Linear | None = None

    def __call__(
        self,
        times: Float[Array, ""],
        s_trunk: Float[Array, "T S"],
        s_inputs: Float[Array, "T Ds"],
        dt: Float[Array, ""],
        is_conditioned: Bool[Array, ""],
    ) -> Float[Array, "T As"]:
        s = self.single_embed(self.norm_single(jnp.concatenate([s_trunk, s_inputs], axis=-1)))
        s = s + self.fourier_to_single(self.norm_fourier(self.fourier_embed(times)))
        if self.dt_embed is not None:
            s = s + self.dt_embed_to_single(self.dt_embed(dt)) * is_conditioned
        for transition in self.transitions:
            s = s + transition(s)
        return s


@register_from_torch("bioemu2.biomoa.modules.diffusion.PairwiseConditioning")
class PairwiseConditioning(AbstractFromTorch):
    dim_pairwise_init_proj: Sequential
    transitions: tuple[Transition, ...]

    def __call__(
        self, z_trunk: Float[Array, "T T Z"], rel_pos: Float[Array, "T T Z"]
    ) -> Float[Array, "T T Az"]:
        z = self.dim_pairwise_init_proj(jnp.concatenate([z_trunk, rel_pos], axis=-1))
        for transition in self.transitions:
            z = z + transition(z)
        return z


class StructureInputs(eqx.Module):
    """Atom/token bookkeeping the per-step network needs (see ``Features``)."""

    atom_valid: Bool[Array, " N"]  # not padding
    atom_mask: Bool[Array, " N"]  # participates in coordinate operations
    token_valid: Bool[Array, " T"]
    atom_to_token: Int[Array, " N"]
    token_center_atom: Int[Array, " T"]


@register_from_torch("bioemu2.biomoa.modules.diffusion.DualStreamCoupling")
class DualStreamCoupling(AbstractFromTorch):
    s_to_q_mixer: Transition
    q_to_s_mixer: Transition
    q_coupling_cond: Sequential
    s_coupling_cond: Sequential
    lns_scale: Float[Array, ""] = eqx.field(converter=jnp.float32)  # a float in torch

    def __call__(
        self,
        s: Float[Array, "T As"],
        q: Float[Array, "N A"],
        c: Float[Array, "T As"],
        c_to_q: Float[Array, "N As"],
        f: StructureInputs,
    ) -> tuple[Float[Array, "T As"], Float[Array, "N A"]]:
        width = s.shape[-1] + q.shape[-1]
        q_alpha, q_beta, q_gamma = jnp.split(self.q_coupling_cond(c_to_q), [width, 2 * width], -1)
        s_alpha, s_beta, s_gamma = jnp.split(self.s_coupling_cond(c), [width, 2 * width], -1)
        s_to_q = jnp.concatenate(
            [broadcast_tokens_to_atoms(s, f.atom_to_token, f.atom_valid), q], axis=-1
        )
        s_to_q = q_gamma * self.s_to_q_mixer(self.lns_scale * ada_ln(s_to_q, q_alpha, q_beta))
        q_to_s = jnp.concatenate(
            [gather_center_atoms(q, f.token_center_atom, f.token_valid), s], axis=-1
        )
        q_to_s = s_gamma * self.q_to_s_mixer(self.lns_scale * ada_ln(q_to_s, s_alpha, s_beta))
        return s + q_to_s, q + s_to_q


@register_from_torch("bioemu2.biomoa.modules.diffusion.DualStreamDiffusionLayer")
class DualStreamDiffusionLayer(AbstractFromTorch):
    """Windowed atom block, full token block, then cross-stream coupling."""

    coupling: DualStreamCoupling
    atom_block: ConditionedTransformerLayer
    token_block: ConditionedTransformerLayer

    def __call__(
        self,
        s: Float[Array, "T As"],
        q: Float[Array, "N A"],
        z: Float[Array, "T T Az"],
        p: Float[Array, "NW W H P"],
        c: Float[Array, "T As"],
        f: StructureInputs,
    ) -> tuple[Float[Array, "T As"], Float[Array, "N A"]]:
        c_to_q = broadcast_tokens_to_atoms(c, f.atom_to_token, f.atom_valid)
        q = self.atom_block(q, p, c_to_q, f.atom_valid)
        s = self.token_block(s, z, c, f.token_valid)
        return self.coupling(s, q, c, c_to_q, f)


@register_from_torch("bioemu2.biomoa.modules.diffusion.ReadoutSuperposition")
class ReadoutSuperposition(AbstractFromTorch):
    s_to_hidden: Linear
    q_to_hidden: Linear
    to_out: Sequential

    def __call__(
        self, a: Float[Array, "T As"], q: Float[Array, "N A"], f: StructureInputs
    ) -> Float[Array, "N 3"]:
        hidden = self.q_to_hidden(q) + broadcast_tokens_to_atoms(
            self.s_to_hidden(a), f.atom_to_token, f.atom_valid
        )
        return self.to_out(hidden)


class StaticContext(eqx.Module):
    """Coordinate-independent structure-module inputs, computed once per system."""

    z: Float[Array, "T T Az"]
    atoms: AtomStaticContext
    inputs: StructureInputs


@register_from_torch("bioemu2.biomoa.modules.diffusion.DualStreamDiffusionModule")
class StructureModule(eqx.Module):
    s_trunk_to_a: Sequential
    single_conditioner: SingleConditioning
    pairwise_conditioner: PairwiseConditioning
    atom_feature_embedder: AtomAttentionEncoder
    r_token_to_a: Linear
    readout: ReadoutSuperposition
    layers: DualStreamDiffusionLayer  # stacked: leading layer axis

    def static_context(
        self,
        atoms: AtomInputs,
        inputs: StructureInputs,
        s_trunk: Float[Array, "T S"],
        z_trunk: Float[Array, "T T Z"],
        rel_pos: Float[Array, "T T Z"],
    ) -> StaticContext:
        return StaticContext(
            z=self.pairwise_conditioner(z_trunk, rel_pos),
            atoms=self.atom_feature_embedder.static_context(atoms, s_trunk, z_trunk),
            inputs=inputs,
        )

    def denoise(
        self,
        context: StaticContext,
        s_inputs: Float[Array, "T Ds"],
        s_trunk: Float[Array, "T S"],
        r_noisy: Float[Array, "N 3"],
        times: Float[Array, ""],
        coords_cond: Float[Array, "N 3"],
        is_conditioned: Bool[Array, ""],
        dt: Float[Array, ""],
    ) -> Float[Array, "N 3"]:
        """Network output ``F(c_in x, c_noise(sigma))`` for one sample (unitless update)."""
        f = context.inputs
        r = jnp.where(f.atom_mask[:, None], r_noisy, 0.0)
        c = self.single_conditioner(times, s_trunk, s_inputs, dt, is_conditioned)
        q = self.atom_feature_embedder.coordinate_queries(
            context.atoms, r, coords_cond, is_conditioned, dt
        )
        a = self.s_trunk_to_a(s_trunk) + self.r_token_to_a(
            gather_center_atoms(r, f.token_center_atom, f.token_valid)
        )

        def step(layer, carry):
            return layer(*carry, context.z, context.atoms.p, c, f)

        a, q = scan_layers(step, (a, q), self.layers)
        return self.readout(a, q, f)

    @classmethod
    def from_torch(cls, m):
        return cls(
            **{
                n: from_torch(getattr(m, n))
                for n in (
                    "s_trunk_to_a",
                    "single_conditioner",
                    "pairwise_conditioner",
                    "atom_feature_embedder",
                    "r_token_to_a",
                    "readout",
                )
            },
            layers=stack_modules([from_torch(layer) for layer in m.layers]),
        )
