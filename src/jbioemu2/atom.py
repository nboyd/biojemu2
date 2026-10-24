"""Atom attention encoder (input embedder + structure-module atom features) and time embeddings.

Atom arrays are unbatched with ``N`` atoms, a multiple of the query window ``W``; the
windowed pair representation has shape ``[NW, W, H, P]`` with ``NW = N // W`` query
windows attending to ``H`` key atoms each.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from jbioemu2.backend import AbstractFromTorch, from_torch, register_from_torch, stack_modules
from jbioemu2.nn import Embedding, Linear, Sequential, scan_layers
from jbioemu2.transformer import (
    ConditionedTransformerLayer,
    broadcast_tokens_to_atoms,
    key_window_index,
    scatter_mean_atoms_to_tokens,
    window_keys,
)


@register_from_torch("bioemu2.biomoa.modules.time.MultiExponentialTimeEmbedding")
class MultiExponentialTimeEmbedding(AbstractFromTorch):
    taus: Float[Array, " K"]

    def __call__(self, t: Float[Array, ""]) -> Float[Array, " K"]:
        return jnp.exp(-t / self.taus)


@register_from_torch("bioemu2.biomoa.modules.time.DampedHarmonicTimeEmbedding")
class DampedHarmonicTimeEmbedding(AbstractFromTorch):
    taus: Float[Array, " K"]
    quality_factors: Float[Array, " Q"]

    def __call__(self, t: Float[Array, ""]) -> Float[Array, " E"]:
        scaled = t / self.taus  # [K]
        envelope = jnp.exp(-scaled)
        phase = 2 * math.pi * self.quality_factors[None, :] * scaled[:, None]  # [K, Q]
        sin = (envelope[:, None] * jnp.sin(phase)).reshape(-1)
        cos = (envelope[:, None] * jnp.cos(phase)).reshape(-1)
        return jnp.concatenate([envelope, sin, cos])


class AtomStaticContext(eqx.Module):
    """Coordinate-independent atom representations: queries ``q``, conditioning ``c``,
    windowed pair representation ``p``."""

    q: Float[Array, "N A"]
    c: Float[Array, "N A"]
    p: Float[Array, "NW W H P"]


class AtomInputs(eqx.Module):
    """Reference-conformer atom features (see ``jbioemu2.features.Features``)."""

    ref_pos: Float[Array, "N 3"]
    charge: Float[Array, " N"]
    element: Int[Array, " N"]
    name_encoding: Int[Array, "N 4"]
    ref_space_uid: Int[Array, " N"]
    atom_valid: Bool[Array, " N"]
    atom_to_token: Int[Array, " N"]
    token_valid: Bool[Array, " T"]


def windowed_pair_geometry(
    ref_pos: Float[Array, "N 3"],
    atom_valid: Bool[Array, " N"],
    uid: Int[Array, " N"],
    queries: int,
    keys: int,
) -> tuple[Float[Array, "NW W H 3"], Float[Array, "NW W H 1"], Bool[Array, "NW W H 1"]]:
    """Key-minus-query displacement, ``1 / (1 + |d|^2)``, and same-reference-space mask."""
    NW = ref_pos.shape[0] // queries
    disp = window_keys(ref_pos, queries, keys)[:, None] - ref_pos.reshape(NW, queries, 1, 3)
    inv_dist = 1 / (1 + jnp.sum(disp * disp, axis=-1, keepdims=True))
    key_uid = window_keys(jnp.where(atom_valid, uid, -1) + 1, queries, keys) - 1  # pad -> -1
    same = atom_valid.reshape(NW, queries, 1) & (uid.reshape(NW, queries, 1) == key_uid[:, None])
    return disp, inv_dist, same[..., None]


def broadcast_token_pair_to_atom_window(
    z: Float[Array, "T T D"],
    atom_to_token: Int[Array, " N"],
    atom_valid: Bool[Array, " N"],
    queries: int,
    keys: int,
) -> Float[Array, "NW W H D"]:
    """``out[k, w, l] = z[token(query k, w), token(key k, l)]``, zero on padding/out of range."""
    NW = atom_to_token.shape[0] // queries
    index, in_range = key_window_index(atom_to_token.shape[0], queries, keys)
    tok_q = atom_to_token.reshape(NW, queries)
    tok_k = atom_to_token[index]
    valid = atom_valid.reshape(NW, queries, 1) & (atom_valid[index] & in_range)[:, None]
    return jnp.where(valid[..., None], z[tok_q[:, :, None], tok_k[:, None, :]], 0.0)


@register_from_torch("bioemu2.biomoa.modules.encoders.AtomAttentionEncoder")
class AtomAttentionEncoder(eqx.Module):
    proj_atom_continuous: Linear
    embed_element: Embedding
    embed_atom_name_chars: tuple[Embedding, ...]
    embed_atompair_ref_pos: Linear
    embed_atompair_ref_dist: Linear
    embed_atompair_mask: Linear
    c_to_p_trans_q: Sequential
    c_to_p_trans_k: Sequential
    p_mlp: Sequential
    # Structure-module only (``structure_prediction=True``).
    s_to_c_trans: Sequential | None
    z_to_p_trans: Sequential | None
    r_to_q_trans: Linear | None
    cond_to_q_trans: Linear | None
    dt_gate_embed: MultiExponentialTimeEmbedding | None
    dt_gate_log_w: Float[Array, " K"] | None
    # Input-embedder only.
    atom_encoder_layers: ConditionedTransformerLayer | None  # stacked
    atom_to_token_trans: Sequential | None
    queries: int = eqx.field(static=True)
    keys: int = eqx.field(static=True)

    def static_context(
        self,
        f: AtomInputs,
        s_trunk: Float[Array, "T S"] | None = None,
        z_trunk: Float[Array, "T T Z"] | None = None,
    ) -> AtomStaticContext:
        W, H = self.queries, self.keys
        NW = f.ref_pos.shape[0] // W
        c = self.proj_atom_continuous(jnp.concatenate([f.ref_pos, f.charge[:, None]], axis=-1))
        c = c + self.embed_element(f.element)
        for i, embed in enumerate(self.embed_atom_name_chars):
            c = c + embed(f.name_encoding[:, i])
        c = jnp.where(f.atom_valid[:, None], c, 0.0)

        disp, inv_dist, same = windowed_pair_geometry(
            f.ref_pos, f.atom_valid, f.ref_space_uid, W, H
        )
        p = (
            self.embed_atompair_ref_pos(disp)
            + self.embed_atompair_ref_dist(inv_dist)
            + self.embed_atompair_mask(same.astype(jnp.float32))
        )
        p = jnp.where(same, p, 0.0)
        q = c
        if self.s_to_c_trans is not None:
            c = c + broadcast_tokens_to_atoms(
                self.s_to_c_trans(s_trunk), f.atom_to_token, f.atom_valid
            )
            p = p + broadcast_token_pair_to_atom_window(
                self.z_to_p_trans(z_trunk), f.atom_to_token, f.atom_valid, W, H
            )
        p = p + self.c_to_p_trans_q(c.reshape(NW, W, 1, -1))
        p = p + self.c_to_p_trans_k(window_keys(c, W, H)[:, None])
        p = p + self.p_mlp(p)
        return AtomStaticContext(q=q, c=c, p=p)

    def coordinate_queries(
        self,
        context: AtomStaticContext,
        r: Float[Array, "N 3"],
        coords_cond: Float[Array, "N 3"],
        is_conditioned: Bool[Array, ""],
        dt: Float[Array, ""],
    ) -> Float[Array, "N A"]:
        """Structure-module atom queries: static features + noisy coordinates (+ dynamics)."""
        q = context.q + self.r_to_q_trans(r)
        if self.cond_to_q_trans is None:
            return q
        cond = self.cond_to_q_trans(coords_cond)
        if self.dt_gate_embed is not None:
            gate = jnp.sum(self.dt_gate_embed(dt) * jax.nn.softplus(self.dt_gate_log_w))
            cond = cond * gate
        return q + jnp.where(is_conditioned, cond, 0.0)

    def encode_tokens(self, context: AtomStaticContext, f: AtomInputs) -> Float[Array, "T S"]:
        """Input embedder: windowed atom transformer, then mean-pool atoms onto tokens."""
        q = scan_layers(
            lambda layer, q: layer(q, context.p, context.c, f.atom_valid),
            context.q,
            self.atom_encoder_layers,
        )
        return scatter_mean_atoms_to_tokens(
            self.atom_to_token_trans(q), f.atom_to_token, f.atom_valid, f.token_valid.shape[0]
        )

    @classmethod
    def from_torch(cls, m):
        def opt(name):  # optional submodules exist exactly when their flag is set
            return from_torch(getattr(m, name)) if hasattr(m, name) else None

        return cls(
            proj_atom_continuous=from_torch(m.proj_atom_continuous),
            embed_element=from_torch(m.embed_element),
            embed_atom_name_chars=from_torch(m.embed_atom_name_chars),
            embed_atompair_ref_pos=from_torch(m.embed_atompair_ref_pos),
            embed_atompair_ref_dist=from_torch(m.embed_atompair_ref_dist),
            embed_atompair_mask=from_torch(m.embed_atompair_mask),
            c_to_p_trans_q=from_torch(m.c_to_p_trans_q),
            c_to_p_trans_k=from_torch(m.c_to_p_trans_k),
            p_mlp=from_torch(m.p_mlp),
            s_to_c_trans=opt("s_to_c_trans"),
            z_to_p_trans=opt("z_to_p_trans"),
            r_to_q_trans=opt("r_to_q_trans"),
            cond_to_q_trans=opt("cond_to_q_trans"),
            dt_gate_embed=opt("dt_gate_embed"),
            dt_gate_log_w=opt("dt_gate_log_w"),
            atom_encoder_layers=(
                stack_modules([from_torch(x) for x in m.atom_encoder_layers])
                if m.use_atom_encoder
                else None
            ),
            atom_to_token_trans=opt("atom_to_token_trans"),
            queries=m.atoms_per_window_queries,
            keys=m.atoms_per_window_keys,
        )


class TokenFeatures(eqx.Module):
    res_type: Int[Array, " T"]
    profile: Float[Array, "T 33"]
    deletion_mean: Float[Array, " T"]
    pocket_feature: Int[Array, " T"]


@register_from_torch("bioemu2.biomoa.modules.embedders.InputEmbedder")
class InputEmbedder(AbstractFromTorch):
    atom_attention_encoder: AtomAttentionEncoder
    num_token_types: int = eqx.field(static=True)
    num_pocket_classes: int = eqx.field(static=True)

    def __call__(self, atoms: AtomInputs, tokens: TokenFeatures) -> Float[Array, "T Ds"]:
        """``[atom encoding, one-hot res type, profile, deletion mean, one-hot pocket]``."""
        a = self.atom_attention_encoder.encode_tokens(
            self.atom_attention_encoder.static_context(atoms), atoms
        )
        return jnp.concatenate(
            [
                a,
                jax.nn.one_hot(tokens.res_type, self.num_token_types),
                tokens.profile,
                tokens.deletion_mean[:, None],
                jax.nn.one_hot(tokens.pocket_feature, self.num_pocket_classes),
            ],
            axis=-1,
        )
