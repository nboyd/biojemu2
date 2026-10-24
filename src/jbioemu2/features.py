"""Typed, unbatched model inputs.

``Features`` holds exactly the per-system arrays the network reads. It is produced from the
reference featurizer's output dict (``bioemu2.biomoa.utils.feature_converter``) by
:meth:`Features.from_reference_dict`, the single boundary between dict-keyed reference
features and the typed JAX model.
"""

from __future__ import annotations

from collections.abc import Mapping

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int

from jbioemu2.atom import AtomInputs, TokenFeatures
from jbioemu2.structure import StructureInputs
from jbioemu2.trunk import TrunkInputs


class Dynamics(eqx.Module):
    """Implicit-transfer-operator conditioning: a companion frame and lag time.

    ``is_conditioned=False`` (equilibrium sampling and structure prediction) zeroes both
    contributions inside the network.
    """

    coords_cond: Float[Array, "N 3"]
    dt: Float[Array, ""]
    is_conditioned: Bool[Array, ""]


class Features(eqx.Module):
    # Tokens.
    res_type: Int[Array, " T"]
    residue_index: Int[Array, " T"]
    asym_id: Int[Array, " T"]
    entity_id: Int[Array, " T"]
    sym_id: Int[Array, " T"]
    method_feature: Int[Array, " T"]
    pocket_feature: Int[Array, " T"]
    token_pad_mask: Bool[Array, " T"]
    token_center_atom_index: Int[Array, " T"]
    profile: Float[Array, "T 33"]
    deletion_mean: Float[Array, " T"]
    token_bonds: Float[Array, "T T 1"]
    contact_conditioning: Float[Array, "T T 5"]
    contact_threshold: Float[Array, "T T"]
    # MSA (rows first).
    msa: Float[Array, "S T 33"]
    msa_mask: Bool[Array, "S T"]
    has_deletion: Float[Array, "S T"]
    deletion_value: Float[Array, "S T"]
    msa_paired: Float[Array, "S T"]
    # Atoms (padded to a multiple of the atom query window).
    template_coords: Float[Array, "N 3"]
    charge: Float[Array, " N"]
    element: Int[Array, " N"]
    name_encoding: Int[Array, "N 4"]
    ref_space_uid: Int[Array, " N"]
    atom_pad_mask: Bool[Array, " N"]  # True = real atom (not padding)
    atom_mask: Bool[Array, " N"]  # True = participates in coordinate operations
    token_idx_at_atom: Int[Array, " N"]
    dynamics: Dynamics

    @property
    def num_atoms(self) -> int:
        return self.atom_pad_mask.shape[0]

    def trunk_inputs(self) -> TrunkInputs:
        return TrunkInputs(
            method_feature=self.method_feature,
            asym_id=self.asym_id,
            entity_id=self.entity_id,
            residue_index=self.residue_index,
            sym_id=self.sym_id,
            token_bonds=self.token_bonds,
            contact_conditioning=self.contact_conditioning,
            contact_threshold=self.contact_threshold,
            token_pad_mask=self.token_pad_mask,
            msa=self.msa,
            msa_mask=self.msa_mask,
            has_deletion=self.has_deletion,
            deletion_value=self.deletion_value,
            msa_paired=self.msa_paired,
        )

    def atom_inputs(self) -> AtomInputs:
        return AtomInputs(
            ref_pos=self.template_coords,
            charge=self.charge,
            element=self.element,
            name_encoding=self.name_encoding,
            ref_space_uid=self.ref_space_uid,
            atom_valid=self.atom_pad_mask,
            atom_to_token=self.token_idx_at_atom,
            token_valid=self.token_pad_mask,
        )

    def token_features(self) -> TokenFeatures:
        return TokenFeatures(
            res_type=self.res_type,
            profile=self.profile,
            deletion_mean=self.deletion_mean,
            pocket_feature=self.pocket_feature,
        )

    def structure_inputs(self) -> StructureInputs:
        return StructureInputs(
            atom_valid=self.atom_pad_mask,
            atom_mask=self.atom_mask,
            token_valid=self.token_pad_mask,
            atom_to_token=self.token_idx_at_atom,
            token_center_atom=self.token_center_atom_index,
        )

    @classmethod
    def from_reference_dict(cls, feats: Mapping[str, np.ndarray]) -> Features:
        """Build from ``convert_densebatch_to_dictionary`` output (numpy, batch size 1)."""
        for key, value in feats.items():
            if np.ndim(value) >= 1 and np.shape(value)[0] != 1:
                raise ValueError(f"expected batch size 1, got {key} with shape {np.shape(value)}")

        def get(key, dtype):
            return jnp.array(np.asarray(feats[key])[0], dtype=dtype)

        i32, f32, b = jnp.int32, jnp.float32, jnp.bool_
        return cls(
            res_type=get("res_type", i32),
            residue_index=get("residue_index", i32),
            asym_id=get("asym_id", i32),
            entity_id=get("entity_id", i32),
            sym_id=get("sym_id", i32),
            method_feature=get("method_feature", i32),
            pocket_feature=get("pocket_feature", i32),
            token_pad_mask=get("token_pad_mask", b),
            token_center_atom_index=get("token_center_atom_index", i32),
            profile=get("profile", f32),
            deletion_mean=get("deletion_mean", f32),
            token_bonds=get("token_bonds", f32),
            contact_conditioning=get("contact_conditioning", f32),
            contact_threshold=get("contact_threshold", f32),
            msa=get("msa", f32),
            msa_mask=get("msa_mask", b),
            has_deletion=get("has_deletion", f32),
            deletion_value=get("deletion_value", f32),
            msa_paired=get("msa_paired", f32),
            template_coords=get("template_coords", f32),
            charge=get("charge", f32),
            element=get("element", i32),
            name_encoding=get("name_encoding", i32),
            ref_space_uid=get("ref_space_uid", i32),
            atom_pad_mask=get("atom_pad_mask", b),
            atom_mask=get("atom_mask", b),
            token_idx_at_atom=get("token_idx_at_atom", i32),
            dynamics=Dynamics(
                coords_cond=get("coords_cond", f32),
                dt=get("dt", f32).reshape(()),
                is_conditioned=get("is_dynamics_conditioned", b).reshape(()),
            ),
        )
