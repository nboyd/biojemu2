"""Typed, unbatched model inputs.

``Features`` holds exactly the per-system arrays the network reads. It is produced from the
reference featurizer's output dict (``bioemu2.biomoa.utils.feature_converter``) by
:meth:`Features.from_reference_dict`, the single boundary between dict-keyed reference
features and the typed JAX model (and the only place reference key names appear).
"""

from __future__ import annotations

from collections.abc import Mapping

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from biojemu2.atom import AtomInputs, TokenFeatures
from biojemu2.structure import Dynamics, StructureInputs
from biojemu2.trunk import TrunkInputs


class Features(eqx.Module):
    """One system's inputs, grouped by the stage that reads them. Masks and atom/token
    indices shared between stages appear in several groups (the same arrays)."""

    tokens: TokenFeatures
    trunk: TrunkInputs
    atoms: AtomInputs
    structure: StructureInputs
    dynamics: Dynamics

    @classmethod
    def from_reference_dict(cls, feats: Mapping[str, np.ndarray]) -> Features:
        """Build from ``convert_densebatch_to_dictionary`` output (numpy, batch size 1)."""
        for key, value in feats.items():
            if np.ndim(value) >= 1 and np.shape(value)[0] != 1:
                raise ValueError(f"expected batch size 1, got {key} with shape {np.shape(value)}")

        def get(key, dtype):
            return jnp.array(np.asarray(feats[key])[0], dtype=dtype)

        i32, f32, b = jnp.int32, jnp.float32, jnp.bool_
        token_valid = get("token_pad_mask", b)
        atom_valid = get("atom_pad_mask", b)  # True = real atom (not padding)
        atom_to_token = get("token_idx_at_atom", i32)
        return cls(
            tokens=TokenFeatures(
                res_type=get("res_type", i32),
                profile=get("profile", f32),
                deletion_mean=get("deletion_mean", f32),
                pocket_feature=get("pocket_feature", i32),
            ),
            trunk=TrunkInputs(
                method_feature=get("method_feature", i32),
                asym_id=get("asym_id", i32),
                entity_id=get("entity_id", i32),
                residue_index=get("residue_index", i32),
                sym_id=get("sym_id", i32),
                token_bonds=get("token_bonds", f32),
                contact_conditioning=get("contact_conditioning", f32),
                contact_threshold=get("contact_threshold", f32),
                token_valid=token_valid,
                msa=get("msa", f32),
                msa_mask=get("msa_mask", b),
                has_deletion=get("has_deletion", f32),
                deletion_value=get("deletion_value", f32),
                msa_paired=get("msa_paired", f32),
            ),
            atoms=AtomInputs(
                ref_pos=get("template_coords", f32),
                charge=get("charge", f32),
                element=get("element", i32),
                name_encoding=get("name_encoding", i32),
                ref_space_uid=get("ref_space_uid", i32),
                atom_valid=atom_valid,
                atom_to_token=atom_to_token,
                token_valid=token_valid,
            ),
            structure=StructureInputs(
                atom_valid=atom_valid,
                atom_mask=get("atom_mask", b),
                token_valid=token_valid,
                atom_to_token=atom_to_token,
                token_center_atom=get("token_center_atom_index", i32),
            ),
            dynamics=Dynamics(
                coords_cond=get("coords_cond", f32),
                dt=get("dt", f32).reshape(()),
                is_conditioned=get("is_dynamics_conditioned", b).reshape(()),
            ),
        )
