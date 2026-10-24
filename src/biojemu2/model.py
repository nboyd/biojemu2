"""BioEmu2 score model: sequence trunk + dual-stream diffusion structure module.

Three stages, each a pure function of the model and its inputs:

1. :meth:`BioEmu2.embed` -- sequence trunk (run once per system).
2. :meth:`BioEmu2.static_context` -- coordinate-independent structure-module inputs
   (once per system).
3. :meth:`BioEmu2.denoise` -- EDM-preconditioned x0 prediction for one noisy sample
   (every diffusion step).
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from biojemu2.atom import InputEmbedder
from biojemu2.backend import from_torch, register_from_torch
from biojemu2.features import Features
from biojemu2.structure import Dynamics, StaticContext, StructureModule
from biojemu2.trunk import DistogramModule, PairMixerSequenceEmbedder, RelativePositionEncoder


class TrunkEmbeddings(eqx.Module):
    s_inputs: Float[Array, "T Ds"]
    single: Float[Array, "T S"]
    pair: Float[Array, "T T Z"]
    relative_position_encoding: Float[Array, "T T Z"]
    distogram: Float[Array, "T T Bins"]


@register_from_torch("bioemu2.biomoa.models.score_model.BioMoaCModel")
class BioEmu2(eqx.Module):
    input_embedder: InputEmbedder
    sequence_embedder: PairMixerSequenceEmbedder
    rel_pos: RelativePositionEncoder
    distogram_module: DistogramModule
    structure_module: StructureModule
    sigma_data: float = eqx.field(static=True)
    sigma_min: float = eqx.field(static=True)
    sigma_max: float = eqx.field(static=True)

    def embed(self, features: Features, *, num_recycles: int) -> TrunkEmbeddings:
        """Run the sequence trunk with ``num_recycles`` recycling iterations (the reference
        inference default is 3)."""
        f = features.trunk
        s_inputs = self.input_embedder(features.atoms, features.tokens)
        single, pair = self.sequence_embedder(s_inputs, f, num_recycles=num_recycles)
        return TrunkEmbeddings(
            s_inputs=s_inputs,
            single=single,
            pair=pair,
            relative_position_encoding=self.rel_pos(
                f.asym_id, f.entity_id, f.residue_index, f.sym_id
            ),
            distogram=self.distogram_module(pair),
        )

    def static_context(self, features: Features, trunk: TrunkEmbeddings) -> StaticContext:
        return self.structure_module.static_context(
            features.atoms,
            features.structure,
            trunk.s_inputs,
            trunk.single,
            trunk.pair,
            trunk.relative_position_encoding,
        )

    # EDM preconditioning (Karras et al. 2022, Table 1; c_noise divides by sigma_data as in AF3).
    def c_skip(self, sigma):
        return self.sigma_data**2 / (sigma**2 + self.sigma_data**2)

    def c_out(self, sigma):
        return sigma * self.sigma_data / jnp.sqrt(self.sigma_data**2 + sigma**2)

    def c_in(self, sigma):
        return 1 / jnp.sqrt(sigma**2 + self.sigma_data**2)

    def c_noise(self, sigma):
        return jnp.log(sigma / self.sigma_data) * 0.25

    def denoise(
        self,
        context: StaticContext,
        x: Float[Array, "N 3"],
        sigma: Float[Array, ""],
        dynamics: Dynamics,
    ) -> Float[Array, "N 3"]:
        """Denoised coordinates (angstrom) for noisy coordinates ``x`` at noise level ``sigma``."""
        update = self.structure_module.denoise(
            context, self.c_in(sigma) * x, self.c_noise(sigma), dynamics
        )
        return self.c_skip(sigma) * x + self.c_out(sigma) * update

    @classmethod
    def from_torch(cls, m):
        schedule = m.edm_schedule
        return cls(
            input_embedder=from_torch(m.input_embedder),
            sequence_embedder=from_torch(m.sequence_embedder),
            rel_pos=from_torch(m.rel_pos),
            distogram_module=from_torch(m.distogram_module),
            structure_module=from_torch(m.structure_module),
            sigma_data=float(schedule.sigma_data),
            sigma_min=float(schedule.sigma_min),
            sigma_max=float(schedule.sigma_max),
        )
