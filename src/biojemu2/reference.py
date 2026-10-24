"""Featurization and structure output via the upstream BioEmu2 package (``reference`` extra).

Building model inputs (tokenization, reference conformers, MSA features) and writing
structures stay on the reference implementation's numpy/RDKit/mdtraj tooling; only the
network and sampler run in JAX.
"""

from __future__ import annotations

from collections.abc import Sequence

import mdtraj
import numpy as np
import torch
from bioemu2.biomoa.utils.feature_converter import convert_densebatch_to_dictionary
from bioemu2.conditioning.method import SamplingMode
from bioemu2.data_interface import Contact, System
from bioemu2.msa.alignment import MSA
from bioemu2.sampling import prepare_sampling_inputs
from bioemu2.structure.batch import batch_collate
from bioemu2.structure.preparation import prepare_structure
from bioemu2.trajectory_io import coords_angstrom_to_trajectory
from jaxtyping import Array, Float

from biojemu2.features import Features


def featurize(
    system: System, *, msa: MSA, mode: SamplingMode, contacts: Sequence[Contact]
) -> Features:
    """Model inputs for ``system``. ``mode`` is ``"structure"`` or ``"equilibrium"``;
    ``msa`` may be a single-sequence alignment (``MSA(unpaired_a3m={chain: ">q\\nSEQ\\n"})``)
    or one from ``bioemu2.msa.alignment.get_msa`` (ColabFold server)."""
    structure = prepare_structure(system, msa=msa, mode=mode, contacts=list(contacts))
    feats = convert_densebatch_to_dictionary(batch_collate([structure]))
    return Features.from_reference_dict({k: v.numpy() for k, v in feats.items()})


def to_trajectory(system: System, coords: Float[Array, "M N 3"]) -> mdtraj.Trajectory:
    """Sampled coordinates (angstrom, including atom padding) as an mdtraj trajectory."""
    topology, _ = prepare_sampling_inputs(system, device=torch.device("cpu"))
    coords = torch.from_numpy(np.array(coords[:, : topology.n_atoms]))
    return coords_angstrom_to_trajectory(coords, topology)
