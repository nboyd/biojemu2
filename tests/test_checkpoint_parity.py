"""End-to-end parity vs the torch reference with the released checkpoint.

Needs the ``reference`` extra and the ``microsoft/bioemu2`` checkpoint (downloaded on first
use). Run with ``pytest -m reference``. Compares trunk embeddings, static context and
denoising, and a full EDM sampling run whose random draws are recorded from torch and
replayed through :func:`biojemu2.sample_from_noise`.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from bioemu2.biomoa.utils.feature_converter import convert_densebatch_to_dictionary
from bioemu2.checkpoints import get_checkpoint_files
from bioemu2.data_interface import System
from bioemu2.denoising import cp_sync, edm
from bioemu2.model import load_model_for_inference
from bioemu2.msa.alignment import MSA
from bioemu2.structure.batch import batch_collate
from bioemu2.structure.preparation import prepare_structure

import biojemu2.convert as convert
from biojemu2 import EDMSampler, Features, Noise, sample_from_noise

pytestmark = pytest.mark.reference
jax.config.update("jax_default_matmul_precision", "highest")
torch.set_float32_matmul_precision("highest")

SEQUENCE = "MKTAYIAKQRQISFVKSHFSRQLE"
# Few-step samplers (especially step_scale=1.5) have not converged and are chaotic: a 1e-6
# perturbation of the initial noise moves 6-step samples by angstroms. 20 steps converge.
NUM_STEPS, NUM_SAMPLES, NUM_RECYCLES = 20, 2, 3


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    files = get_checkpoint_files("bioemu2-preview")
    # Same weights; CPU torch has no cuEquivariance, so use the plain-einsum TriMul variant.
    config = tmp_path_factory.mktemp("config") / "config.yaml"
    config.write_text(
        files.config_path.read_text().replace("trimul_variant: cueq", "trimul_variant: full")
    )
    reference = load_model_for_inference(
        checkpoint_path=files.checkpoint_path,
        config_path=config,
        device=torch.device("cpu"),
        num_steps_recycle=NUM_RECYCLES,
        precision="32-true",
    )
    with torch.no_grad():
        return reference, convert.from_torch(reference.score_model)


def _record_sampling(reference, batch, trunk, step_scale):
    draws = {"randn": [], "aug": []}
    randn, aug = cp_sync.randn, cp_sync.random_augmentation

    def record_randn(*args, **kwargs):
        draws["randn"].append(randn(*args, **kwargs).clone())
        return draws["randn"][-1]

    def record_aug(*args, **kwargs):
        draws["aug"].append(tuple(x.clone() for x in aug(*args, **kwargs)))
        return draws["aug"][-1]

    cp_sync.randn, cp_sync.random_augmentation = record_randn, record_aug
    try:
        torch.manual_seed(0)
        with torch.no_grad():
            coords = edm.edm_sampler(
                batch=batch,
                num_steps=NUM_STEPS,
                score_model=reference,
                trunk_embeddings=trunk,
                step_scale=step_scale,
                noise_scale=1.0,
                multiplicity=NUM_SAMPLES,
            ).coords.numpy()
    finally:
        cp_sync.randn, cp_sync.random_augmentation = randn, aug
    churn = torch.stack(draws["randn"][1:]).swapaxes(0, 1).numpy()
    rotations = torch.stack([r for r, _ in draws["aug"]]).swapaxes(0, 1).numpy()
    translations = torch.stack([t[:, 0] for _, t in draws["aug"]]).swapaxes(0, 1).numpy()
    noise = Noise(
        init=jnp.array(draws["randn"][0].numpy()),
        churn=jnp.array(churn),
        rotations=jnp.array(rotations),
        translations=jnp.array(translations),
    )
    return coords, noise


@pytest.fixture(
    scope="module",
    params=[
        pytest.param(("equilibrium", None), id="protein-equilibrium"),
        pytest.param(("structure", "ATP"), id="protein-ATP-structure"),
    ],
)
def case(request, models):
    mode, ligand = request.param
    reference, _ = models
    system = System.from_description(
        proteins=[SEQUENCE], ligand_ccd_codes=[ligand] if ligand else None
    )
    msa = MSA(unpaired_a3m={"A": f">query\n{SEQUENCE}\n"})
    batch = batch_collate([prepare_structure(system, msa=msa, mode=mode, contacts=[])])
    feats = convert_densebatch_to_dictionary(batch)
    # Noised reference coordinates: in-distribution inputs at a high and a low noise level.
    torch.manual_seed(1)
    sigmas = torch.tensor([37.0, 2.0])
    coords = batch.template_coords + sigmas[:, None, None] * torch.randn(
        NUM_SAMPLES, batch.coords.shape[1], 3
    )
    with torch.no_grad():
        trunk = reference._run_trunk(batch)
        context = reference.compute_static_context(batch, trunk, multiplicity=NUM_SAMPLES)
        denoised = reference.denoise(context, trunk, coords, sigmas).denoised_coords.numpy()
    step_scale = 1.5 if mode == "structure" else 1.0
    sampled, noise = _record_sampling(reference, batch, trunk, step_scale)
    return {
        "features": Features.from_reference_dict({k: v.numpy() for k, v in feats.items()}),
        "trunk": trunk,
        "context": context,
        "coords": coords.numpy(),
        "sigmas": sigmas.numpy(),
        "denoised": denoised,
        "sampled": sampled,
        "noise": noise,
        "step_scale": step_scale,
    }


def rel_err(ours, ref):
    ours, ref = np.asarray(ours), np.asarray(ref)
    assert ours.shape == ref.shape
    err = np.abs(ours - ref).max() / np.abs(ref).max()
    print(f"rel err {err:.1e}", end="  ")
    return err


def test_forward(models, case):
    _, model = models
    f = case["features"]
    trunk = model.embed(f, num_recycles=NUM_RECYCLES)
    for name in ("s_inputs", "single", "pair", "relative_position_encoding", "distogram"):
        assert rel_err(getattr(trunk, name), getattr(case["trunk"], name)[0].numpy()) < 2e-5, name
    context = model.static_context(f, trunk)
    assert rel_err(context.z, case["context"].z[0].numpy()) < 2e-5
    atoms = case["context"].atom_encoder_context
    for name in ("q", "c", "p"):
        assert rel_err(getattr(context.atoms, name), getattr(atoms, name)[0].numpy()) < 2e-5, name
    denoise = jax.vmap(model.denoise, in_axes=(None, 0, 0, None))
    x0 = denoise(context, jnp.array(case["coords"]), jnp.array(case["sigmas"]), f.dynamics)
    assert rel_err(x0, case["denoised"]) < 2e-5


def test_sampling_with_replayed_noise(models, case):
    _, model = models
    f = case["features"]
    trunk = model.embed(f, num_recycles=NUM_RECYCLES)
    context = model.static_context(f, trunk)
    sampler = EDMSampler(
        NUM_STEPS,
        step_scale=case["step_scale"],
        noise_scale=1.0,
        gamma_0=0.8,
        gamma_min=1.0,
        rho=7.0,
        align=True,
    )
    run = jax.vmap(sample_from_noise, in_axes=(None, None, None, None, 0))
    ours = np.asarray(run(model, context, f.dynamics, sampler, case["noise"]))
    mask = np.asarray(f.structure.atom_mask)
    ref = case["sampled"]
    scale = np.abs(ref[:, mask]).max()
    print(f"sample max|Δ| {np.abs(ours - ref)[:, mask].max():.1e} Å of {scale:.1f} Å", end="  ")
    assert np.abs(ours - ref)[:, mask].max() / scale < 1e-4
