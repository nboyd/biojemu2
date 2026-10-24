# jbioemu2

A JAX/[Equinox](https://github.com/patrick-kidger/equinox) port of the
[BioEmu2](https://github.com/microsoft/bioemu2) score model and its EDM sampler
(reference commit `06be369`, checkpoint `bioemu2-preview`).

BioEmu2 is an all-atom diffusion model (≈512M parameters) for proteins, nucleic acids and
ligands: a PairMixer sequence trunk (MSA module + 48 triangle-multiplication blocks,
3 recycles) conditions a dual-stream diffusion module (16 layers of windowed atom attention
and full token attention coupled per layer) that denoises Cartesian atom coordinates.

## What is ported

| Ported (JAX) | Kept on the reference (`reference` extra) | Not ported |
|---|---|---|
| Input embedder + atom attention encoder | Featurization (`System`/MSA → `Features`) | Steering samplers (`edm_gd`, `edm_fk`, `edm_fkc`) and potentials |
| PairMixer trunk, MSA module, contact/method conditioning, recycling | MSA search (ColabFold) | Context-parallel (multi-GPU) inference |
| Dual-stream diffusion structure module, EDM preconditioning | Structure output (mdtraj / CIF / XTC) | Physicality checks, MD relaxation |
| Dynamics (ITO) conditioning inputs: `coords_cond`, `dt` | | Propagation workflow helpers (the network supports it; see below) |
| EDM sampler (stochastic churn, augmentation, Kabsch alignment) | | Embedding disk cache, bf16 autocast |
| Weight conversion, torch-free save/load | | |

`import jbioemu2` never imports torch; only `jbioemu2.convert` and `jbioemu2.reference` do.

## Install

```bash
uv sync --extra reference          # CPU JAX + reference tooling (torch CPU, bioemu2)
uv sync --extra reference --extra cuda   # + CUDA 12 JAX
```

## Usage

```python
import jax
from bioemu2.data_interface import System
from bioemu2.msa.alignment import MSA

from jbioemu2 import EDMSampler, save_model, load_model, sample
from jbioemu2.convert import convert_released          # needs `reference`
from jbioemu2.reference import featurize, to_trajectory

model = convert_released()            # downloads microsoft/bioemu2 and converts (~10 s)
save_model(model, "bioemu2_preview")  # later: load_model("bioemu2_preview"), torch-free

seq = "MKTAYIAKQRQISFVKSHFSRQLE"
system = System.from_description(proteins=[seq])
msa = MSA(unpaired_a3m={"A": f">q\n{seq}\n"})  # or bioemu2.msa.alignment.get_msa(system)
features = featurize(system, msa=msa, mode="structure", contacts=[])

trunk = model.embed(features, num_recycles=3)          # once per system
context = model.static_context(features, trunk)       # once per system
sampler = EDMSampler.structure_prediction(num_steps=200)  # or EDMSampler.equilibrium(...)
keys = jax.random.split(jax.random.key(0), 8)
coords = jax.vmap(lambda k: sample(model, context, trunk, features.dynamics,
                                   features.atom_mask, sampler, key=k))(keys)
to_trajectory(system, coords).save_xtc("samples.xtc")
```

See `examples/sample.py` for a runnable script. Everything is unbatched (one system;
`T` tokens, `N` atoms); use `jax.vmap` over keys for several samples, which shares the
trunk and static context exactly like the reference's `multiplicity`. Models are always
passed as arguments to jitted functions.

### API

```python
BioEmu2.embed(features: Features, *, num_recycles: int) -> TrunkEmbeddings
BioEmu2.static_context(features: Features, trunk: TrunkEmbeddings) -> StaticContext
BioEmu2.denoise(context, trunk, x: f32[N, 3], sigma: f32[], dynamics: Dynamics) -> f32[N, 3]  # x0, Å

EDMSampler(num_steps, step_scale, noise_scale, gamma_0, gamma_min, rho, align)
EDMSampler.structure_prediction(num_steps)  # step_scale 1.5
EDMSampler.equilibrium(num_steps)           # step_scale 1.0
draw_noise(key, *, num_steps, num_atoms) -> Noise
sample_from_noise(model, context, trunk, dynamics, atom_mask, sampler, noise) -> f32[N, 3]  # deterministic
sample(model, context, trunk, dynamics, atom_mask, sampler, *, key) -> f32[N, 3]

Features.from_reference_dict(feats)   # boundary adapter from bioemu2's feature dict
convert_checkpoint(checkpoint_path, config_path) / convert_released(name)
save_model(model, path) / load_model(path)
```

### The generative model

- **What is diffused:** all-atom Cartesian coordinates (Å), EDM-preconditioned with
  `sigma_data = 16 Å` (`c_noise = log(σ/σ_data)/4`, as in AF3). The network predicts the
  denoised coordinates x0.
- **Schedule:** Karras ρ = 7 from `160·16 Å` to `4e-4·16 Å`, then 0.
- **Sampler:** stochastic-churn Euler (AF3 Algorithm 18): per step, masked-centre + random
  rotation/translation of the sample (and of `coords_cond`), churn `γ0 = 0.8` while the next
  σ > 1, denoise, Kabsch-align the noisy sample onto x0, Euler step scaled by `step_scale`.
  It is an SDE sampler with one network evaluation per step.
- **Dynamics conditioning (implicit transfer operator):** `Features.dynamics` carries a
  companion frame `coords_cond`, lag `dt` and a flag; with the flag set the network
  conditions on the frame (the reference's `propagate`). Equilibrium sampling and structure
  prediction leave it off.

## Parity with the PyTorch reference

All on CPU, float32, `"highest"` matmul precision, real converted `bioemu2-preview` weights
(the torch reference runs the plain-einsum triangle-multiplication variant; all variants
share weights). Errors are max |Δ| / max |reference|.

| Level | Max relative error |
|---|---|
| 15 per-module tests, random weights and inputs (`tests/test_modules.py`) | ≤ 2.8e-7 |
| Trunk outputs (s_inputs, single, pair, rel. pos., distogram), 3 recycles | ≤ 1.5e-6 |
| Static context (pair conditioning, atom q / c / p) | ≤ 1.3e-6 |
| One denoising step, σ = 37 Å and 2 Å | ≤ 3.1e-6 |
| 20-step sampling, replayed torch noise, 2 samples | 1.4e-3 Å of 16 Å (protein), 1.2e-3 Å of 18 Å (protein + ATP) |

Checkpoint cases: a 24-residue protein (equilibrium mode) and the same protein with ATP
(structure mode, 55 tokens / 234 atoms). Gradients flow through the trunk and the full
sampler (`jax.grad` of a coordinate loss is finite).

Numerical notes:

- **Few-step samplers are chaotic.** At 6 steps with `step_scale = 1.5` the sample has not
  converged and a 1e-6 relative change of the initial noise moves atoms by ~5 Å, so torch
  and JAX trajectories diverge at the fp32 level. At 20+ steps they agree to ~1e-3 Å.
- **Alignment is detached**, as in the reference (`weighted_rigid_align` detaches its
  output): gradients reach the next step only through the denoised coordinates.
- The churn noise level is clamped at 0 (`sqrt(max(t̂² − σ², 0))`); the reference computes it
  in float64 on the host, and without the clamp XLA's fused multiply-add can produce NaN
  when γ = 0.

## Tests

```bash
uv run pytest                       # pure-JAX unit tests
uv run --extra reference pytest -m reference   # module + checkpoint parity (downloads weights)
```

Reference tests fail (they do not skip) if torch, bioemu2 or the checkpoint is unavailable.

## License

MIT, as the upstream BioEmu2 code (Copyright (c) Microsoft Corporation). Portions derive
from Boltz (MIT) and PyTorch3D (BSD) via BioEmu2; see `NOTICE` and `licenses/`. Model
weights come from [microsoft/bioemu2](https://huggingface.co/microsoft/bioemu2) and are not
redistributed.
