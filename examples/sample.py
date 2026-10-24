"""Predict / sample structures for a small protein with biojemu2.

    uv run --extra reference python examples/sample.py --out outputs/example

Converts the released checkpoint on first use (cached next to the output as ``.eqx``).
"""

import argparse
from pathlib import Path

import equinox as eqx
import jax
from bioemu2 import save_frames
from bioemu2.data_interface import System
from bioemu2.msa.alignment import MSA

from biojemu2 import EDMSampler, load_model, sample, save_model
from biojemu2.reference import featurize, to_trajectory

parser = argparse.ArgumentParser()
parser.add_argument("--sequence", default="MKTAYIAKQRQISFVKSHFSRQLE")
parser.add_argument("--mode", choices=["structure", "equilibrium"], default="structure")
parser.add_argument("--num-samples", type=int, default=4)
parser.add_argument("--num-steps", type=int, default=200)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()

weights = args.out / "bioemu2_preview"
if weights.with_suffix(".eqx").exists():
    model = load_model(weights)
else:
    from biojemu2.convert import convert_released

    model = convert_released()
    args.out.mkdir(parents=True, exist_ok=True)
    save_model(model, weights)

system = System.from_description(proteins=[args.sequence])
msa = MSA(unpaired_a3m={"A": f">query\n{args.sequence}\n"})  # or bioemu2.msa.alignment.get_msa
features = featurize(system, msa=msa, mode=args.mode, contacts=[])
sampler = (EDMSampler.structure_prediction if args.mode == "structure" else EDMSampler.equilibrium)(
    args.num_steps
)


@eqx.filter_jit
def run(model, features, sampler, keys):
    trunk = model.embed(features, num_recycles=3)
    context = model.static_context(features, trunk)
    return jax.vmap(lambda key: sample(model, context, features.dynamics, sampler, key=key))(keys)


coords = run(
    model, features, sampler, jax.random.split(jax.random.key(args.seed), args.num_samples)
)
save_frames(to_trajectory(system, coords), args.out)
print(f"wrote {args.num_samples} samples to {args.out}")
