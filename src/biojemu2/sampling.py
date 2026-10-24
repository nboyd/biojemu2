"""EDM sampler (stochastic-churn Euler, Karras et al. 2022 Algorithm 2 / AF3 Algorithm 18).

Each step: centre (masked) + random rigid augmentation, churn noise, denoise, rigid-align
the noisy sample onto the denoised one (detached, as in the reference), Euler step.
All randomness lives in :class:`Noise`, so :func:`sample_from_noise` is deterministic.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from biojemu2.features import Dynamics
from biojemu2.model import BioEmu2, TrunkEmbeddings
from biojemu2.structure import StaticContext


def edm_sigmas(
    num_steps: int, *, sigma_min: float, sigma_max: float, sigma_data: float, rho: float
) -> np.ndarray:
    """Karras schedule, ``num_steps`` levels from ``sigma_max`` to ``sigma_min`` (scaled by
    ``sigma_data``, in angstrom) followed by 0."""
    if num_steps < 2:
        raise ValueError(f"num_steps must be at least 2, got {num_steps}")
    t = np.arange(num_steps, dtype=np.float32) / (num_steps - 1)
    inv = 1 / rho
    sigmas = (sigma_max**inv + t * (sigma_min**inv - sigma_max**inv)) ** rho * sigma_data
    return np.append(sigmas, 0.0).astype(np.float32)


class EDMSampler(eqx.Module):
    """Sampler settings. ``step_scale`` 1.5 sharpens towards a single predicted structure;
    1.0 samples the equilibrium ensemble (the reference's two presets)."""

    num_steps: int = eqx.field(static=True)
    step_scale: float
    noise_scale: float = 1.0
    gamma_0: float = 0.8
    gamma_min: float = 1.0
    rho: float = eqx.field(static=True, default=7.0)
    align: bool = eqx.field(static=True, default=True)

    @classmethod
    def structure_prediction(cls, num_steps: int) -> EDMSampler:
        return cls(num_steps, step_scale=1.5)

    @classmethod
    def equilibrium(cls, num_steps: int) -> EDMSampler:
        return cls(num_steps, step_scale=1.0)


class Noise(eqx.Module):
    """Every random draw of one sampling run: standard-normal initial and churn noise,
    rotations (applied as ``x @ R``) and standard-normal translations."""

    init: Float[Array, "N 3"]
    churn: Float[Array, "K N 3"]
    rotations: Float[Array, "K 3 3"]
    translations: Float[Array, "K 3"]


def random_rotation(key: PRNGKeyArray) -> Float[Array, "3 3"]:
    """Haar-random rotation from a Gaussian quaternion (reference convention)."""
    q = jax.random.normal(key, (4,))
    q = q / jnp.where(q[0] < 0, -1.0, 1.0) / jnp.linalg.norm(q)
    r, i, j, k = q
    two_s = 2.0 / jnp.sum(q * q)
    return jnp.array(
        [
            [1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r)],
            [two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r)],
            [two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j)],
        ]
    )


def draw_noise(key: PRNGKeyArray, *, num_steps: int, num_atoms: int) -> Noise:
    k_init, k_churn, k_rot, k_trans = jax.random.split(key, 4)
    return Noise(
        init=jax.random.normal(k_init, (num_atoms, 3)),
        churn=jax.random.normal(k_churn, (num_steps, num_atoms, 3)),
        rotations=jax.vmap(random_rotation)(jax.random.split(k_rot, num_steps)),
        translations=jax.random.normal(k_trans, (num_steps, 3)),
    )


def augment(
    x: Float[Array, "N 3"],
    rotation: Float[Array, "3 3"],
    translation: Float[Array, " 3"],
    mask: Bool[Array, " N"],
) -> Float[Array, "N 3"]:
    """Centre on the masked mean, then ``x @ R + t``."""
    w = mask[:, None].astype(x.dtype)
    centred = x - (x * w).sum(0) / w.sum(0)
    return centred @ rotation + translation


def rigid_align(
    source: Float[Array, "N 3"], dest: Float[Array, "N 3"], mask: Bool[Array, " N"]
) -> Float[Array, "N 3"]:
    """Weighted Kabsch: move ``source`` onto ``dest`` (proper rotation, masked atoms)."""
    w = mask[:, None].astype(source.dtype)
    src_mean = (source * w).sum(0) / w.sum(0)
    dst_mean = (dest * w).sum(0) / w.sum(0)
    src_c = source - src_mean
    cov = jnp.einsum("ni,nj->ij", w * (dest - dst_mean), src_c, precision="highest")
    u, _, vh = jnp.linalg.svd(cov)
    rotation = u @ vh
    rotation = rotation + (jnp.sign(jnp.linalg.det(rotation)) - 1) * jnp.outer(u[:, 2], vh[2])
    return jnp.einsum("nj,ij->ni", src_c, rotation, precision="highest") + dst_mean


@eqx.filter_jit
def sample_from_noise(
    model: BioEmu2,
    context: StaticContext,
    trunk: TrunkEmbeddings,
    dynamics: Dynamics,
    atom_mask: Bool[Array, " N"],
    sampler: EDMSampler,
    noise: Noise,
) -> Float[Array, "N 3"]:
    """Run the sampler with the given draws. ``atom_mask`` is ``Features.atom_mask``."""
    sigmas = edm_sigmas(
        sampler.num_steps,
        sigma_min=model.sigma_min,
        sigma_max=model.sigma_max,
        sigma_data=model.sigma_data,
        rho=sampler.rho,
    )
    gammas = jnp.where(sigmas[1:] > sampler.gamma_min, sampler.gamma_0, 0.0)

    @jax.checkpoint
    def step(carry, inputs):
        x, coords_cond = carry
        sigma, sigma_next, gamma, eps, rotation, translation = inputs
        x = augment(x, rotation, translation, atom_mask)
        # The companion frame moves with the sample (it only matters when conditioned).
        coords_cond = augment(coords_cond, rotation, translation, atom_mask)
        t_hat = sigma * (1 + gamma)
        # Clamp: with gamma = 0, t_hat == sigma and a fused multiply-add can leave a tiny
        # negative residual (the reference computes this in float64 on the host).
        x = x + sampler.noise_scale * jnp.sqrt(jnp.maximum(t_hat**2 - sigma**2, 0.0)) * eps
        x0 = model.denoise(
            context, trunk, x, t_hat, Dynamics(coords_cond, dynamics.dt, dynamics.is_conditioned)
        )
        if sampler.align:
            x = jax.lax.stop_gradient(rigid_align(x, x0, atom_mask))
        x = x + sampler.step_scale * (sigma_next - t_hat) * (x - x0) / t_hat
        return (x, coords_cond), None

    xs = (sigmas[:-1], sigmas[1:], gammas, noise.churn, noise.rotations, noise.translations)
    (x, _), _ = jax.lax.scan(step, (sigmas[0] * noise.init, dynamics.coords_cond), xs)
    return x


def sample(
    model: BioEmu2,
    context: StaticContext,
    trunk: TrunkEmbeddings,
    dynamics: Dynamics,
    atom_mask: Bool[Array, " N"],
    sampler: EDMSampler,
    *,
    key: PRNGKeyArray,
) -> Float[Array, "N 3"]:
    """Draw one structure. ``jax.vmap`` over keys for several (the trunk and static context
    are shared, like the reference's ``multiplicity``)."""
    noise = draw_noise(key, num_steps=sampler.num_steps, num_atoms=atom_mask.shape[0])
    return sample_from_noise(model, context, trunk, dynamics, atom_mask, sampler, noise)
