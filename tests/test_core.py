"""Pure-JAX tests (no torch, no checkpoint)."""

import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np

from jbioemu2.sampling import augment, edm_sigmas, random_rotation, rigid_align
from jbioemu2.transformer import key_window_index, scatter_mean_atoms_to_tokens, window_keys


def test_import_is_torch_free():
    code = "import sys, jbioemu2; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_random_rotation_is_proper():
    rotations = jax.vmap(random_rotation)(jax.random.split(jax.random.key(0), 16))
    eye = np.broadcast_to(np.eye(3), rotations.shape)
    np.testing.assert_allclose(rotations @ np.swapaxes(rotations, 1, 2), eye, atol=1e-5)
    np.testing.assert_allclose(np.linalg.det(rotations), 1.0, atol=1e-5)


def test_rigid_align_recovers_rigid_motion():
    x = jax.random.normal(jax.random.key(1), (40, 3)) * 10
    moved = x @ random_rotation(jax.random.key(2)) + jnp.array([1.0, -2.0, 3.0])
    mask = jnp.ones(40, bool)
    np.testing.assert_allclose(rigid_align(moved, x, mask), x, atol=1e-3)


def test_augment_centres_masked_atoms():
    x = jax.random.normal(jax.random.key(3), (10, 3)) + 5
    mask = jnp.arange(10) < 7
    out = augment(x, jnp.eye(3), jnp.zeros(3), mask)
    np.testing.assert_allclose(out[:7].mean(0), 0.0, atol=1e-5)


def test_edm_sigmas_shape_and_ends():
    sigmas = edm_sigmas(10, sigma_min=4e-4, sigma_max=160.0, sigma_data=16.0, rho=7.0)
    assert sigmas.shape == (11,) and sigmas[-1] == 0.0
    np.testing.assert_allclose(sigmas[[0, -2]], [2560.0, 0.0064], rtol=1e-5)
    assert np.all(np.diff(sigmas) < 0)


def test_window_keys_matches_pad_unfold():
    n, w, h = 96, 32, 128
    x = np.arange(n, dtype=np.float32)[:, None] + 1
    context = (h - w) // 2
    padded = np.pad(x, ((context, context), (0, 0)))
    expected = np.stack([padded[k * w : k * w + h] for k in range(n // w)])
    np.testing.assert_array_equal(window_keys(jnp.array(x), w, h), expected)
    _, in_range = key_window_index(n, w, h)
    assert in_range.shape == (n // w, h) and not in_range.all()


def test_scatter_mean_excludes_padding():
    values = jnp.array([[1.0], [3.0], [10.0], [100.0]])
    atom_to_token = jnp.array([0, 0, 1, 0])
    valid = jnp.array([True, True, True, False])
    out = scatter_mean_atoms_to_tokens(values, atom_to_token, valid, 2)
    np.testing.assert_allclose(out[:, 0], [2.0, 10.0], rtol=1e-5)
