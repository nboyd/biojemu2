"""Per-module parity vs randomly initialised torch reference modules on random inputs.

Needs the ``reference`` extra (torch + bioemu2); no checkpoint. Run with ``pytest -m reference``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.reference

import torch  # noqa: E402  (fails loudly without the `reference` extra)
from bioemu2.biomoa.modules import (  # noqa: E402
    conditioning,
    diffusion,
    embedders,
    encoders,
    pairmixer,
    transformer,  # noqa: E402
)
from bioemu2.biomoa.modules import relative_position_encoder as relpos  # noqa: E402
from bioemu2.biomoa.modules import time as time_embed  # noqa: E402
from bioemu2.biomoa.ops import attention_pair_bias as apb  # noqa: E402
from bioemu2.biomoa.ops import (  # noqa: E402
    outer_product_mean,
    pair_weighted_averaging,
    transition,
    triangle_multiplication,
)

from jbioemu2.atom import AtomInputs, TokenFeatures  # noqa: E402
from jbioemu2.convert import from_torch  # noqa: E402
from jbioemu2.structure import StructureInputs  # noqa: E402

jax.config.update("jax_default_matmul_precision", "highest")
torch.set_float32_matmul_precision("highest")
W, H = 32, 128


def randomize(module, seed=0):
    """Re-draw every parameter (some are zero-initialised in the reference)."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) / max(p.shape[-1], 1) ** 0.5)
    return module.eval()


def t(x):
    return torch.from_numpy(np.asarray(x))


def check(ours, ref, tol=1e-5):
    ours, ref = np.asarray(ours), ref.detach().numpy()
    assert ours.shape == ref.shape, (ours.shape, ref.shape)
    err = np.abs(ours - ref).max() / max(np.abs(ref).max(), 1e-12)
    print(f"max rel err {err:.1e}", end="  ")
    assert err < tol, f"max rel err {err:.2e}"


class System:
    """Random tokens/atoms: variable atoms per token, trailing atom and token padding."""

    def __init__(self, seed=0, tokens=12, real_tokens=10):
        rng = np.random.default_rng(seed)
        per_token = rng.integers(1, 9, real_tokens)
        n_real = int(per_token.sum())
        n = -(-n_real // W) * W + W  # at least one extra window of padding
        self.T, self.N = tokens, n
        self.atom_valid = np.arange(n) < n_real
        # Padding atoms point at token 0; they are excluded through ``atom_valid``.
        self.atom_to_token = np.zeros(n, np.int64)
        self.atom_to_token[:n_real] = np.repeat(np.arange(real_tokens), per_token)
        self.token_valid = np.arange(tokens) < real_tokens
        center = np.zeros(tokens, np.int64)
        center[:real_tokens] = np.cumsum(per_token) - per_token
        self.center = center
        self.ref_pos = rng.normal(size=(n, 3)).astype(np.float32) * 3
        self.charge = rng.integers(-1, 2, n).astype(np.float32)
        self.element = rng.integers(0, 128, n)
        self.name = rng.integers(0, 64, (n, 4))
        self.uid = np.where(self.atom_valid, self.atom_to_token, -1)
        self.rng = rng

    def atoms(self):
        return AtomInputs(
            ref_pos=jnp.array(self.ref_pos),
            charge=jnp.array(self.charge),
            element=jnp.array(self.element),
            name_encoding=jnp.array(self.name),
            ref_space_uid=jnp.array(self.uid),
            atom_valid=jnp.array(self.atom_valid),
            atom_to_token=jnp.array(self.atom_to_token),
            token_valid=jnp.array(self.token_valid),
        )

    def normal(self, *shape):
        return self.rng.normal(size=shape).astype(np.float32)


@pytest.mark.parametrize("mode", ["outgoing", "incoming"])
def test_triangle_multiplication(mode):
    s = System()
    ref = randomize(triangle_multiplication.TriangleMultiplicationFull(dim=16, mode=mode))
    z, mask = s.normal(s.T, s.T, 16), np.outer(s.token_valid, s.token_valid)
    check(from_torch(ref)(jnp.array(z), jnp.array(mask)), ref(t(z)[None], t(mask)[None])[0])


def test_outer_product_mean():
    s = System()
    ref = randomize(outer_product_mean.OuterProductMeanVanilla(c_in=8, c_hidden=4, c_out=16))
    m, mask = s.normal(3, s.T, 8), np.ones((3, s.T), bool)
    mask[2] = False
    check(from_torch(ref)(jnp.array(m), jnp.array(mask)), ref(t(m)[None], t(mask)[None])[0])


def test_pair_weighted_averaging():
    s = System()
    ref = randomize(
        pair_weighted_averaging.PairWeightedAveraging(c_m=8, c_z=16, c_h=4, num_heads=2)
    )
    m, z, mask = s.normal(3, s.T, 8), s.normal(s.T, s.T, 16), np.outer(s.token_valid, s.token_valid)
    check(
        from_torch(ref)(jnp.array(m), jnp.array(z), jnp.array(mask)),
        ref(t(m)[None], t(z)[None], t(mask)[None])[0],
    )


@pytest.mark.parametrize("use_norm", [True, False])
def test_transition(use_norm):
    s = System()
    ref = randomize(transition.Transition(dim=8, hidden=16, out_dim=6, use_norm=use_norm))
    x = s.normal(5, 8)
    check(from_torch(ref)(jnp.array(x)), ref(t(x)))


def test_conditioning_and_relpos():
    s = System()
    cc = randomize(conditioning.ContactConditioning(token_z=16))
    contact = np.eye(5, dtype=np.float32)[s.rng.integers(0, 5, (s.T, s.T))]
    thr = s.rng.uniform(4, 20, (s.T, s.T)).astype(np.float32)
    mask = np.outer(s.token_valid, s.token_valid)
    check(
        from_torch(cc)(jnp.array(contact), jnp.array(thr), jnp.array(mask)),
        cc(t(contact)[None], t(thr)[None], t(s.token_valid)[None])[0],
    )
    mc = randomize(conditioning.MethodConditioning(token_s=8, token_z=16, embed_into_pairs=True))
    method = s.rng.integers(0, 2, s.T)
    ours = from_torch(mc)(jnp.array(method), jnp.array(s.token_valid))
    ref = mc(t(method)[None], t(s.token_valid)[None])
    check(ours[0], ref[0][0])
    check(ours[1], ref[1][0])
    rp = randomize(relpos.FusedRelativePositionEncoder(token_z=16))
    ids = [s.rng.integers(0, 3, s.T) for _ in range(2)] + [
        np.arange(s.T) * 2,
        s.rng.integers(0, 2, s.T),
    ]
    check(from_torch(rp)(*map(jnp.array, ids)), rp(*[t(i)[None] for i in ids])[0])


def test_pairmixer_layers():
    s = System()
    pm = randomize(
        pairmixer.PairMixerLayer(token_s=8, token_z=16, dropout=0.0, trimul_variant="full")
    )
    z, mask = s.normal(s.T, s.T, 16), np.outer(s.token_valid, s.token_valid)
    check(from_torch(pm)(jnp.array(z), jnp.array(mask)), pm(None, t(z)[None], t(mask)[None])[1][0])
    msa_layer = randomize(pairmixer.PairMixerMSALayer(8, 8, 16, 0.0, 0.0, trimul_variant="full"))
    m, msa_mask = s.normal(3, s.T, 8), np.ones((3, s.T), bool)
    z_ours, m_ours = from_torch(msa_layer)(
        jnp.array(z), jnp.array(m), jnp.array(mask), jnp.array(msa_mask)
    )
    _, z_ref, m_ref = msa_layer(None, t(z)[None], t(m)[None], t(mask)[None], t(msa_mask)[None])
    check(z_ours, z_ref[0])
    check(m_ours, m_ref[0])


def test_attention_pair_bias():
    s = System()
    full = randomize(apb.AttentionPairBias(c_s=16, c_z=8, num_heads=4, initial_norm=False))
    x, z = s.normal(s.T, 16), s.normal(s.T, s.T, 8)
    check(
        from_torch(full)(jnp.array(x), jnp.array(z), jnp.array(s.token_valid)),
        full(t(x)[None], t(z)[None], t(s.token_valid)[None])[0],
    )
    window = randomize(
        apb.AttentionPairBiasSlidingWindow(c_s=16, c_z=8, num_heads=4, initial_norm=False)
    )
    q, p = s.normal(s.N, 16), s.normal(s.N // W, W, H, 8)
    check(
        from_torch(window)(jnp.array(q), jnp.array(p), jnp.array(s.atom_valid)),
        window(t(q).reshape(-1, W, 16), t(p)[None], t(s.atom_valid).reshape(-1, W)).reshape(
            s.N, 16
        ),
    )


@pytest.mark.parametrize("windowed", [False, True])
def test_conditioned_transformer_layer(windowed):
    s = System()
    kwargs = dict(dim=16, dim_pairwise=8, dim_cond=12, heads=4, transition_hidden=32, lns_scale=0.7)
    if windowed:
        ref = randomize(transformer.ConditionedSlidingWindowTransformerLayer(W, H, **kwargs))
        x, pairs, c, mask = (
            s.normal(s.N, 16),
            s.normal(s.N // W, W, H, 8),
            s.normal(s.N, 12),
            s.atom_valid,
        )
    else:
        ref = randomize(transformer.ConditionedTransformerLayer(**kwargs))
        x, pairs, c, mask = (
            s.normal(s.T, 16),
            s.normal(s.T, s.T, 8),
            s.normal(s.T, 12),
            s.token_valid,
        )
    check(
        from_torch(ref)(jnp.array(x), jnp.array(pairs), jnp.array(c), jnp.array(mask)),
        ref(t(x)[None], t(pairs)[None], t(c)[None], mask=t(mask)[None])[0],
    )


def _atom_kwargs(s):
    return dict(
        atom_ref_pos=t(s.ref_pos)[None],
        ref_charge=t(s.charge)[None],
        ref_element=t(s.element)[None],
        ref_atom_name_chars=t(s.name)[None],
        atom_ref_space_uid=t(s.uid)[None],
        atom_pad_mask=t(s.atom_valid)[None],
        atom_to_token_ids=t(s.atom_to_token)[None],
        token_pad_mask=t(s.token_valid)[None],
    )


def test_input_embedder():
    s = System()
    ref = randomize(
        embedders.InputEmbedder(
            atom_s=16,
            atom_z=8,
            token_s=12,
            token_z=10,
            atoms_per_window_queries=W,
            atoms_per_window_keys=H,
            atom_encoder_depth=2,
            atom_encoder_heads=4,
        )
    )
    tokens = TokenFeatures(
        res_type=jnp.array(s.rng.integers(0, 33, s.T)),
        profile=jnp.array(s.normal(s.T, 33)),
        deletion_mean=jnp.array(s.normal(s.T)),
        pocket_feature=jnp.array(s.rng.integers(0, 4, s.T)),
    )
    kw = _atom_kwargs(s)
    expected = ref(
        res_type=t(np.asarray(tokens.res_type))[None].long(),
        profile=t(np.asarray(tokens.profile))[None],
        deletion_mean=t(np.asarray(tokens.deletion_mean))[None],
        pocket_feature=t(np.asarray(tokens.pocket_feature))[None].long(),
        ref_pos=kw["atom_ref_pos"],
        ref_charge=kw["ref_charge"],
        ref_element=kw["ref_element"],
        ref_atom_name_chars=kw["ref_atom_name_chars"],
        ref_space_uid=kw["atom_ref_space_uid"],
        atom_pad_mask=kw["atom_pad_mask"],
        atom_to_token_ids=kw["atom_to_token_ids"],
        token_pad_mask=kw["token_pad_mask"],
    )[0]
    check(from_torch(ref)(s.atoms(), tokens), expected)


def test_structure_atom_encoder_with_dynamics():
    s = System()
    ref = randomize(
        encoders.AtomAttentionEncoder(
            atom_s=16,
            atom_z=8,
            token_s=12,
            token_z=10,
            atoms_per_window_queries=W,
            atoms_per_window_keys=H,
            structure_prediction=True,
            use_atom_encoder=False,
            use_atom_to_token_aggregation=False,
            use_dynamics_conditioning=True,
            downscale_condition_contrib=True,
            dt_embed_num_timescales=8,
        )
    )
    s_trunk, z_trunk = s.normal(s.T, 12), s.normal(s.T, s.T, 10)
    r, cond = s.normal(s.N, 3), s.normal(s.N, 3)
    kw = _atom_kwargs(s)
    ours = from_torch(ref)
    ctx = ours.static_context(s.atoms(), jnp.array(s_trunk), jnp.array(z_trunk))
    for conditioned in (True, False):
        _, q_ref, c_ref, p_ref = ref(
            **kw,
            s_trunk=t(s_trunk)[None],
            z=t(z_trunk)[None],
            r=t(r)[None],
            coords_cond=t(cond)[None],
            is_dynamics_conditioned=torch.tensor([[conditioned]]),
            dt=torch.tensor([[3.0]]),
        )
        q = ours.coordinate_queries(
            ctx, jnp.array(r), jnp.array(cond), jnp.array(conditioned), jnp.float32(3.0)
        )
        check(q, q_ref[0])
        check(ctx.c, c_ref[0])
        check(ctx.p, p_ref[0])


def test_single_and_pairwise_conditioning():
    s = System()
    single = randomize(
        diffusion.SingleConditioning(
            sigma_data=16.0,
            token_s=12,
            s_input_dim=10,
            a_s=16,
            dim_fourier=8,
            use_dt_conditioning=True,
            dt_embed_num_timescales=4,
        )
    )
    s_trunk, s_inputs = s.normal(s.T, 12), s.normal(s.T, 10)
    for conditioned in (True, False):
        expected, _ = single(
            times=torch.tensor([-0.3]),
            s_trunk=t(s_trunk)[None],
            s_inputs=t(s_inputs)[None],
            dt=torch.tensor([[5.0]]),
            is_dynamics_conditioned=torch.tensor([[conditioned]]),
        )
        ours = from_torch(single)(
            jnp.float32(-0.3),
            jnp.array(s_trunk),
            jnp.array(s_inputs),
            jnp.float32(5.0),
            jnp.array(conditioned),
        )
        check(ours, expected[0])
    pair = randomize(diffusion.PairwiseConditioning(token_z=10, dim_token_rel_pos_feats=10, a_z=12))
    z, rp = s.normal(s.T, s.T, 10), s.normal(s.T, s.T, 10)
    check(from_torch(pair)(jnp.array(z), jnp.array(rp)), pair(t(z)[None], t(rp)[None])[0])
    damped = time_embed.DampedHarmonicTimeEmbedding(num_timescales=6)
    check(from_torch(damped)(jnp.float32(42.0)), damped(torch.tensor([[42.0]]))[0])


def test_dual_stream_layer_and_readout():
    s = System()
    a_s, atom_s = 16, 12
    layer = randomize(
        diffusion.DualStreamDiffusionLayer(
            a_s=a_s,
            a_z=8,
            atom_s=atom_s,
            atom_z=6,
            atom_attention_heads=4,
            token_attention_heads=4,
            layer_index=2,
            use_lns=True,
        )
    )
    readout = randomize(diffusion.ReadoutSuperposition(atom_s=atom_s, token_s=a_s, hidden_dim=10))
    a, z, q = s.normal(s.T, a_s), s.normal(s.T, s.T, 8), s.normal(s.N, atom_s)
    p, c = s.normal(s.N // W, W, H, 6), s.normal(s.T, a_s)
    atom_to_token = s.atom_to_token
    f = StructureInputs(
        atom_valid=jnp.array(s.atom_valid),
        atom_mask=jnp.array(s.atom_valid),
        token_valid=jnp.array(s.token_valid),
        atom_to_token=jnp.array(atom_to_token),
        token_center_atom=jnp.array(s.center),
    )
    a_ours, q_ours = from_torch(layer)(
        jnp.array(a), jnp.array(q), jnp.array(z), jnp.array(p), jnp.array(c), f
    )
    a_ref, q_ref = layer(
        t(a)[None],
        t(z)[None],
        t(q)[None],
        t(p)[None],
        t(c)[None],
        t(s.atom_valid)[None],
        t(s.token_valid)[None],
        t(atom_to_token)[None],
        t(~s.atom_valid)[None],
        t(s.center)[None],
    )
    check(a_ours, a_ref[0])
    check(q_ours, q_ref[0])
    check(
        from_torch(readout)(jnp.array(a), jnp.array(q), f),
        readout(t(a)[None], t(q)[None], t(atom_to_token)[None], t(~s.atom_valid)[None])[0],
    )
