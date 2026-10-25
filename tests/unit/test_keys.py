import numpy as np
from sprongmotion import keyframes


def walk(n=200):
    t = np.arange(n, dtype=float)
    q = np.stack([np.ones(n), 0.3 * np.sin(t / 9), 0.2 * np.cos(t / 13), 0.05 * np.sin(t / 4)], 1)
    return t, q / np.linalg.norm(q, axis=1, keepdims=True)


def test_fit_passes_its_keys_and_slopes():
    x = np.linspace(0, 10, 1001)
    y, m = np.stack([np.sin(x), x * x], 1), np.stack([np.cos(x), 2 * x], 1)
    k = np.zeros(y.shape, bool)
    k[[0, 400, -1], 0], k[[0, 700, -1], 1] = True, True
    got = keyframes.fit(x, y, m, k)
    assert np.allclose(got[k], y[k])
    assert np.allclose(np.gradient(got, x, axis=0)[[1, 401, 999], 0], m[[1, 401, 999], 0], atol=0.02)
    assert np.allclose(got[:, 1], x * x), "a cubic through a quadratic's keys and slopes is the quadratic"


def test_unbake_keeps_every_bone_within_the_limit():
    x, q = walk()
    loc = np.stack([x * 0.01, np.sin(x / 7) * 0.02, x * 0], 1)
    limit = np.radians(0.5)
    picks, slopes = keyframes.unbake(x, [(q, keyframes.angle, limit), (loc, keyframes.distance, 0.002)], marks=[50])
    mask = np.zeros(q.shape, bool)
    for c, k in enumerate(picks[0]):
        mask[k, c] = True
    fit = keyframes.fit(x, q, slopes[0], mask)
    assert keyframes.angle(fit, q).max() <= limit * 1.0001
    assert all(50 in k for k in picks[0]) and sum(len(k) for k in picks[0]) < 0.5 * 4 * len(x)


def test_shared_frames_and_per_frame_limits():
    x, q = walk(120)
    tight = np.where(np.arange(120) < 60, np.radians(0.1), np.radians(2))
    picks, _ = keyframes.unbake(
        x, [(q, keyframes.angle, tight), (q[:, [0, 2, 1, 3]], keyframes.angle, np.radians(2))], [], True
    )
    k = picks[0][0]
    assert all((p == k).all() for g in picks for p in g)
    assert (k < 60).sum() > (k >= 60).sum(), "the tight half needs more keys"


def test_turns_ignore_jitter():
    y = np.sin(np.arange(100) / 10.0)[:, None] + 0.001 * np.random.default_rng(0).normal(size=(100, 1))
    assert 0 < keyframes.extremes(y, 0.01).sum() <= 4
