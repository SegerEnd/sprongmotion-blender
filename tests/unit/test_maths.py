import numpy as np
from sprongmotion import quat
from sprongmotion import retarget as r


def test_quaternion_basics():
    q = quat.arc(np.array([1.0, 0, 0]), np.array([0, 1.0, 0]))
    assert np.allclose(quat.qrot(q, np.array([1.0, 0, 0])), [0, 1, 0])
    assert np.allclose(quat.qmul(q, quat.qinv(q)), [0, 0, 0, 1])
    flips = np.array([[0, 0, 0, 1.0], [0, 0, 0, -1], [0, 0, 0.1, -0.99]])
    assert (quat.continuous(flips)[:, 3] > 0).all()


def test_resample_and_positions():
    arr = np.arange(5, dtype=float)[:, None]
    assert np.allclose(r.resample(arr, np.array([0.5, 3.25])), [[0.5], [3.25]])
    g = np.tile([0, 0, 0, 1.0], (1, 22, 1))
    p = r.positions(g, np.zeros((1, 3)), np.ones((22, 3)))
    assert np.allclose(p[0, 15] - p[0, 0], [5, 5, 5]), "head = pelvis + 5 offsets up the spine"


def test_planted_ankle_holds_still_and_fades():
    a = np.stack([np.arange(40) * 0.01, np.zeros(40), np.zeros(40)], 1)
    planted = np.zeros(40, bool)
    planted[10:20] = True
    t = r.hold(a, planted, blend=5)
    assert np.ptp(t[10:20], axis=0).max() == 0 and np.allclose(t[:4], a[:4]) and np.allclose(t[26:], a[26:])
    assert np.abs(np.diff(t[:, 0])).max() < 0.03, "no jump at the span edges"


def test_two_bone_reaches_without_overextending():
    h, k, a = np.array([[0, 0, 1.0]]), np.array([[0, -0.05, 0.5]]), np.array([[0, 0, 0.0]])
    front = np.array([0, -1.0, 0])
    for target in ([0.1, 0.05, 0.1], [0, 0, -0.5]):
        thigh, shin = r.two_bone(h, k, a, np.array([target]), front)
        knee = h + quat.qrot(thigh, k - h)
        ankle = knee + quat.qrot(shin, a - k)
        leg = np.linalg.norm(k - h) + np.linalg.norm(a - k)
        if np.linalg.norm(np.array(target) - h) < 0.99 * leg:
            assert np.allclose(ankle, target, atol=1e-6)
        assert np.linalg.norm(ankle - h) <= leg + 1e-9 and (knee - h)[0, 1] < 1e-9, "knee never goes behind"
