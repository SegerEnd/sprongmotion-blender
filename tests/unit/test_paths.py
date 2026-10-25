import numpy as np
from sprongmotion import paths, quat


def test_chain_takes_the_first_spline_and_sees_a_closed_one():
    verts = np.arange(18.0).reshape(6, 3)
    pts, closed = paths.chain(verts, [(0, 1), (1, 2), (3, 4), (4, 5)])
    assert len(pts) == 3 and not closed
    assert paths.chain(verts[:4], [(0, 1), (1, 2), (2, 3), (3, 0)])[1]


def test_resample_spacing_and_closing():
    pts = paths.resample(np.array([[0, 0], [1.0, 0], [1, 1]]), False)
    assert len(pts) == 21 and np.allclose(pts[-1], [1, 1]) and np.allclose(np.diff(pts, axis=0).sum(0), [1, 1])
    square = np.array([[0, 0], [1.0, 0], [1, 1], [0, 1]])
    ring = paths.resample(square, True)
    assert np.allclose(ring[0], ring[-1]) and abs(paths.length(ring) - 4) < 1e-9
    assert len(paths.resample(np.c_[np.linspace(0, 500, 9), np.zeros(9)], False)) == 1024


def test_to_model_inverts_how_the_pelvis_is_placed():
    rng = np.random.default_rng(1)
    axes = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])  # model left, up, forward in a Z-up armature
    ratio, src0, head = 1.3, np.array([0, 0.9, 0]), np.array([0, 0.1, 1.2])
    yaw = quat.arc(np.array([0, 1.0, 0]), np.array([1.0, 1, 0]))
    start = (np.array([0.4, -0.2, 0]), yaw)
    world_from_arm = np.eye(4)
    world_from_arm[:3, :3] = 2 * r_mat(quat.arc(np.array([1.0, 0, 0]), np.array([0, 1.0, 0])))
    world_from_arm[:3, 3] = [3, -1, 0.5]
    root = np.c_[rng.normal(size=5), np.full(5, 0.9), rng.normal(size=5)]
    move = quat.qrot(yaw, ratio * (root - src0) @ axes.T) + start[0]
    world = (np.c_[head + move, np.ones(5)] @ world_from_arm.T)[:, :3]
    xz = paths.to_model(world, np.linalg.inv(world_from_arm), head, start, axes, ratio, src0)
    assert np.allclose(xz, root[:, [0, 2]])


def r_mat(q):
    return np.stack([quat.qrot(q, e) for e in np.eye(3)], 1)


def test_fit_shares_the_path_by_gait_and_keeps_idle_and_fixed_blocks():
    fps, speed = 20, np.r_[np.full(40, 1.0), np.full(20, 0.05), np.full(40, 2.0), np.full(20, 1.0)]
    spans = [(0, 40), (40, 60), (60, 100), (100, 120)]
    travel = [2.0, 0.05, 4.0, 1.0]
    seconds = paths.fit(spans, speed, fps, 10.0, [False, False, False, True])
    assert seconds[1] is None and seconds[3] is None, "idle and locked keep their lengths"
    rest = 10 - travel[1] - travel[3]
    assert np.allclose(seconds[0], rest / 3 / 1.0) and np.allclose(seconds[2], rest * 2 / 3 / 2.0)
    assert paths.fit(spans, speed, fps, 10.0, [True] * 4) == [None] * 4


def test_turn_rate_of_a_circle_and_the_notes():
    fps, r_, v = 20, 2.0, 1.0
    a = np.linspace(0, np.pi, 200)
    arc = np.c_[r_ * np.sin(a), r_ * (1 - np.cos(a))]
    speed = np.full(int(np.pi * r_ / v * fps) + 1, v)
    rates = paths.turn_rates(arc, speed, fps)
    assert np.allclose(np.median(rates), np.degrees(v / r_), rtol=0.03)
    b = np.linspace(0, 2 * np.pi, 400)
    assert paths.notes([(0, len(speed))], speed, fps, np.c_[np.sin(b), 1 - np.cos(b)]) == [
        "Turn r 1.0 m is too tight for this motion"
    ]
    assert paths.notes([(0, len(speed))], speed, fps, arc * 0.5) == ["Path needs 0.5x this gait"]
    assert paths.notes([(0, len(speed))], speed, fps, np.c_[np.zeros(9), np.linspace(0, np.pi * r_, 9)]) == [""]


def test_a_take_that_turns_along_the_path_is_not_warned():
    fps, v = 20, 1.0
    speed = np.full(int(2 * np.pi / v * fps) + 1, v)
    a = np.cumsum(speed) / fps - speed[0] / fps
    circle = np.c_[np.sin(a), 1 - np.cos(a)]
    own = paths.headings(circle, speed, fps)
    assert np.allclose(np.degrees(np.abs(np.gradient(own)))[15:-15] * fps, np.degrees(v), rtol=0.05)
    path = np.c_[np.sin(np.linspace(0, 2 * np.pi, 400)), 1 - np.cos(np.linspace(0, 2 * np.pi, 400))]
    assert paths.notes([(0, len(speed))], speed, fps, path, own) == [""]
    straight = paths.headings(np.c_[np.zeros_like(a), a], speed, fps)
    assert paths.notes([(0, len(speed))], speed, fps, path, straight) == ["Turn r 1.0 m is too tight for this motion"]


def test_a_blocks_part_of_the_path_starts_at_the_origin_along_z():
    a = np.linspace(0, np.pi, 300)
    arc = np.c_[3 + 2 * np.cos(a), 1 + 2 * np.sin(a)]
    root = arc[[100, 150, 200]] + 0.01
    piece = paths.part(arc, root)
    assert np.allclose(piece[0, 1:3], 0) and np.isnan(piece[:, [0, 3]]).all()
    d = piece[1, 1:3] - piece[0, 1:3]
    assert abs(d[0]) < 1e-6 and d[1] > 0
    assert abs(paths.length(piece[:, 1:3]) - 2 * np.pi / 3) < 0.03
    assert paths.part(arc, arc[[100, 100]]) is False
