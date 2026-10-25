"""Return to Start: a block walks back to where its run started, on a way the add-on makes."""

import numpy as np

from . import blocks, paths, timeline_motion

TURN = 2.0  # m: the least radius of the U-turn, at the model's scale like the turn budget
AHEAD = 0.5  # m: the most it goes on straight before it turns (the body's momentum)


def why_not_back(s, b):
    """Why a block's Return to Start does not apply, None when it does."""
    if s.path is not None:
        return "The rig's Path decides where it walks"
    return "In Place: it does not travel" if b.in_place else None


def walks_back(s, b):
    return b.back_to_start and blocks.has_motion(b) and why_not_back(s, b) is None


def way_back(p, h, goal, r):
    """[n, 2] points from p heading h: at most AHEAD straight, a turn of radius r, then straight to goal."""
    forward, to_goal = np.array([np.sin(h), np.cos(h)]), goal - p
    toward = -1.0 if np.linalg.norm(to_goal) > 1e-3 and np.sin(np.arctan2(*to_goal) - h) < 0 else 1.0
    for side in (toward, -toward):
        w = p + r * side * np.array([forward[1], -forward[0]]) - goal
        need = (1.05 * r) ** 2 - w @ w + (w @ forward) ** 2
        straight = 0.0 if w @ w >= (1.05 * r) ** 2 else -w @ forward + np.sqrt(max(need, 0))
        if straight <= AHEAD:
            break
    turn_start = p + min(straight, AHEAD) * forward
    centre = turn_start + r * side * np.array([forward[1], -forward[0]])
    psi = h + side * np.arange(0, 2 * np.pi, 0.002)
    arc = centre - r * side * np.stack([np.cos(psi), -np.sin(psi)], 1)
    ahead = np.angle(np.exp(1j * (np.arctan2(*(goal - arc).T) - psi))) * side
    k = next((i for i in range(1, len(psi)) if -np.pi / 2 < ahead[i] <= 0), len(psi) - 1)
    return paths.resample(np.vstack([p, arc[: k + 1], goal]), False, most=700)


def _in_place_frames(segs, first, n):
    """[n] frames of blocks in place, as timeline_motion.without_travel drops their travel."""
    still = np.zeros(n, bool)
    for (b, _), f0, f1 in zip(segs, first, first[1:], strict=False):
        still[f0:f1] = b.uid >= 0 and b.in_place
    still[0] = False
    return still


def _way_back_points(root, speed, fps, f0, f1, back):
    """The path for a Return to Start block: the run so far, the way back, and on past its end."""
    so_far = root[: f0 + 1, [0, 2]]
    lead = paths.resample(so_far, False, max(paths.SPACING, paths.length(so_far) / 256))[:-1]
    out = back[-1] - back[-2]
    on_to = speed[f1:].sum() / fps + 0.5  # past the end, so the warp does not ease the root to a stop on it
    tail = back[-1] + out / np.linalg.norm(out) * np.linspace(0.5, on_to, 1 + int(on_to)).reshape(-1, 1)
    points = np.vstack([lead, back, tail if f1 < len(root) else back[:0]])
    if np.linalg.norm(points[-1] - points[0]) < 0.01:  # the runtime closes a path whose ends meet
        points[-1] += 0.01 * out / np.linalg.norm(out)
    times = np.full(len(points), np.nan)
    times[len(lead)], times[len(lead) + len(back) - 1] = f0 / fps, (f1 - 1) / fps
    return np.c_[times, points[:, 0], points[:, 1], times * np.nan]


def walk_back(obj, m, segs, first, run_arrays, fits=None):
    """Decode each Return to Start block again on a way back to the run's start (run_arrays changes in place)."""
    s, (rot, joints, root, _, speed) = blocks.settings(obj), run_arrays
    walking = [k for k, (b, _) in enumerate(segs) if b.uid >= 0 and walks_back(s, b)]
    if not walking:
        return
    start = root[first[next(k for k, (b, _) in enumerate(segs) if b.uid >= 0)], [0, 2]].copy()
    still = _in_place_frames(segs, first, len(root))
    for k in walking:
        b, (f0, f1) = segs[k][0], first[k : k + 2]
        travel, gait = paths.gaits([(f0, f1)], speed, m.fps)[0]
        if not gait:
            paths.set_note(b, "Stands still: no way back to walk")
            continue
        goal = start + (np.diff(root[:f0, [0, 2]], axis=0) * still[1:f0, None]).sum(0)
        hips, radius = joints[f0, 1] - joints[f0, 2], max(TURN, gait / np.radians(paths.YAW_BUDGET))
        back = way_back(root[f0, [0, 2]], np.arctan2(-hips[2], hips[0]), goal, radius)
        path_points = _way_back_points(root, speed, m.fps, f0, f1, back)
        with timeline_motion.run_blocks(m, obj, segs, path=path_points) as (motion, _):
            new = timeline_motion.arrays(motion)
        _fix_ends(new, root, goal, f0, f1)
        if f1 < len(root):  # the blocks after it go on from where it ends, turned to its facing
            ends = [np.arctan2(*(j[f1 - 1, 1] - j[f1 - 1, 2])[[0, 2]]) for j in (new[1], joints)]
            _turn_about_pivot(rot[f1:], joints[f1:], root[f1:], ends[0] - ends[1], root[f1 - 1], new[2][f1 - 1])
        for old, made in zip((rot, joints, root), new[:3], strict=True):
            old[f0:f1] = made[f0:f1]
        ratio = paths.length(back) / travel
        paths.set_note(b, f"Return needs {ratio:.1f}x this gait" if abs(ratio - 1) > paths.STRIDE else "")
        if fits is not None:
            fits[b.uid] = paths.length(back) / gait


def _fix_ends(new, root, goal, f0, f1):
    """Shift the remade block so it starts where it was and ends on the goal, blended linearly."""
    e0, e1 = root[f0] - new[2][f0], np.insert(goal, 1, 0) - new[2][f1 - 1]
    blend = np.linspace(0, 1, f1 - f0)
    error = np.outer(1 - blend, e0) + np.outer(blend, e1)
    error[:, 1] = 0
    new[1][f0:f1] += error[:, None]
    new[2][f0:f1] += error


def _turn_about_pivot(rot, joints, root, angle, pivot, to):
    """Frames turned by angle about pivot, then moved by to - pivot (in place)."""
    for p in (joints, root):
        p[..., [0, 2]] -= pivot[[0, 2]]
    timeline_motion.turn_about_up(rot, joints, root, angle)
    for p in (joints, root):
        p[..., [0, 2]] += to[[0, 2]]


def refit(obj, s, m):
    """Give Return to Start blocks the seconds their way back takes, at most twice per edit."""
    if m is None or not any(walks_back(s, b) for b in s.blocks) or paths.blocks_key(s) == s.path_fit:
        return False
    again = s.path_fit != "again" + paths.blocks_key(s)  # a take of the new length has a gait of its own
    fits, changed = {}, False
    timeline_motion.decode(obj, fits)
    for uid, sec in fits.items():
        b = s.blocks[blocks.find(s, uid)]
        if not b.locked:
            changed |= paths.resize(s, b, sec)
    s.path_fit = ("again" if changed and again else "") + paths.blocks_key(s)
    return changed
