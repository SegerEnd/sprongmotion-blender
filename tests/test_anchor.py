"""Continue from Keys, Fill Gap and Regenerate Range: the rig's own keys (a baked take with a part cut out) joined by
takes, measured on the rig as Blender evaluates it."""

import random

import bpy
import numpy as np
from bl_ext.smtest.sprongmotion import blocks, continue_keys, generate, model, retarget, rig, rig_data

FPS, CUT, GAP = 24, 61, (49, 73)
MOVING = [("a person walks forward", 12), ("a person jogs forward", 3), ("a person dances", 5)]


def baked(text, seed, seconds=5.0):
    bpy.context.scene.render.fps, bpy.context.scene.frame_current = FPS, 1
    bpy.ops.sm.add_rig()
    obj = bpy.context.active_object
    obj.sprongmotion.auto = False
    blocks.add(obj, text, seconds).seed = seed
    generate.run_now(obj)
    assert bpy.ops.sm.bake(keys="0.5") == {"FINISHED"}
    return obj


def cut(obj, lo, hi=None, shift=None):
    """The keys after lo go (up to hi); the keys at lo and hi get selected, those from hi on move the root by shift."""
    for fc in retarget.channelbag(obj).fcurves:
        kp = fc.keyframe_points
        for k in [k for k in kp if lo < k.co.x and (hi is None or k.co.x < hi)][::-1]:
            kp.remove(k, fast=True)
        for k in kp:
            k.select_control_point = k.co.x in (lo, hi)
            if shift and fc.data_path.endswith('"pelvis"].location') and k.co.x >= hi:
                for co in (k.co, k.handle_left, k.handle_right):
                    co.y += shift[fc.array_index]
        fc.update()


def own_keys(obj):
    """Every key with its handles, by curve."""
    return {(fc.data_path, fc.array_index): [tuple(np.round([*k.co, *k.handle_left, *k.handle_right], 6))
                                             for k in fc.keyframe_points]
            for fc in retarget.channelbag(obj).fcurves}  # fmt: skip


def outside(keys, lo, hi):
    return {c: [k for k in ks if not lo < k[0] < hi] for c, ks in keys.items()}


def same(a, b):
    """The same keys within float32 handle noise (Blender recomputes aligned handles next to a written key)."""
    return a.keys() == b.keys() and all(len(a[c]) == len(b[c]) and np.allclose(a[c], b[c], atol=1e-5) for c in a)


def world(obj, frames):
    out = []
    for f in frames:
        bpy.context.scene.frame_set(int(f))
        m = np.array(obj.matrix_world)
        out.append([m[:3, :3] @ np.array(obj.pose.bones[n].head) + m[:3, 3] for n in rig.JOINTS])
    return np.array(out)


def jerk(p):
    """Per window of 4 frames: the joints' jerk relative to the pelvis in the hips' heading frame (m/s^3, mean over
    joints), the runtime's seam measure on the rig."""
    hip = p[:, 1] - p[:, 2]
    a = -np.arctan2(hip[:, 1], hip[:, 0])[:, None]
    rel = p - p[:, :1]
    q = np.stack([np.cos(a) * rel[..., 0] - np.sin(a) * rel[..., 1],
                  np.sin(a) * rel[..., 0] + np.cos(a) * rel[..., 1], rel[..., 2]], -1)  # fmt: skip
    return np.linalg.norm(q[3:] - 3 * q[2:-1] + 3 * q[1:-2] - q[:-3], axis=-1)[:, 1:].mean(1) * FPS**3


def slide(p):
    """Per frame step: cm/s an ankle near the floor slides, NaN where none is."""
    out = []
    for a in (7, 8):
        on = p[1:, a, 2] < p[:, a, 2].min() + 0.03
        out.append(np.where(on, np.linalg.norm(np.diff(p[:, a, :2], axis=0), axis=-1) * FPS * 100, np.nan))
    return np.nanmean(np.stack(out), 0)


def at_seams(n, seams, offset):
    """(near a seam, far from any) masks for n values whose first one centres on frame offset."""
    d = np.min([np.abs(np.arange(n) + offset - s) for s in seams], 0)
    return d <= 3.5, d > 12


def gate(results, label):
    """results: [(jerk, slide, seams)] pooled: the median jerk at the seams vs inside (its median, its 75th
    percentile: the motion's own spread) and the mean slide at the seams vs inside."""
    j_at, j_in, s_at, s_in = [], [], [], []
    for j, s, seams in results:
        (a, b), (c, d) = at_seams(len(j), seams, 2.5), at_seams(len(s), seams, 1.5)
        j_at, j_in, s_at, s_in = [*j_at, *j[a]], [*j_in, *j[b]], [*s_at, *s[c]], [*s_in, *s[d]]
    jm, sm = (np.median(j_at), np.median(j_in)), (np.nanmean(s_at), np.nanmean(s_in))
    print(f"> {label}: jerk median {jm[0]:.0f} at the seams vs {jm[1]:.0f} inside ({jm[0] / jm[1] - 1:+.0%}), "
          f"foot slide {sm[0]:.1f} vs {sm[1]:.1f} cm/s ({sm[0] / sm[1] - 1:+.0%})")  # fmt: skip
    return jm[0] / jm[1], jm[0] / np.percentile(j_in, 75), sm[0] / sm[1]


def continued(text, seed, on=True):
    obj = baked(text, seed)
    cut(obj, CUT)
    before = outside(own_keys(obj), CUT, 10**6)
    bpy.context.scene.frame_current = CUT
    b = blocks.add(obj, text, 2.5)
    b.seed, b.continues = seed + 7, on
    generate.run_now(obj)
    assert not on or same(outside(own_keys(obj), CUT, 10**6), before), "the keys up to the block stay as they were"
    assert bool(b.anchor_key.strip("|")) == on
    p = world(obj, range(1, b.end + 1))
    return obj, (jerk(p), slide(p), [CUT])


def test_continue_from_keys():
    on = [continued(t, s)[1] for t, s in MOVING]
    off = [continued(t, s, on=False)[1] for t, s in MOVING]
    _, jerk_on, slide_on = gate(on, "continue from keys")
    jerk_off, *_ = gate(off, "the same blocks, Continue from Keys off")
    assert jerk_on <= 1.0 and slide_on <= 1.2 and jerk_off > 1.5, (jerk_on, slide_on, jerk_off)


def test_edited_keys_make_the_block_dirty():
    obj, _ = continued(*MOVING[0])
    b, m = blocks.settings(obj).blocks[0], model.load()
    continue_keys.check(obj)  # notes what is there
    assert not b.dirty
    fc = retarget.channelbag(obj).fcurves.find('pose.bones["l_elbow"].rotation_quaternion', index=1)
    next(k for k in fc.keyframe_points if k.co.x >= CUT - 10).co.y += 0.2
    continue_keys.check(obj)
    assert b.dirty, "a key before the block changed"
    generate.run_now(obj)
    assert (
        not b.dirty
        and b.anchor_key == continue_keys.block_keys(blocks.settings(obj), continue_keys.plan(obj, m))[b.uid]
    )
    obj.sprongmotion.bones[rig.JOINTS.index("l_knee")].bone = ""
    assert continue_keys.why_not(obj, m) and continue_keys.plan(obj, m) == (None, None), (
        "no legs mapped: off, with a message"
    )


def test_fill_gap():
    obj = baked(*MOVING[0])
    name = obj.name
    p0 = world(obj, range(1, 121))
    cut(obj, *GAP)
    before = outside(own_keys(obj), *GAP)
    bpy.context.scene.frame_current = 60
    random.seed(1)  # the block's seed
    bpy.ops.ed.undo_push(message="before")
    assert bpy.ops.sm.fill_gap(text="a person walks forward") == {"FINISHED"}
    bpy.ops.ed.undo_push(message="Fill Gap")
    obj = bpy.data.objects[name]
    s = blocks.settings(obj)
    b = s.blocks[0]
    assert len(s.blocks) == 1 and (b.start, b.end) == GAP and all(b.anchor_key.split("|")), b.anchor_key
    assert same(outside(own_keys(obj), *GAP), before), "the keys outside the gap stay as they were"
    print(f"> fill gap far seam: {b.seam_note or 'within what the gap absorbs'}")
    p = world(obj, range(1, 121))
    gate([(jerk(p), slide(p), list(GAP))], "fill gap")
    for f, d in ((GAP[0], 1), (GAP[1], -1)):  # index f - 1 is frame f: the take's first frame past the seam
        bend, own = (np.linalg.norm(q[f - 1 + d] - 2 * q[f - 1] + q[f - 1 - d], axis=-1) for q in (p, p0))
        print(
            f"> fill gap at {f}: joints off the keys' own step by {100 * bend.mean():.1f} cm mean, "
            f"{100 * bend.max():.1f} max (the take without the gap: {100 * own.mean():.1f}, {100 * own.max():.1f})"
        )
        assert bend.mean() - own.mean() < 0.02
    bpy.ops.ed.undo()
    obj = bpy.data.objects[name]
    assert not len(blocks.settings(obj).blocks) and same(outside(own_keys(obj), *GAP), before), "one Cmd+Z"
    bpy.ops.ed.redo()
    obj = bpy.data.objects[name]
    assert bpy.ops.sm.remove_data(bake=True) == {"FINISHED"}
    assert "sprongmotion" not in (obj.bl_system_properties_get() or {})
    assert {k.type for fc in retarget.channelbag(obj).fcurves for k in fc.keyframe_points} == {"KEYFRAME"}
    assert continue_keys._on_depsgraph not in bpy.app.handlers.depsgraph_update_post


def test_far_keys_warn():
    obj = baked(*MOVING[0])
    cut(obj, *GAP, shift=(1.0, 0, 0))
    bpy.context.scene.frame_current = 60
    assert bpy.ops.sm.fill_gap(text="a person walks forward") == {"FINISHED"}
    note = blocks.settings(obj).blocks[0].seam_note
    print(f"> keys after moved 1 m: {note}")
    assert note.startswith("The keys after are 0.9") or note.startswith("The keys after are 1.0")


def test_a_block_added_in_a_gap_fills_it_and_frozen_bones_keep_their_keys():
    obj = baked(*MOVING[0])
    rig_data.freeze(obj, ["l_elbow"])
    cut(obj, *GAP)
    elbow = own_keys(obj)[('pose.bones["l_elbow"].rotation_quaternion', 0)]
    bpy.context.scene.frame_current = 60
    b = blocks.add(obj, "a person walks forward", 3)
    generate.run_now(obj)
    assert (b.start, b.end) == GAP and all(b.anchor_key.split("|"))
    assert own_keys(obj)[('pose.bones["l_elbow"].rotation_quaternion', 0)] == elbow


def test_regenerate_range():
    bpy.context.scene.render.fps, bpy.context.scene.frame_current = FPS, 1
    bpy.ops.sm.add_rig()
    obj = bpy.context.active_object
    obj.sprongmotion.auto = False
    for text in ("a person walks forward", "a person waves"):
        blocks.add(obj, text, 3).seed = 4
    generate.run_now(obj)
    m, b = model.load(), blocks.settings(obj).blocks[0]
    old = blocks.codes(b, m.id).copy()
    sc = bpy.context.scene
    sc.use_preview_range, sc.frame_preview_start, sc.frame_preview_end = True, b.start + 24, b.start + 48
    name = obj.name
    bpy.ops.ed.undo_push(message="before")
    assert bpy.ops.sm.block_regenerate_range(index=0) == {"FINISHED"}
    bpy.ops.ed.undo_push(message="range")
    b = blocks.settings(bpy.data.objects[name]).blocks[0]
    new, k0, k1 = blocks.codes(b, m.id), 10, 20  # 1 s to 2 s of a 3 s block: tokens 10..20
    assert (new[:k0] == old[:k0]).all() and (new[k1:] == old[k1:]).all(), "the codes outside stay bit-exact"
    assert not (new[k0:k1] == old[k0:k1]).all() and not b.dirty
    bpy.ops.ed.undo()
    assert (blocks.codes(blocks.settings(bpy.data.objects[name]).blocks[0], m.id) == old).all(), "one Cmd+Z"
