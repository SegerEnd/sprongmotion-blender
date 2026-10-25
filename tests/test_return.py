"""Return to Start: a block walks back to where its run started, on a way the add-on makes."""

import bpy
import numpy as np
from bl_ext.smtest.sprongmotion import blocks, engine, model, retarget, return_to_start, timeline
from test_paths import WALK, lengths, pelvis_xy, rig, s_curve
from undos import NAME, one_undo


def keys_before(obj, frame):
    """Every generated key before frame, rounded: what the earlier block shows."""
    return {(fc.data_path, fc.array_index): np.array([k.co for k in fc.keyframe_points if k.co[0] < frame]).round(5)
            for fc in obj.animation_data.action.layers[0].strips[0].channelbags[0].fcurves}  # fmt: skip


def radius(xy, stride=1.4):
    """The tightest turn of a path: its heading change per metre, averaged over a stride (the sway's wavelength)."""
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    p = np.stack([np.interp(np.arange(0, s[-1], 0.1), s, xy[:, i]) for i in range(2)], 1)
    k = np.ones(round(stride / 0.1)) / round(stride / 0.1)
    d = np.diff(np.stack([np.convolve(p[:, i], k, "valid") for i in range(2)], 1)[::5], axis=0)
    h = np.unwrap(np.arctan2(d[:, 0], d[:, 1]))
    return 1 / np.abs(h[2:] - h[:-2]).max()


def test_a_walk_returns_to_the_start_and_the_block_before_stays():
    obj = rig((WALK, 3), ("a person walks", 3))
    s, m = blocks.settings(obj), model.load()
    s.blocks[1].seed = 4  # a take that keeps walking when it is longer
    a = s.blocks[0]
    codes = blocks.codes(a, m.id).tolist()
    way, made = return_to_start.way_back, []
    return_to_start.way_back = lambda *a: made.append(way(*a)) or made[-1]
    try:
        one_undo(lambda o: setattr(blocks.settings(o).blocks[1], "back_to_start", True), "Return to Start")
    finally:
        return_to_start.way_back = way
    obj = bpy.data.objects[NAME]
    s = blocks.settings(obj)
    a, b = s.blocks
    xy = pelvis_xy(obj, range(a.start, b.end + 1))
    end, ratio = np.linalg.norm(xy[-1] - xy[0]), retarget.Target(obj, m.offsets).ratio
    turn, walked = radius(made[-1], 0.1), radius(xy[b.start - a.start :] / ratio)
    print(f"> end {end * 100:.1f} cm from the start, turn {turn:.2f} m (pelvis {walked:.2f} m), note {b.path_note!r}")
    assert end < 0.05, f"ends {end:.3f} m from where the run started"
    steps, at = np.linalg.norm(np.diff(xy, axis=0), axis=1), b.start - a.start - 1
    assert steps[at] < max(steps[at - 1], steps[at + 1]) + 0.01, f"a jump at the seam: {steps[at - 2 : at + 3]}"
    # the pelvis sways about the path, more so on a stretched walk
    assert turn > 1.99 and walked > 1.8, f"turn radius {turn:.2f} m, walked {walked:.2f} m"
    assert blocks.codes(a, m.id).tolist() == codes, "the block before keeps its codes"
    on = keys_before(obj, b.start - 1)  # the frame before b samples between the two blocks' model frames
    b.back_to_start = False  # the same take without the way back
    assert lengths(obj)[1] == b.end - b.start
    off = keys_before(obj, b.start - 1)
    worst = max(np.abs(on[k] - v).max() for k, v in off.items())
    pelvis = [k for k in off if '"pelvis"' in k[0]]
    print(f"> the block before: pelvis keys equal, legs within {worst:.4f} (IK holds a foot planted across the seam)")
    assert all(np.array_equal(on[k], off[k]) for k in pelvis) and worst < 0.02, "the block before keeps its keys"
    b.back_to_start = True

    b.locked, was = True, b.end - b.start  # the padlock keeps a length the way back does not fit: amber
    blocks.resize(obj, 1, end=b.start + was // 2)
    blocks.mark_dirty(obj, b.uid)
    assert lengths(obj)[1] == was // 2 and "Return needs" in b.path_note, (lengths(obj), b.path_note)
    xy = pelvis_xy(obj, range(a.start, b.end + 1))
    assert np.linalg.norm(xy[-1] - xy[0]) < 0.05, "still ends on the start"


def test_an_idle_block_gets_a_note_and_a_user_path_wins():
    obj = rig((WALK, 2), ("a person stands still", 2))
    s = blocks.settings(obj)
    was = lengths(obj)
    s.blocks[1].back_to_start = True
    assert lengths(obj) == was and s.blocks[1].path_note.startswith("Stands still"), s.blocks[1].path_note
    s.blocks[1].back_to_start = False
    assert not s.blocks[1].path_note
    s.blocks[1].back_to_start = True
    s.path = s_curve()
    assert return_to_start.why_not_back(s, s.blocks[1]) and not return_to_start.walks_back(s, s.blocks[1])


def test_copy_paste_and_split_carry_it_and_clear_motion_removes_it():
    obj = rig((WALK, 2), ("a person walks", 3))
    s = blocks.settings(obj)
    s.blocks[1]["back_to_start"] = True
    pasted = blocks.paste(obj, blocks.to_clip(s.blocks[1]), 1)
    assert pasted.back_to_start
    blocks.split(obj, 1, (s.blocks[1].start + s.blocks[1].end) // 2)
    assert [b.back_to_start for b in s.blocks] == [False, False, True, True], "the second half walks back"
    assert bpy.ops.sm.remove_data(bake=False) == {"FINISHED"}
    assert blocks.settings(obj) is None or not len(blocks.settings(obj).blocks)


def test_after_a_block_in_place_it_returns_to_where_the_rig_stands():
    obj = rig(("a person waves", 2), (WALK, 3), ("a person walks", 3))
    s = blocks.settings(obj)
    s.blocks[0].in_place = True
    s.blocks[2].seed, s.blocks[2].back_to_start = 2, True
    xy = pelvis_xy(obj, (s.blocks[0].start, s.blocks[2].end))
    print(f"> after a block in place: ends {np.linalg.norm(xy[1] - xy[0]) * 100:.1f} cm from the start")
    assert np.linalg.norm(xy[1] - xy[0]) < 0.05


def test_a_block_at_the_limit_before_a_block_that_needs_a_take_is_made():
    obj = rig((WALK, 2), ("a person waves", 2))
    s = blocks.settings(obj)
    obj.sprongmotion.auto = False
    blocks.resize(obj, 0, end=s.blocks[0].start + blocks.limits()[1] + 50)
    s.blocks[1].text = "a person jumps"
    blocks.mark_dirty(obj, s.blocks[0].uid, s.blocks[1].uid)
    engine.run_now(obj)
    assert lengths(obj)[0] == blocks.limits()[1] and not model.error(), (lengths(obj), model.error())
    assert not any(timeline.needs_take(b, model.load().id) for b in s.blocks)
