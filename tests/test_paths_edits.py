"""Path edits by hand and through undo: a hand resize, its mark across undo and reload, unlocking, Cmd+Z after an
edit, and per-block actions. Apart from test_paths.py so both files run side by side."""

import os
import tempfile

import bpy
import numpy as np
from bl_ext.smtest.sprongmotion import blocks, engine, paths
from test_paths import WALK, lengths, pelvis_xy, rig, s_curve, unwatch, watch
from undos import NAME, edits, one_undo


def mark(i):
    b = blocks.settings(bpy.data.objects[NAME]).blocks[i]
    return b.remake, b.locked, engine.needs_take(b, engine.model().id)


def test_a_hand_resize_locks_survives_undo_and_reload_and_unlocking_gives_it_back_to_the_fit():
    obj = rig((WALK, 2), ("a person stands still", 1.5))  # a block that does not travel: the fit leaves it alone
    s = blocks.settings(obj)
    s.path = s_curve()
    fitted, b, fit, fixed = lengths(obj)[0], s.blocks[0], paths.fit, []
    paths.fit = lambda spans, speed, fps, total, held: fixed.append(held[0]) or fit(spans, speed, fps, total, held)
    made, out = watch()
    try:
        blocks.resize(obj, 0, end=b.start + 2 * fitted)  # what a lane edge drag does, then on release:
        blocks.mark_dirty(obj, b.uid)
        assert lengths(obj)[0] == 2 * fitted and b.locked and not b.dirty and not out, (lengths(obj), made, out)
        assert made[0] == 1 and sum(made) == 1 and fixed == [True], "remade at the hand-set length, kept by the fit"
    finally:
        unwatch()
        paths.fit = fit

    obj.sprongmotion.auto = False  # the mark waits for its take: through undo, redo and a reload
    bpy.ops.ed.undo_push(message="before")
    blocks.resize(obj, 1, end=s.blocks[1].end + 12)
    bpy.ops.ed.undo_push(message="resize")
    assert mark(1) == (True, True, True)
    bpy.ops.ed.undo()
    assert mark(1)[:2] == (False, False)
    bpy.ops.ed.redo()
    assert mark(1) == (True, True, True)
    saved = os.path.join(tempfile.mkdtemp(), "remake.blend")
    bpy.ops.wm.save_as_mainfile(filepath=saved)
    bpy.ops.wm.open_mainfile(filepath=saved)
    assert mark(1) == (True, True, True)
    obj = bpy.data.objects[NAME]
    engine.run_now(obj)
    assert mark(1) == (False, True, False), "made once at the new length, then kept"

    obj.sprongmotion.auto, fixed[:] = True, []
    paths.fit = lambda spans, speed, fps, total, held: fixed.append(held[0]) or fit(spans, speed, fps, total, held)
    made, out = watch()
    try:
        blocks.settings(obj).blocks[0].locked = False
    finally:
        unwatch()
        paths.fit = fit
    print(f"> unlock: blocks made per call {made}, length {2 * fitted} -> {lengths(obj)[0]}")
    assert fixed == [False] and not out, "unlocking refits it in the same undo step"


def test_with_a_path_one_cmd_z_undoes_an_edit_and_a_block_action_follows_its_part():
    blocks.settings(rig((WALK, 2), ("a person waves", 1.5))).path = s_curve()
    for label, edit in reversed(edits()):  # the text first: a resize with a path locks the block
        one_undo(edit, label + " with a path")
    obj = bpy.data.objects[NAME]
    walk = engine.block_actions(obj, ["walk", "wave"])[0]
    ad = obj.animation_data
    ad.action, ad.action_slot = walk, walk.slots[0]
    xy = pelvis_xy(obj, range(1, int(walk.frame_range[1]) + 1))
    chord = (xy[-1] - xy[0]) / np.linalg.norm(xy[-1] - xy[0])
    side = np.abs(np.cross(chord, xy - xy[0]))
    t = np.linspace(0, 1, 200)
    c = np.c_[0.4 * np.sin(t * 3), 2.5 * t]
    bend = np.abs(np.cross((c[-1] - c[0]) / np.linalg.norm(c[-1] - c[0]), c - c[0])).max()
    print(f"> block action bends {side.max():.2f} m off its chord, the curve {bend:.2f} m")
    assert np.linalg.norm(xy[0]) < 0.02 and abs(side.max() - bend) < 0.1, "from the origin, bent like the path"
