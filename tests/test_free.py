import os
import tempfile
from types import SimpleNamespace

import bpy
import numpy as np
import undos
from bl_ext.smtest.sprongmotion import blocks, engine, lane
from test_copy import counting
from test_rest import ease_frames, pose, prefs, rest


class _Region:
    """2 px per frame, like region.view2d."""

    view2d = SimpleNamespace(view_to_region=lambda f, y, clip=False: (2 * f, y), region_to_view=lambda x, y: (x / 2, y))


def setup(gaps=(0, 0), auto=False):
    """Three 2 s blocks after the given gaps (frames), each with its own seed, made."""
    bpy.context.scene.render.fps, bpy.context.scene.frame_current = 24, 1
    bpy.context.scene.tool_settings.use_snap_anim = False
    prefs().after_block, prefs().ease_seconds = "HOLD", 0.4
    assert bpy.ops.sm.add_rig() == {"FINISHED"}, engine.error()
    obj = bpy.context.active_object
    s = obj.sprongmotion
    s.auto, s.key_density = auto, "EXACT"
    for k, text in enumerate(("a person walks forward", "a person waves", "a person jumps")):
        b = blocks.add(obj, text, 2)
        b.seed = k + 1
        if k:
            blocks.shift(blocks.settings(obj), [k], gaps[k - 1])
    engine.run_now(obj)
    return obj


def codes(obj):
    return [blocks.codes(b, engine.model().id).tolist() for b in blocks.settings(obj).blocks]


def drag(obj, uid, frames, shift=False):
    """A lane drag of block uid's middle by frames, as SM_OT_lane_click does it (mouse move, then release)."""
    s = blocks.settings(obj)
    b = s.blocks[blocks.find(s, uid)]
    snapshot = [(x.uid, x.start, x.end, x.locked, x.dirty, x.remake) for x in s.blocks]
    f0 = (b.start + b.end) / 2
    op = SimpleNamespace(kind="body", uid=uid, shift=shift, snapshot=snapshot, f0=f0, region=_Region,
                         before=blocks.continues(s))  # fmt: skip
    lane.SM_OT_lane_click.drag(op, obj, 2 * (f0 + frames))
    lane.SM_OT_lane_click.done(op, obj)


def alone(text, seed, frames):
    """The codes of one block made on a rig of its own."""
    keep = bpy.context.active_object
    bpy.ops.sm.add_rig()
    other = bpy.context.active_object
    other.sprongmotion.auto = False
    b = blocks.add(other, text, frames / blocks.fps())
    b.seed = seed
    engine.run_now(other)
    made = blocks.codes(b, engine.model().id).tolist()
    bpy.data.objects.remove(other)
    bpy.context.view_layer.objects.active = keep
    return made


def test_a_gap_starts_a_new_run():
    obj = setup()
    s = blocks.settings(obj)
    before = codes(obj)
    drag(obj, 3, 24)
    engine.run_now(obj)
    a, b, c = s.blocks
    assert c.start - b.end == 24 and [len(r) for r in blocks.runs(s)] == [2, 1]
    after = codes(obj)
    assert after[:2] == before[:2], "the blocks before the gap keep their takes"
    assert after[2] != before[2] and after[2] == alone(c.text, c.seed, c.end - c.start), "a take made alone"
    n, held = ease_frames(), pose(obj, b.end)
    for f in range(b.end, c.start - n):
        assert np.abs(pose(obj, f) - held).max() < 1e-4, f"Hold Last Pose in the gap at {f}"
    assert np.abs(pose(obj, c.start) - pose(obj, c.start - n)).max() > 1e-3, "it eases into the next block"
    prefs().after_block = "REST"
    assert np.abs(pose(obj, b.end + 12) - rest(obj, b.end)).max() < 1e-4, "Ease to Rest in the middle of the gap"
    assert np.abs(pose(obj, b.end) - held).max() < 1e-3, "the block before ends on its own pose"
    prefs().after_block = "HOLD"
    hips = [obj.pose.bones["pelvis"]]
    bpy.context.scene.frame_set(b.end)
    end = np.array(hips[0].head)
    bpy.context.scene.frame_set(c.start)
    assert np.linalg.norm((np.array(hips[0].head) - end)[:2]) < 0.15, "the run after the gap starts where it stood"


def test_a_drop_onto_a_block_goes_to_the_nearest_free_edge():
    obj = setup()
    s = blocks.settings(obj)
    a, b, c = ((x.start, x.end) for x in s.blocks)
    drag(obj, 1, c[0] + 10 - a[0])  # onto block 3, nearer its end than block 2's start
    assert [x.uid for x in s.blocks] == [2, 3, 1] and (s.blocks[2].start, s.blocks[2].end) == (c[1], c[1] + 48)
    assert (s.blocks[0].start, s.blocks[1].start) == (b[0], c[0]) and s.active == 2, "the others stay"
    drag(obj, 1, b[0] - c[1] + 20)  # back onto block 2, nearer its start: before it, where block 1 was
    assert [x.uid for x in s.blocks] == [1, 2, 3] and (s.blocks[0].start, s.blocks[0].end) == a


def test_touching_within_two_frames_continues():
    chained, near = codes(setup()), setup(gaps=(2, 1))
    assert codes(near) == chained, "1-2 frame gaps are made as one run"
    s = blocks.settings(near)
    assert len(blocks.runs(s)) == 1 and blocks.continues(s) == {1: 0, 2: 1, 3: 2}
    fc = next(fc for fc in undos.retarget.channelbag(near).fcurves if "l_knee" in fc.data_path)
    assert [k.co.x for k in fc.keyframe_points] == list(range(s.blocks[0].start, s.blocks[-1].end + 1))


def test_fill_gap_in_the_lane():
    obj = setup(gaps=(36, 0))
    s = blocks.settings(obj)
    before, made = codes(obj), counting()
    try:
        assert bpy.ops.sm.fill_gap(after=0, text="a person turns around") == {"FINISHED"}
    finally:
        del engine.model().blocks
    a, f, b, c = s.blocks
    assert (f.start, f.end, f.text) == (a.end, b.start, "a person turns around")
    assert made == [1] and codes(obj)[:1] + codes(obj)[2:] == before, "one block made, its neighbours kept"
    assert len(blocks.runs(s)) == 1 and blocks.continues(s)[f.uid] == a.uid and blocks.continues(s)[b.uid] == f.uid
    assert bpy.ops.sm.fill_gap(after=0) == {"CANCELLED"}, "no gap left"


def test_shift_drag_moves_all():
    obj = setup(gaps=(30, 0))
    s = blocks.settings(obj)
    spans, before, made = [(b.start, b.end) for b in s.blocks], codes(obj), counting()
    try:
        drag(obj, 2, 20, shift=True)
    finally:
        del engine.model().blocks
    assert [(b.start, b.end) for b in s.blocks] == [(x + 20, y + 20) for x, y in spans]
    assert not any(made) and codes(obj) == before


def test_one_undo_step_each():
    setup(gaps=(30, 0), auto=True)

    def fill(o):
        blocks.fill(o, blocks.settings(o).blocks[0].end, blocks.settings(o).blocks[1].start, "a person turns around")

    undos.one_undo(lambda o: drag(o, 3, 30), "move away")
    undos.one_undo(lambda o: drag(o, 1, -10, shift=True), "shift-move all", makes=False)
    undos.one_undo(fill, "fill a lane gap")


def test_an_old_chained_file_is_unchanged():
    obj = setup()
    m, s = engine.model(), blocks.settings(obj)
    with engine.timeline(obj) as (motion, segs):
        spans = [(b.uid, f0, f1) for b, f0, f1 in segs]
        whole = engine._decoded(s, engine._arrays(motion), spans)
    for x, y in zip(whole, engine._decode(obj), strict=False):
        assert np.array_equal(x, y), "one run decodes as one motion, as before"
    path = os.path.join(tempfile.mkdtemp(), "chained.blend")
    bpy.ops.wm.save_as_mainfile(filepath=path)
    bpy.ops.wm.open_mainfile(filepath=path)
    s = blocks.settings(bpy.data.objects[undos.NAME])
    assert not any(engine.needs_take(b, m.id) for b in s.blocks) and engine.fresh_uids(s) == set()


def test_duplicate_goes_right_after_and_shifts_the_touching_blocks():
    obj = setup(gaps=(0, 60))
    s = blocks.settings(obj)
    spans = [(b.start, b.end) for b in s.blocks]
    assert bpy.ops.sm.block_duplicate(index=0) == {"FINISHED"}
    assert [(b.start, b.end) for b in s.blocks] == [spans[0], spans[1], (spans[1][1], spans[1][1] + 48), spans[2]]
    assert bpy.ops.sm.block_duplicate(index=0) == {"FINISHED"}
    assert [b.end for b in s.blocks] == [spans[0][1] + 48 * k for k in range(4)] + [spans[2][1] + 36]


def test_the_lane_finds_gaps_and_both_edges():
    obj = setup(gaps=(30, 0))
    s, y = blocks.settings(obj), sum(lane.rect()) / 2
    a, b, c = s.blocks
    assert lane.hit(_Region, s, 2 * (a.end + 15), y) == ("gap", 0)
    assert lane.hit(_Region, s, 2 * a.end, y) == ("right", 0) and lane.hit(_Region, s, 2 * b.start, y) == ("left", 1)
    assert lane.hit(_Region, s, 2 * b.end, y) == ("right", 1), "where blocks touch: the left one's end"
