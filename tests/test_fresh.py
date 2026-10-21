import bpy
import numpy as np
from bl_ext.smtest.sprongmotion import blocks, engine, retarget
from test_generate import all_codes, codes, setup
from undos import one_undo


def keys(obj):
    return {(fc.data_path, fc.array_index): [tuple(k.co) for k in fc.keyframe_points]
            for fc in retarget.channelbag(obj).fcurves}  # fmt: skip


def heads(obj, frames):
    """[frames, bones, 3]: every pose bone's world head."""
    out = []
    for f in frames:
        bpy.context.scene.frame_set(int(f))
        out.append([np.array(obj.matrix_world @ pb.head) for pb in obj.pose.bones])
    return np.array(out)


def test_off_and_back_gives_the_same_keys():
    obj = setup()
    engine.run_now(obj)
    b = blocks.settings(obj).blocks[1]
    b.dirty = True  # made as an edit makes it: between its kept neighbours
    engine.run_now(obj)
    before, made = keys(obj), all_codes(obj)
    for on in (True, False):
        b.fresh = on
        engine.run_now(obj)
        assert (keys(obj) == before) != on
    assert keys(obj) == before and all((a == c).all() for a, c in zip(made, all_codes(obj), strict=True))


def test_the_take_is_the_block_made_alone():
    obj = setup()
    engine.run_now(obj)
    b = blocks.settings(obj).blocks[2]
    b.fresh = True
    engine.run_now(obj)
    m = engine.model()
    with m.blocks([b.text], [b.tokens * m.fpt / m.fps], [b.seed], [None]) as (alone, _):
        assert np.array_equal(alone.codes(), codes(b))
    first = blocks.settings(obj).blocks[0]
    first.dirty = True
    engine.run_now(obj)
    before = all_codes(obj)
    first.fresh = True
    engine.run_now(obj)
    assert all((a == c).all() for a, c in zip(before, all_codes(obj), strict=True)), "the first block: nothing"


def test_no_step_at_the_seam():
    obj = setup()
    s = blocks.settings(obj)
    obj.sprongmotion.key_density = "EXACT"
    s.blocks[1].fresh = True
    engine.run_now(obj)
    b = s.blocks[1]
    pos = heads(obj, range(b.start - 4, b.end))
    steps = np.linalg.norm(np.diff(pos, axis=0), axis=2).mean(1)
    before, seam, inside = steps[:3], steps[3], steps[4:]
    own = max(np.median(before), np.median(inside))  # the motion's own speed on either side of the seam
    assert seam <= 1.5 * own, (seam, np.median(before), np.median(inside))  # 12 seeds: 0.6-1.2x
    root = pos[2:5, list(obj.pose.bones).index(obj.pose.bones["pelvis"])]
    assert np.linalg.norm(root[2] - 2 * root[1] + root[0]) < 0.01, "the root carries on"


def test_toggling_makes_one_take_in_one_undo_step():
    obj = setup()
    engine.run_now(obj)
    obj.sprongmotion.auto = True
    m, made, run = engine.model(), [], engine.model().blocks

    def counted(texts, seconds, seeds, keep, *a):
        made.append(sum(k is None for k in keep))
        return run(texts, seconds, seeds, keep, *a)

    m.blocks = counted
    try:
        blocks.settings(obj).blocks[1].fresh = True
    finally:
        del m.blocks
    assert [n for n in made if n] == [1], made

    def toggle(o):
        blocks.settings(o).blocks[1].fresh = False

    one_undo(toggle, "fresh start")
