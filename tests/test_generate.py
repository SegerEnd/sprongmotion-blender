import bpy
import numpy as np
from bl_ext.smtest.sprongmotion import blocks, generate, model, retarget, takes, timeline_motion


def new_rig(auto=False, fps=24):
    """The add-on's own rig, the scene at fps and frame 1."""
    bpy.context.scene.render.fps, bpy.context.scene.frame_current = fps, 1
    assert bpy.ops.sm.add_rig() == {"FINISHED"}, model.error()
    obj = bpy.context.active_object
    obj.sprongmotion.auto = auto
    return obj


def setup(auto=False):
    obj = new_rig(auto)
    for text, seconds in (("a person walks forward", 2), ("a person jumps", 1), ("a person sits down", 1.5)):
        blocks.add(obj, text, seconds)
    return obj


def codes(b, variant=None):
    return blocks.codes(b, model.load().id, variant)


def all_codes(obj):
    return [codes(b).copy() for b in blocks.settings(obj).blocks]


def curve(obj, path='pose.bones["l_knee"].rotation_quaternion', index=1):
    return retarget.channelbag(obj).fcurves.find(path, index=index)


def test_generate_chain_and_keys():
    obj = setup()
    obj.sprongmotion.key_density = "EXACT"
    generate.run_now(obj)
    s = blocks.settings(obj)
    assert not model.error(), model.error()
    assert all(not b.dirty and b.variants == 1 for b in s.blocks)
    assert [b.tokens for b in s.blocks] == [20, 10, 15]
    fc = curve(obj)
    assert [k.co.x for k in fc.keyframe_points] == list(range(1, s.blocks[-1].end + 1))
    assert {k.type for k in fc.keyframe_points} == {"GENERATED"}
    assert bpy.ops.sm.fit_scene() == {"FINISHED"} and bpy.context.scene.frame_end == s.blocks[-1].end
    hips = curve(obj, 'pose.bones["pelvis"].location', 1)
    assert hips.keyframe_points[-1].co.y != hips.keyframe_points[0].co.y, "the root travels"


def test_edit_regenerates_block_and_later_only():
    obj = setup(auto=True)
    before = all_codes(obj)
    s = blocks.settings(obj)
    s.blocks[2].locked = True
    s.blocks[1].text = "a person waves with both hands"
    after = all_codes(obj)
    assert not any(b.dirty and not b.locked for b in s.blocks), "a small edit generates at once"
    assert (after[0] == before[0]).all() and (after[2] == before[2]).all()
    assert not (after[1] == before[1]).all()


def test_same_seed_same_take():
    obj = setup()
    generate.run_now(obj)
    before = all_codes(obj)
    for b in blocks.settings(obj).blocks:
        b.dirty = True
    generate.run_now(obj)
    assert all((a == b).all() for a, b in zip(before, all_codes(obj), strict=True))


def test_timing_stretches_each_take_onto_its_block():
    obj = setup()
    generate.run_now(obj)
    with timeline_motion.whole(obj) as (_, segs):
        frames, times = takes.timing(segs)
    b = blocks.settings(obj).blocks
    assert frames[0] == b[0].start and frames[-1] == b[-1].end
    assert times[int(np.flatnonzero(frames == b[1].start)[0])] == 2 * b[0].tokens


def test_unknown_words():
    assert "pirouettez" in model.unknown_words("a person does a pirouettez")
    assert model.unknown_words("a person walks forward") == []
