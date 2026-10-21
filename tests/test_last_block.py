import bpy
from bl_ext.smtest.sprongmotion import blocks, engine, retarget
from test_rest import prefs


def last_keys(obj):
    """Each keyed bone's last key frame (its earliest over the curves)."""
    out = {}
    for fc in retarget.channelbag(obj).fcurves:
        bone = fc.data_path.split('"')[1]
        out[bone] = min(out.get(bone, float("inf")), fc.keyframe_points[-1].co[0])
    return out


def test_a_newly_added_last_block_is_keyed_over_its_whole_range():
    bpy.context.scene.render.fps, bpy.context.scene.frame_current = 24, 1
    assert bpy.ops.sm.add_rig() == {"FINISHED"}, engine.error()
    obj = bpy.context.active_object
    obj.sprongmotion.key_density = "UNBAKED"
    for k, text in enumerate(("a person walks forward", "a person waves")):
        blocks.add(obj, text, 2).seed = k + 1
    engine.run_now(obj)
    last = blocks.add(obj, "a person does jumping jacks", 3)
    engine.run_now(obj)
    try:
        for after in ("HOLD", "REST"):
            for in_place in (False, True):
                prefs().after_block = after
                for b in blocks.settings(obj).blocks:
                    b.in_place = in_place
                engine.apply(obj, push=False)
                short = {n: f for n, f in last_keys(obj).items() if f < last.end}
                assert not short, f"{after}, in place {in_place}: keys stop before {last.end}: {short}"
    finally:
        prefs().after_block = "HOLD"


def test_a_half_written_bone_map_is_filled_again():
    """A map an interrupted fill left with 2 of 22 entries keys only those bones: the next apply fills it again."""
    bpy.context.scene.render.fps, bpy.context.scene.frame_current = 24, 1
    assert bpy.ops.sm.add_rig() == {"FINISHED"}, engine.error()
    obj = bpy.context.active_object
    blocks.add(obj, "a person does jumping jacks", 2)
    engine.run_now(obj)
    keyed = set(last_keys(obj))
    while len(obj.sprongmotion.bones) > 2:
        obj.sprongmotion.bones.remove(len(obj.sprongmotion.bones) - 1)
    engine.apply(obj, push=False)
    assert len(obj.sprongmotion.bones) == 22 and set(last_keys(obj)) == keyed
