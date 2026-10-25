from types import SimpleNamespace

import bpy
from bl_ext.smtest.sprongmotion import blocks, generate, panels, retarget, rig_data

ARM = ("l_shoulder", "l_elbow")


def setup(take=True):
    bpy.context.scene.render.fps = 24
    bpy.context.scene.frame_current = 1
    bpy.ops.sm.add_rig()
    obj = bpy.context.active_object
    obj.sprongmotion.auto = False
    if take:
        blocks.add(obj, "a person walks forward and waves", 3).seed = 2
        generate.run_now(obj)
    return obj


def curve(obj, bone, index=1):
    return retarget.channelbag(obj).fcurves.find(f'pose.bones["{bone}"].rotation_quaternion', index=index)


def keyset(obj, bone):
    return [(k.co.x, round(k.co.y, 6), k.type) for k in curve(obj, bone).keyframe_points]


def freeze(obj, names, **kw):
    bpy.ops.object.mode_set(mode="POSE")
    for pb in obj.pose.bones:
        pb.select = pb.name in names
    assert bpy.ops.sm.freeze(**kw) == {"FINISHED"}


def test_frozen_bones_keep_what_they_show():
    obj = setup()
    curve(obj, "l_elbow").keyframe_points.insert(500, 0.25).type = "KEYFRAME"
    before = keyset(obj, "l_shoulder")
    pushes, push = [], generate.undo_push
    generate.undo_push = pushes.append
    try:
        freeze(obj, ARM)
    finally:
        generate.undo_push = push
    assert not pushes, "freezing and its re-apply are the operator's one undo step"
    assert rig_data.frozen(obj) == set(ARM) and obj.data.bones["l_elbow"].color.palette == "THEME04"
    got = keyset(obj, "l_shoulder")
    assert [(x, y) for x, y, _ in got] == [(x, y) for x, y, _ in before] and {t for *_, t in got} == {"KEYFRAME"}
    blocks.settings(obj).blocks[0].text = "a person jumps"
    generate.run_now(obj)
    assert keyset(obj, "l_shoulder") == got, "a new take leaves frozen bones alone"
    assert 500 in [k.co.x for k in curve(obj, "l_elbow").keyframe_points], "the user's own key stays"
    assert {k.type for k in curve(obj, "l_knee").keyframe_points} == {"GENERATED"}
    freeze(obj, (), unfreeze=True, every=True)
    assert not rig_data.frozen(obj) and {t for *_, t in keyset(obj, "l_shoulder")} == {"GENERATED"}


def test_no_leg_ik_on_a_frozen_leg():
    obj = setup()
    legs, plant = [], retarget.Target._plant
    retarget.Target._plant = lambda self, absolute, move, planted, names: legs.append((names, planted.any()))
    try:
        freeze(obj, ("l_hip", "l_knee", "l_ankle"))
    finally:
        retarget.Target._plant = plant
    assert [on for names, on in legs if "l_hip" in names] == [False] and any(on for n, on in legs if "r_hip" in n)


def test_own_bone_colours_stay():
    obj = setup(take=False)
    obj.data.bones["l_elbow"].color.palette = "THEME09"
    freeze(obj, ARM)
    assert [obj.data.bones[n].color.palette for n in ARM] == ["THEME04", "THEME09"]
    freeze(obj, ARM, unfreeze=True)
    assert [obj.data.bones[n].color.palette for n in ARM] == ["DEFAULT", "THEME09"]
    freeze(obj, ARM)
    bpy.ops.object.mode_set(mode="OBJECT")
    assert bpy.ops.sm.remove_data() == {"FINISHED"}
    assert [obj.data.bones[n].color.palette for n in ARM] == ["DEFAULT", "THEME09"]


class _Layout:
    def __init__(self):
        self.items = []

    def separator(self):
        pass

    def operator(self, idname, **kw):
        self.items.append((idname, kw.get("text")))
        return SimpleNamespace()


def test_pose_menus_offer_freeze():
    setup(take=False)
    bpy.ops.object.mode_set(mode="POSE")
    for menu, fn in panels.MENUS[1:]:
        me = SimpleNamespace(layout=_Layout())
        fn(me, bpy.context)
        assert [i for i, _ in me.layout.items] == ["sm.freeze", "sm.freeze"], menu
