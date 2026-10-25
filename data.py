"""What the add-on keeps on a rig besides its settings: the Frozen bone collection, and taking it all out again."""

import bpy
from bpy_extras import anim_utils

from . import blocks, keys
from .retarget import channelbag

FROZEN = "Frozen"  # the bone collection of bones takes leave alone
COLOR = "THEME04"  # a quiet blue: frozen at a glance. Only bones without a colour of their own get it.


def why_not_editable(obj):
    if obj.library is not None:
        return f"{obj.name} is linked from {obj.library.name}: make a library override (Object > Library Override)"
    action = obj.animation_data.action if obj.animation_data else None
    if action is not None and not blocks.local(action) and blocks.stored(action) and len(action.sprongmotion.blocks):
        lib = action.library or action.override_library.reference.library
        return (f"The action {action.name} is linked from {lib.name}: its blocks are read-only. Make it local or pick "
                "another action in the Action Editor to change them")  # fmt: skip
    return None


def frozen(obj):
    coll = obj.data.collections.get(FROZEN)
    return {b.name for b in coll.bones} if coll else set()


def _paint(bone, on):
    old, new = ("DEFAULT", COLOR) if on else (COLOR, "DEFAULT")
    if bone.color.palette == old:
        bone.color.palette = new


def freeze(obj, names, on=True):
    """Freeze or unfreeze bones: frozen bones get no generated keys and keep theirs. None when done, else why not."""
    arm = obj.data
    if arm.library is not None:
        return f"{arm.name} is linked: make it a library override too (Library Override > Make > Selected & Content)"
    coll = arm.collections.get(FROZEN)
    if coll is None and on:
        try:
            coll = arm.collections.new(FROZEN)
        except RuntimeError as e:
            return f"Cannot add the {FROZEN} bone collection here: {e}"
    for name in names if coll else ():
        bone = arm.bones[name]
        (coll.assign if on else coll.unassign)(bone)
        _paint(bone, on)
    if on and obj.animation_data and obj.animation_data.action and obj.animation_data.action_slot:
        for fc in channelbag(obj).fcurves:
            if fc.data_path.startswith("pose.bones[") and fc.data_path.split('"')[1] in names:
                keys.retype(fc.keyframe_points, keys.GENERATED, keys.KEYFRAME)
    return None


def rest_pose(obj, names):
    for name in names:
        pb = obj.pose.bones.get(name)
        if pb is not None:
            pb.location, pb.rotation_euler, pb.scale = (0, 0, 0), (0, 0, 0), (1, 1, 1)
            pb.rotation_quaternion, pb.rotation_axis_angle = (1, 0, 0, 0), (0, 0, 1, 0)


def _bone(fc):
    return fc.data_path.split('"')[1] if fc.data_path.startswith("pose.bones[") else None


def _take_keys(obj, cb, bake):
    """obj: the rig playing the channels, whose bones the generated keys animated go back to rest (None: no rig)."""
    animated = set()
    for fc in list(cb.fcurves):
        if bake:
            keys.retype(fc.keyframe_points, keys.GENERATED, keys.KEYFRAME)
        else:
            generated = keys.get(fc.keyframe_points)[1] == keys.GENERATED
            animated |= {_bone(fc)} if generated.any() else set()
            keys.remove(fc.keyframe_points, generated)
        if not len(fc.keyframe_points):
            cb.fcurves.remove(fc)
    for group in [g for g in cb.groups if not g.channels]:
        cb.groups.remove(group)
    if obj is not None:
        rest_pose(obj, animated - {_bone(fc) for fc in cb.fcurves} - {None})


def clear_action(action, bake=True):
    """Remove the action's blocks and generated keys (bake: keep them as own keys), and the action if left empty."""
    players = blocks.players(action)
    for slot in action.slots:
        cb = anim_utils.action_get_channelbag_for_slot(action, slot)
        if cb is not None:
            _take_keys(next((o for o in players if o.animation_data.action_slot == slot), None), cb, bake)
    action.property_unset("sprongmotion")
    empty = not any(len(cb.fcurves) for layer in action.layers for strip in layer.strips for cb in strip.channelbags)
    if empty and action.users == len(players):
        for o in players:
            o.animation_data.action = None
            if not o.animation_data.drivers and not o.animation_data.nla_tracks:
                o.animation_data_clear()
        bpy.data.actions.remove(action)


def remove_data(obj, bake=True):
    """Take the add-on out of the action obj plays, and the rig's settings unless another action has blocks for it."""
    ad = obj.animation_data
    if ad and ad.action and blocks.local(ad.action):
        clear_action(ad.action, bake)
    if any(blocks.stored(a) and len(a.sprongmotion.blocks) and a.slots.get("OB" + obj.name) for a in bpy.data.actions):
        return
    arm = obj.data
    coll = arm.collections.get(FROZEN) if arm.library is None else None
    if coll is not None:
        for bone in coll.bones:
            _paint(bone, False)
        arm.collections.remove(coll)
    obj.property_unset("sprongmotion")


def remove_everywhere(bake=True):
    """Remove from the whole file: every local action's blocks and generated keys, every local rig's settings."""
    for action in [a for a in bpy.data.actions if blocks.local(a) and blocks.stored(a)]:
        clear_action(action, bake)
    for obj in [o for o in bpy.data.objects if o.type == "ARMATURE" and o.library is None]:
        remove_data(obj, bake)
