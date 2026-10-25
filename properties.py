import bpy
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)

from . import blocks, generate, model, paths, return_to_start, rig


def _player_rig(owner):
    return blocks.player(owner.id_data)


def _on_block_edit(block, _context):
    model.unknown_words(block.text)
    if obj := _player_rig(block):
        blocks.mark_dirty(obj, block.uid)


def _on_setting_change(owner, _context):
    if obj := _player_rig(owner):
        generate.edited(obj)


def _get_in_place(timeline):
    return len(timeline.blocks) > 0 and all(b.in_place for b in timeline.blocks)


def _set_in_place(timeline, value):
    for b in timeline.blocks:
        b["in_place"] = value  # ID writes: one reapply below, not one per block
    _on_setting_change(timeline, None)


TIMELINE = ("blocks", "active", "next_uid", "loop", "path", "path_key", "path_length", "path_fit", "take_due")


def upgrade(obj):
    """Move blocks, Loop and path kept on the rig (older files) bit for bit to the action it plays."""
    old = (obj.bl_system_properties_get() or {}).get("sprongmotion")
    if old is None:
        return
    if "blocks" in old:
        ad = obj.animation_data
        if ad and ad.action and blocks.stored(ad.action) and len(ad.action.sprongmotion.blocks):
            ad.action = ad.action.copy()
            ad.action.property_unset("sprongmotion")
        new = blocks.settings(obj, create=True).id_data.bl_system_properties_get()["sprongmotion"]
        for k in TIMELINE:
            if k in old:
                new[k] = old[k]
                del old[k]
    if "in_place" in old:
        t = blocks.settings(obj)
        for b in t.blocks if t else ():
            b["in_place"] = b.in_place or bool(old["in_place"])
        del old["in_place"]


def _on_preset(settings, _context):
    rig.fill_map(settings.id_data)
    generate.edited(settings.id_data)


def _on_lock(block, _context):
    obj = _player_rig(block)
    fitted = obj and (paths.path(obj) is not None or return_to_start.walks_back(block.id_data.sprongmotion, block))
    if fitted and not block.locked:
        generate.edited(obj)  # back to the fit


def _on_prompt(settings, _context):
    if settings.prompt.strip():
        text, settings.prompt = settings.prompt.strip(), ""
        blocks.add(settings.id_data, text)


def _on_path(timeline, _context):
    if obj := _player_rig(timeline):
        paths.attached(obj)


def _redraw_lane(_self, _context):
    blocks.redraw()


def _get_seconds(block):
    return (block.end - block.start) / blocks.fps()


def _set_seconds(block, value):
    obj = _player_rig(block)
    if obj is None:
        return
    i = blocks.find(block.id_data.sprongmotion, block.uid)
    blocks.resize(obj, i, end=block.start + round(value * blocks.fps()), follow=True)
    blocks.mark_dirty(obj, block.uid, wait=True)


def _overridable(cls):
    """Editable on library-override armatures (a linked character), like Blender's own properties."""
    for prop in cls.__annotations__.values():
        prop.keywords["override"] = {"LIBRARY_OVERRIDABLE"} | (
            {"USE_INSERTION"} if prop.function is CollectionProperty else set()
        )
    return cls


class SMBlock(bpy.types.PropertyGroup):
    uid: IntProperty()
    text: StringProperty(name="Text", update=_on_block_edit)
    start: IntProperty(name="Start")
    end: IntProperty(name="End")
    seconds: FloatProperty(
        name="Seconds", get=_get_seconds, set=_set_seconds, min=blocks.MIN_SECONDS, max=blocks.MAX_SECONDS, step=10
    )
    seed: IntProperty(name="Seed", min=0, max=2**31 - 1, update=_on_block_edit)
    color: FloatVectorProperty(name="Colour", subtype="COLOR", size=3, min=0, max=1, update=_redraw_lane)
    locked: BoolProperty(
        name="Locked",
        description="Keep this block's motion when blocks before it change (resizing it stretches the motion in time). "
        "With a path: keep its length",
        update=_on_lock,
    )
    enabled: BoolProperty(name="Enabled", default=True, update=_on_block_edit)
    variants: IntProperty(min=0)
    variant: IntProperty(name="Take", min=0, update=_on_setting_change)
    tokens: IntProperty(min=0)
    dirty: BoolProperty(default=True)
    remake: BoolProperty(options={"HIDDEN"})  # resized by hand with a path: locked, but made once at its new length
    path_note: StringProperty()  # why the rig's path is more than this block's motion follows, "" if it fits
    continues: BoolProperty(
        name="Continue from Keys",
        description="Where the rig's own keys end right before this block (or start again right after it), the motion "
        "continues from them (and leads into them)",
        default=True,
        update=_on_block_edit,
    )
    fresh: BoolProperty(
        name="Fresh Start",
        description="Make this block's motion without the block before it, then blend into it over 0.4 s",
        update=_on_block_edit,
    )
    in_place: BoolProperty(
        name="In Place",
        description="No horizontal root travel in this block: the block after it goes on from where it stands",
        update=_on_setting_change,
    )
    back_to_start: BoolProperty(
        name="Return to Start",
        description="This block walks back to where its run started, on a way the add-on makes: its length follows "
        "it at the block's gait",
        update=_on_setting_change,
    )
    anchor_key: StringProperty(default="|")  # the keys the take joins, continue_keys.block_keys: "before|after" hashes
    seam_note: StringProperty()  # why the keys after this block are further off than the take can blend in


@_overridable
class SMBone(bpy.types.PropertyGroup):
    joint: StringProperty()
    bone: StringProperty(name="Bone")


@_overridable
class SMSettings(bpy.types.PropertyGroup):
    """The rig's own settings: the bone map, how takes are made and keyed. The blocks are on the action it plays."""

    prompt: StringProperty(name="Motion", description="Describe a motion: Enter adds it as a block", update=_on_prompt)
    preset: EnumProperty(name="Rig", items=rig.PRESET_ITEMS, default="AUTO", update=_on_preset)
    bones: CollectionProperty(type=SMBone)
    variants: IntProperty(name="Takes", description="Takes made per block to pick from", default=1, min=1, max=8)
    foot_lock: BoolProperty(
        name="Foot Lock", description="Planted feet stay put", default=True, update=_on_setting_change
    )
    hand_lock: BoolProperty(
        name="Hand Lock",
        description="Hands on the floor stay put, on the floor",
        default=True,
        update=_on_setting_change,
    )
    floor_clamp: BoolProperty(
        name="Floor", description="No part of the body goes below the floor", default=True, update=_on_setting_change
    )
    key_density: EnumProperty(
        name="Keys",
        items=[
            ("UNBAKED", "Unbaked", "A few Bezier keys that look the same as a key on every frame"),
            ("EXACT", "Exact", "A key on every frame"),
        ],
        update=_on_setting_change,
    )
    key_error: FloatProperty(
        name="Key Error", description="How far unbaked keys may stray (degrees, cm)", default=0.5, min=0.05, max=10,
        update=_on_setting_change,
    )  # fmt: skip
    key_frames: EnumProperty(
        name="Key Frames",
        items=[
            ("BONE", "Per Bone", "Each bone gets the keys it needs"),
            ("SHARED", "Same Frames", "One set of key frames for all bones, for posing"),
        ],  # fmt: skip
        update=_on_setting_change,
    )
    auto: BoolProperty(name="Auto", description="Regenerate after every edit", default=True)


class SMTimeline(bpy.types.PropertyGroup):
    """The blocks of an action, with what belongs to its timeline: Loop and the path. Not overridable: the blocks of
    a linked or overridden action are read-only (rig_data.why_not_editable)."""

    blocks: CollectionProperty(type=SMBlock)
    active: IntProperty(name="Active block", update=_redraw_lane)
    next_uid: IntProperty(default=1)
    loop: BoolProperty(name="Loop", description="The timeline closes into a seamless cycle", update=_on_setting_change)
    in_place: BoolProperty(
        name="In Place", description="No horizontal root travel: sets or clears In Place on every block (on when all "
        "blocks are in place)", get=_get_in_place, set=_set_in_place,
    )  # fmt: skip
    path: PointerProperty(
        type=bpy.types.Object,
        name="Path",
        description="A curve the root follows over the whole timeline; the walking blocks' lengths fit it",
        poll=lambda _s, obj: obj.type == "CURVE",
        update=_on_path,
    )
    path_key: StringProperty()  # the curve as the keys were last made for it
    path_length: FloatProperty()  # metres in the model's frame when the block lengths were last fitted to it
    path_fit: StringProperty()  # the blocks as they were after that fit
    take_due: BoolProperty(options={"HIDDEN"})  # a take for this state is being made from the timer, in a later step
