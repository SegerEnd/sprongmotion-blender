import random

import bpy
from bpy.props import BoolProperty, CollectionProperty, EnumProperty, IntProperty, StringProperty
from bpy_extras.io_utils import ExportHelper

from . import anchor, blocks, data, engine, keys, paths, retarget, rig, sm


def editable(context):
    obj = blocks.armature(context)
    return obj is not None and data.why_not_editable(obj) is None


_has_blocks = classmethod(lambda cls, context: blocks.has_blocks(context) and editable(context))
_has_rig = classmethod(lambda cls, context: editable(context))


class _BlockOp:
    bl_options = {"REGISTER", "UNDO"}
    index: IntProperty(default=-1, options={"SKIP_SAVE"})
    poll = _has_blocks


class SM_OT_block_add(bpy.types.Operator):
    """Add a block with the motion text after the last one"""

    bl_idname = "sm.block_add"
    bl_label = "Add Motion"
    bl_options = {"REGISTER", "UNDO"}
    text: StringProperty(name="Text", options={"SKIP_SAVE"})
    poll = _has_rig

    def execute(self, context):
        s = context.active_object.sprongmotion
        blocks.add(context.active_object, self.text or s.prompt)
        s.prompt = ""
        return {"FINISHED"}


class SM_OT_block_remove(_BlockOp, bpy.types.Operator):
    """Remove the block; the ones after it move up"""

    bl_idname = "sm.block_remove"
    bl_label = "Remove Block"

    def execute(self, context):
        blocks.remove(*blocks.pick(context, self.index))
        return {"FINISHED"}


class SM_OT_block_move(_BlockOp, bpy.types.Operator):
    """Swap the block with its neighbour"""

    bl_idname = "sm.block_move"
    bl_label = "Move Block"
    step: IntProperty(default=1)

    def execute(self, context):
        obj, i = blocks.pick(context, self.index)
        if not 0 <= i + self.step < len(blocks.settings(obj).blocks):
            return {"CANCELLED"}
        blocks.move(obj, i, i + self.step)
        return {"FINISHED"}


class SM_OT_block_split(_BlockOp, bpy.types.Operator):
    """Split the block in two at a frame (default: the playhead)"""

    bl_idname = "sm.block_split"
    bl_label = "Split Block"
    frame: IntProperty(default=-1, options={"SKIP_SAVE"})

    def execute(self, context):
        frame = self.frame if self.frame >= 0 else context.scene.frame_current
        if blocks.split(*blocks.pick(context, self.index), frame) is None:
            self.report({"WARNING"}, "Too short to split here")
            return {"CANCELLED"}
        return {"FINISHED"}


class Clipboard:
    """Blender's clipboard (background Blender has none: tests swap in their own)."""

    get = staticmethod(lambda: bpy.context.window_manager.clipboard)
    set = staticmethod(lambda text: setattr(bpy.context.window_manager, "clipboard", text))


class SM_OT_block_copy(bpy.types.Operator):
    """Copy the block (text, length, seed, colour, lock, Fresh Start and its takes) to the clipboard"""

    bl_idname = "sm.block_copy"
    bl_label = "Copy Block"
    index: IntProperty(default=-1, options={"SKIP_SAVE"})
    poll = _has_blocks

    def execute(self, context):
        obj, i = blocks.pick(context, self.index)
        Clipboard.set(blocks.to_clip(blocks.settings(obj).blocks[i]))
        return {"FINISHED"}


class SM_OT_block_paste(bpy.types.Operator):
    """Paste a copied block after the block (on an empty lane: at the playhead); the later blocks move on"""

    bl_idname = "sm.block_paste"
    bl_label = "Paste Block"
    bl_options = {"REGISTER", "UNDO"}
    index: IntProperty(default=-1, options={"SKIP_SAVE"})
    poll = _has_rig

    def execute(self, context):
        obj = context.active_object
        after = blocks.pick(context, self.index)[1] if blocks.has_blocks(context) else -1
        if blocks.paste(obj, Clipboard.get(), after) is None:
            self.report({"INFO"}, "No SprongMotion block on the clipboard")
            return {"CANCELLED"}
        return {"FINISHED"}


class SM_OT_block_duplicate(_BlockOp, bpy.types.Operator):
    """Insert a copy of the block right after it, with the same take; the later blocks move on"""

    bl_idname = "sm.block_duplicate"
    bl_label = "Duplicate Block"

    def execute(self, context):
        obj, i = blocks.pick(context, self.index)
        blocks.paste(obj, blocks.to_clip(blocks.settings(obj).blocks[i]), i)
        return {"FINISHED"}


class SM_OT_block_regenerate(_BlockOp, bpy.types.Operator):
    """Make a new take of this block (and the unlocked blocks after it)"""

    bl_idname = "sm.block_regenerate"
    bl_label = "Regenerate"
    new_seed: BoolProperty(name="New Seed")

    def execute(self, context):
        obj, i = blocks.pick(context, self.index)
        b = blocks.settings(obj).blocks[i]
        with engine.hold():
            b.locked = False
            if self.new_seed:
                b.seed = random.randrange(2**31)
            blocks.mark_dirty(obj, b.uid)
        return {"FINISHED"}


class SM_OT_block_regenerate_range(_BlockOp, bpy.types.Operator):
    """Make a new take of a range of the block (default: the scene's preview range); the rest stays exactly"""

    bl_idname = "sm.block_regenerate_range"
    bl_label = "Regenerate Range"
    frame_start: IntProperty(name="Start", default=-1, options={"SKIP_SAVE"})
    frame_end: IntProperty(name="End", default=-1, options={"SKIP_SAVE"})

    def execute(self, context):
        obj, i = blocks.pick(context, self.index)
        scene, f0, f1 = context.scene, self.frame_start, self.frame_end
        if f0 < 0 or f1 < 0:
            if not scene.use_preview_range:
                self.report({"WARNING"}, "Set a preview range (Ctrl+Alt+P) or Ctrl+drag over the block")
                return {"CANCELLED"}
            f0, f1 = scene.frame_preview_start, scene.frame_preview_end
        engine.cancel()
        why = engine.remake_range(obj, i, min(f0, f1), max(f0, f1))
        if why:
            self.report({"WARNING"}, why)
            return {"CANCELLED"}
        return {"FINISHED"}


class SM_OT_fill_gap(bpy.types.Operator):
    """Fill the gap between the selected keys (or after block After in the lane) with motion that continues what is
    before it and leads into what is after it (Text: what happens in it)"""

    bl_idname = "sm.fill_gap"
    bl_label = "Fill Gap"
    bl_options = {"REGISTER", "UNDO"}
    text: StringProperty(name="Text", description="What happens in the gap", default="a person walks forward")
    after: IntProperty(name="After", description="The block before a gap in the lane, -1: the keys' gap", default=-1,
                       options={"SKIP_SAVE"})  # fmt: skip
    poll = _has_rig

    def execute(self, context):
        obj = context.active_object
        span, why = self.lane_gap(obj) if self.after >= 0 else self.keys_gap(obj, context.scene.frame_current)
        if span is not None and span[1] - span[0] < round(blocks.MIN_SECONDS * blocks.fps()):
            why = f"The gap is shorter than {blocks.MIN_SECONDS:g} s"
        if why or engine.model() is None:
            self.report({"WARNING"}, why or engine.error())
            return {"CANCELLED"}
        if self.after < 0:
            anchor.clear(obj, *span)
        blocks.fill(obj, *span, self.text.strip() or "a person walks forward")
        engine.edited(obj, force=True)
        return {"FINISHED"}

    def keys_gap(self, obj, frame):
        why = anchor.why_not(obj, engine.model())
        span = anchor.selected_gap(obj, frame)
        if blocks.has_blocks(bpy.context):
            why = "This rig has blocks already: bake or remove them first, or fill a gap between blocks in the lane"
        elif span is None:
            why = "Select keys on both sides of a gap first"
        return span, why

    def lane_gap(self, obj):
        """Between block after and the next one: both keep their takes, the new block goes on from one into the next."""
        s = blocks.settings(obj)
        if self.after + 1 >= len(s.blocks) or s.blocks[self.after + 1].start - s.blocks[self.after].end <= blocks.TOUCH:
            return None, "No gap after this block"
        return (s.blocks[self.after].end, s.blocks[self.after + 1].start), None


class SM_OT_variant(_BlockOp, bpy.types.Operator):
    """Show the block's next or previous take"""

    bl_idname = "sm.variant"
    bl_label = "Cycle Take"
    step: IntProperty(default=1)

    def execute(self, context):
        obj, i = blocks.pick(context, self.index)
        b = blocks.settings(obj).blocks[i]
        if b.variants < 2:
            return {"CANCELLED"}
        b.variant = (b.variant + self.step) % b.variants
        return {"FINISHED"}


class SM_OT_generate(bpy.types.Operator):
    """Make takes for the blocks that need one and put the motion on the armature"""

    bl_idname = "sm.generate"
    bl_label = "Generate"
    bl_options = {"REGISTER", "UNDO"}
    poll = _has_blocks

    def execute(self, context):
        if engine.model() is None:
            self.report({"ERROR"}, engine.error())
            return {"CANCELLED"}
        engine.edited(context.active_object, force=True)
        return {"FINISHED"}


class SM_OT_cancel(bpy.types.Operator):
    """Stop generating; unfinished blocks stay marked"""

    bl_idname = "sm.cancel"
    bl_label = "Cancel"
    poll = classmethod(lambda cls, _context: engine.busy())

    def execute(self, _context):
        engine.cancel()
        blocks.redraw()
        return {"FINISHED"}


class SM_OT_bake(bpy.types.Operator):
    """Keep the motion as ordinary keys and remove the blocks; optionally also one action per block, for game clips"""

    bl_idname = "sm.bake"
    bl_label = "Bake to Keys"
    bl_options = {"REGISTER", "UNDO"}
    poll = _has_blocks
    keys: EnumProperty(
        name="Keys",
        items=[
            ("0.5", "Invisible", "Unbaked within 0.5 degrees and 0.5 cm"),
            ("1.5", "Light", "Unbaked within 1.5 degrees and 1.5 cm"),
            ("4", "Strong", "Unbaked within 4 degrees and 4 cm: fewest keys"),
            ("EXACT", "Exact", "A key on every frame"),
        ],
    )
    actions: EnumProperty(
        name="Actions",
        items=[
            ("ONE", "One Action", "The whole timeline on the rig's action"),
            ("PER_BLOCK", "One Action per Block", "Also each block as its own action from frame 1 (the rig keeps "
             "the timeline)"),
        ],
    )  # fmt: skip
    names: CollectionProperty(type=bpy.types.PropertyGroup, options={"SKIP_SAVE"})
    asset: BoolProperty(name="Mark as Assets", description="Show the block actions in the Asset Browser")

    def invoke(self, context, _event):
        self.names.clear()
        for name in engine.action_names(blocks.settings(context.active_object)):
            self.names.add().name = name
        return self.execute(context)

    def draw(self, _context):
        layout = self.layout
        layout.prop(self, "keys")
        layout.prop(self, "actions", expand=True)
        if self.actions == "PER_BLOCK":
            col = layout.column(align=True)
            for item in self.names:
                col.prop(item, "name", text="")
            layout.prop(self, "asset")

    def execute(self, context):
        obj, s = context.active_object, context.active_object.sprongmotion
        if engine.model() is None:
            self.report({"ERROR"}, engine.error())
            return {"CANCELLED"}
        with engine.hold():
            s.key_density = "EXACT" if self.keys == "EXACT" else "UNBAKED"
            s.key_error = s.key_error if self.keys == "EXACT" else float(self.keys)
            engine.run_now(obj)
            if self.actions == "PER_BLOCK":
                engine.block_actions(obj, [n.name for n in self.names], self.asset)
            if obj.animation_data and obj.animation_data.action:
                for fc in retarget.channelbag(obj).fcurves:
                    keys.retype(fc.keyframe_points, keys.GENERATED, keys.KEYFRAME)
            blocks.settings(obj).blocks.clear()
        anchor.sync()
        return {"FINISHED"}


class SM_OT_export_bvh(bpy.types.Operator, ExportHelper):
    """Export the decoded timeline as BVH (the model's own skeleton, 20 fps, centimetres)"""

    bl_idname = "sm.export_bvh"
    bl_label = "Export BVH"
    filename_ext = ".bvh"
    filter_glob: StringProperty(default="*.bvh", options={"HIDDEN"})
    poll = _has_blocks

    def execute(self, context):
        try:
            obj = context.active_object
            with engine.timeline(obj, in_place=blocks.settings(obj).in_place) as (motion, _):
                text = motion.bvh() if motion else None
        except sm.Error as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        if text is None:
            self.report({"ERROR"}, engine.error() or "No block has a take from this model yet")
            return {"CANCELLED"}
        with open(self.filepath, "w", encoding="utf-8") as f:
            f.write(text)
        return {"FINISHED"}


class SM_OT_add_rig(bpy.types.Operator):
    """Add the 22-bone armature the model's motion fits without any mapping"""

    bl_idname = "sm.add_rig"
    bl_label = "SprongMotion Rig"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        m = engine.model()
        if m is None:
            self.report({"ERROR"}, engine.error())
            return {"CANCELLED"}
        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        for o in context.selected_objects:
            o.select_set(False)
        rig.build_own_rig(context, m.offsets).sprongmotion.preset = "OWN"
        return {"FINISHED"}


class SM_OT_match_bones(bpy.types.Operator):
    """Fill the bone map from the rig preset (Auto: known rigs by name, any other by its hierarchy)"""

    bl_idname = "sm.match_bones"
    bl_label = "Match Bones"
    bl_options = {"REGISTER", "UNDO"}
    poll = _has_rig

    def execute(self, context):
        rig.fill_map(context.active_object)
        n = sum(1 for b in context.active_object.sprongmotion.bones if b.bone)
        self.report({"INFO"}, f"{n} of {len(rig.JOINTS)} joints matched")
        engine.edited(context.active_object)
        return {"FINISHED"}


class SM_OT_freeze(bpy.types.Operator):
    """Keep the selected bones as they are: takes no longer key them, and their keys become your own"""

    bl_idname = "sm.freeze"
    bl_label = "Freeze Bones"
    bl_options = {"REGISTER", "UNDO"}
    unfreeze: BoolProperty(name="Unfreeze", options={"SKIP_SAVE"})
    every: BoolProperty(name="All Bones", options={"SKIP_SAVE"})
    poll = _has_rig

    def execute(self, context):
        obj = context.active_object
        names = (
            [b.name for b in obj.data.bones] if self.every else [pb.name for pb in context.selected_pose_bones or ()]
        )
        if not names:
            self.report({"WARNING"}, "Select bones in Pose Mode first")
            return {"CANCELLED"}
        why = data.freeze(obj, names, on=not self.unfreeze)
        if why:
            self.report({"ERROR"}, why)
            return {"CANCELLED"}
        if blocks.has_blocks(context) and engine.model():
            engine.run_now(obj)
        return {"FINISHED"}


class SM_OT_remove_data(bpy.types.Operator):
    """Take SprongMotion out of the action this rig plays (or the whole file): its blocks and takes go, and the rig's
    settings and Frozen bone collection unless another of its actions has blocks. The motion stays as ordinary keys
    when baked first"""

    bl_idname = "sm.remove_data"
    bl_label = "Remove SprongMotion Data"
    bl_options = {"REGISTER", "UNDO"}
    bake: BoolProperty(name="Bake First", description="Keep the motion as ordinary keys (off: it goes)", default=True)
    every: BoolProperty(name="Whole File", description="Every armature in the file", options={"SKIP_SAVE"})

    def execute(self, context):
        if not self.every and blocks.armature(context) is None:
            self.report({"WARNING"}, "Select an armature first")
            return {"CANCELLED"}
        engine.cancel()
        if self.every:
            data.remove_everywhere(self.bake)
        else:
            data.remove_data(context.active_object, self.bake)
        paths.sync()
        anchor.sync()
        return {"FINISHED"}


def _stroke_object(context):
    picked = [o for o in context.selected_objects if o.type == "GREASEPENCIL"]
    every = [o for o in context.scene.objects if o.type == "GREASEPENCIL"]
    return picked[0] if picked else every[0] if len(every) == 1 else None


class SM_OT_path_from_stroke(bpy.types.Operator):
    """Turn the last Grease Pencil stroke (of the selected Grease Pencil object, or the scene's only one) into a curve
    and make it the rig's path"""

    bl_idname = "sm.path_from_stroke"
    bl_label = "Path from Stroke"
    bl_options = {"REGISTER", "UNDO"}
    poll = _has_rig

    def execute(self, context):
        gp = _stroke_object(context)
        pts = paths.stroke_points(gp) if gp else None
        if pts is None:
            self.report({"WARNING"}, "Draw a stroke with Grease Pencil first (select it if there are several)")
            return {"CANCELLED"}
        obj = context.active_object
        blocks.settings(obj, create=True).path = paths.curve_from(pts, f"{obj.name} Path", context.collection)
        return {"FINISHED"}


class SM_OT_fit_scene(bpy.types.Operator):
    """End the scene where the last block ends"""

    bl_idname = "sm.fit_scene"
    bl_label = "Fit Scene to Blocks"
    bl_options = {"REGISTER", "UNDO"}
    poll = _has_blocks

    def execute(self, context):
        context.scene.frame_end = blocks.settings(context.active_object).blocks[-1].end
        return {"FINISHED"}


class SM_OT_preferences(bpy.types.Operator):
    """Open the add-on's preferences: the runtime library and the model file"""

    bl_idname = "sm.preferences"
    bl_label = "Open Preferences"

    def execute(self, _context):
        bpy.ops.preferences.addon_show(module=__package__)
        return {"FINISHED"}


CLASSES = (
    SM_OT_block_add, SM_OT_block_remove, SM_OT_block_move, SM_OT_block_split, SM_OT_block_copy, SM_OT_block_paste,
    SM_OT_block_duplicate, SM_OT_block_regenerate,
    SM_OT_block_regenerate_range, SM_OT_fill_gap, SM_OT_variant,
    SM_OT_generate, SM_OT_cancel, SM_OT_bake, SM_OT_export_bvh, SM_OT_add_rig, SM_OT_match_bones, SM_OT_freeze,
    SM_OT_remove_data, SM_OT_path_from_stroke, SM_OT_fit_scene, SM_OT_preferences,
)  # fmt: skip
TAKES = [
    ("sm.variant", "RIGHT_BRACKET", "PRESS", {"alt": True}, {"step": 1}),
    ("sm.variant", "LEFT_BRACKET", "PRESS", {"alt": True}, {"step": -1}),
]
KEYMAPS = [
    ("Dopesheet", "DOPESHEET_EDITOR", [*TAKES, ("sm.cancel", "ESC", "PRESS", {}, {})]),
    ("3D View", "VIEW_3D", TAKES),
    ("Pose", "EMPTY", [("sm.freeze", "F", "PRESS", {"shift": True, "alt": True}, {}),
                       ("sm.freeze", "F", "PRESS", {"ctrl": True, "alt": True}, {"unfreeze": True})]),
]  # fmt: skip


def add_keymaps(keymaps, added):
    """keymaps: [(keymap name, space, [(operator, key, value, modifiers, properties)])]; added collects the items."""
    kc = bpy.context.window_manager.keyconfigs.addon
    for km_name, space, rows in keymaps if kc else ():
        km = kc.keymaps.new(name=km_name, space_type=space)
        for idname, key, value, mods, props in rows:
            kmi = km.keymap_items.new(idname, key, value, **mods)
            for k, v in props.items():
                setattr(kmi.properties, k, v)
            added.append((km, kmi))


def remove_keymaps(added):
    for km, kmi in added:
        km.keymap_items.remove(kmi)
    added.clear()
