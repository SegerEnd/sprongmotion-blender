"""The operators that edit blocks: add, remove, move, split, copy and paste, regenerate, fill a gap, cycle takes."""

import random

import bpy
from bpy.props import BoolProperty, IntProperty, StringProperty

from . import blocks, continue_keys, generate, model, rig_data, timeline_motion


def editable(context):
    """The active object is an armature whose blocks can be changed."""
    obj = blocks.armature(context)
    return obj is not None and rig_data.why_not_editable(obj) is None


poll_blocks = classmethod(lambda cls, context: blocks.has_blocks(context) and editable(context))
poll_rig = classmethod(lambda cls, context: editable(context))


class _BlockOp:
    bl_options = {"REGISTER", "UNDO"}
    index: IntProperty(default=-1, options={"SKIP_SAVE"})
    poll = poll_blocks


class SM_OT_block_add(bpy.types.Operator):
    """Add a block with the motion text after the last one"""

    bl_idname = "sm.block_add"
    bl_label = "Add Motion"
    bl_options = {"REGISTER", "UNDO"}
    text: StringProperty(name="Text", options={"SKIP_SAVE"})
    poll = poll_rig

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
    poll = poll_blocks

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
    poll = poll_rig

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
        with generate.hold():
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
        generate.cancel()
        why = timeline_motion.remake_range(obj, i, min(f0, f1), max(f0, f1))
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
    poll = poll_rig

    def execute(self, context):
        obj = context.active_object
        span, why = self.lane_gap(obj) if self.after >= 0 else self.keys_gap(obj, context.scene.frame_current)
        if span is not None and span[1] - span[0] < round(blocks.MIN_SECONDS * blocks.fps()):
            why = f"The gap is shorter than {blocks.MIN_SECONDS:g} s"
        if why or model.load() is None:
            self.report({"WARNING"}, why or model.error())
            return {"CANCELLED"}
        if self.after < 0:
            continue_keys.clear(obj, *span)
        blocks.fill(obj, *span, self.text.strip() or "a person walks forward")
        generate.edited(obj, force=True)
        return {"FINISHED"}

    def keys_gap(self, obj, frame):
        why = continue_keys.why_not(obj, model.load())
        span = continue_keys.selected_gap(obj, frame)
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


CLASSES = (
    SM_OT_block_add, SM_OT_block_remove, SM_OT_block_move, SM_OT_block_split, SM_OT_block_copy, SM_OT_block_paste,
    SM_OT_block_duplicate, SM_OT_block_regenerate, SM_OT_block_regenerate_range, SM_OT_fill_gap, SM_OT_variant,
)  # fmt: skip
