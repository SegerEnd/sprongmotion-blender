"""The other operators (generate, bake, export, rig, freeze, remove data, path) and the keymaps."""

import bpy
from bpy.props import BoolProperty, CollectionProperty, EnumProperty, StringProperty
from bpy_extras.io_utils import ExportHelper

from . import (
    block_operators,
    blocks,
    continue_keys,
    generate,
    keyframes,
    model,
    paths,
    retarget,
    rig,
    rig_data,
    runtime,
    takes,
    timeline_motion,
)
from .block_operators import poll_blocks, poll_rig


class SM_OT_generate(bpy.types.Operator):
    """Make takes for the blocks that need one and put the motion on the armature"""

    bl_idname = "sm.generate"
    bl_label = "Generate"
    bl_options = {"REGISTER", "UNDO"}
    poll = poll_blocks

    def execute(self, context):
        if model.load() is None:
            self.report({"ERROR"}, model.error())
            return {"CANCELLED"}
        generate.edited(context.active_object, force=True)
        return {"FINISHED"}


class SM_OT_cancel(bpy.types.Operator):
    """Stop generating; unfinished blocks stay marked"""

    bl_idname = "sm.cancel"
    bl_label = "Cancel"
    poll = classmethod(lambda cls, _context: generate.busy())

    def execute(self, _context):
        generate.cancel()
        blocks.redraw()
        return {"FINISHED"}


class SM_OT_bake(bpy.types.Operator):
    """Keep the motion as ordinary keys and remove the blocks; optionally also one action per block, for game clips"""

    bl_idname = "sm.bake"
    bl_label = "Bake to Keys"
    bl_options = {"REGISTER", "UNDO"}
    poll = poll_blocks
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
        for name in takes.action_names(blocks.settings(context.active_object)):
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
        if model.load() is None:
            self.report({"ERROR"}, model.error())
            return {"CANCELLED"}
        with generate.hold():
            s.key_density = "EXACT" if self.keys == "EXACT" else "UNBAKED"
            s.key_error = s.key_error if self.keys == "EXACT" else float(self.keys)
            generate.run_now(obj)
            if self.actions == "PER_BLOCK":
                takes.block_actions(obj, [n.name for n in self.names], self.asset)
            if obj.animation_data and obj.animation_data.action:
                for fc in retarget.channelbag(obj).fcurves:
                    keyframes.retype(fc.keyframe_points, keyframes.GENERATED, keyframes.KEYFRAME)
            blocks.settings(obj).blocks.clear()
        continue_keys.sync()
        return {"FINISHED"}


class SM_OT_export_bvh(bpy.types.Operator, ExportHelper):
    """Export the decoded timeline as BVH (the model's own skeleton, 20 fps, centimetres)"""

    bl_idname = "sm.export_bvh"
    bl_label = "Export BVH"
    filename_ext = ".bvh"
    filter_glob: StringProperty(default="*.bvh", options={"HIDDEN"})
    poll = poll_blocks

    def execute(self, context):
        try:
            obj = context.active_object
            with timeline_motion.whole(obj, in_place=blocks.settings(obj).in_place) as (motion, _):
                text = motion.bvh() if motion else None
        except runtime.Error as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        if text is None:
            self.report({"ERROR"}, model.error() or "No block has a take from this model yet")
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
        m = model.load()
        if m is None:
            self.report({"ERROR"}, model.error())
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
    poll = poll_rig

    def execute(self, context):
        rig.fill_map(context.active_object)
        n = sum(1 for b in context.active_object.sprongmotion.bones if b.bone)
        self.report({"INFO"}, f"{n} of {len(rig.JOINTS)} joints matched")
        generate.edited(context.active_object)
        return {"FINISHED"}


class SM_OT_freeze(bpy.types.Operator):
    """Keep the selected bones as they are: takes no longer key them, and their keys become your own"""

    bl_idname = "sm.freeze"
    bl_label = "Freeze Bones"
    bl_options = {"REGISTER", "UNDO"}
    unfreeze: BoolProperty(name="Unfreeze", options={"SKIP_SAVE"})
    every: BoolProperty(name="All Bones", options={"SKIP_SAVE"})
    poll = poll_rig

    def execute(self, context):
        obj = context.active_object
        names = (
            [b.name for b in obj.data.bones] if self.every else [pb.name for pb in context.selected_pose_bones or ()]
        )
        if not names:
            self.report({"WARNING"}, "Select bones in Pose Mode first")
            return {"CANCELLED"}
        why = rig_data.freeze(obj, names, on=not self.unfreeze)
        if why:
            self.report({"ERROR"}, why)
            return {"CANCELLED"}
        if blocks.has_blocks(context) and model.load():
            generate.run_now(obj)
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
        generate.cancel()
        if self.every:
            rig_data.remove_everywhere(self.bake)
        else:
            rig_data.remove_data(context.active_object, self.bake)
        paths.sync()
        continue_keys.sync()
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
    poll = poll_rig

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
    poll = poll_blocks

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
    *block_operators.CLASSES,
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
