import textwrap
import traceback

import bpy

from . import anchor, blocks, data, engine, paths, rig

SPACES = ("DOPESHEET_EDITOR",)  # next to the blocks; the 3D view only points there
_matches = {}


class SM_UL_blocks(bpy.types.UIList):
    def draw_item(self, _context, layout, _data, item, _icon, _active_data, _active_prop, _index=0, _flt_flag=0):
        row = layout.row(align=True)
        row.prop(item, "text", text="", emboss=False, placeholder="Type a motion")
        if item.locked:
            row.prop(item, "locked", text="", icon="LOCKED", emboss=False)
        row.prop(item, "enabled", text="", icon="HIDE_OFF" if item.enabled else "HIDE_ON", emboss=False)


def draw_main(layout, context):
    m, obj = engine.loaded(), blocks.armature(context)
    if engine.error():
        box = layout.box().column(align=True)
        for k, line in enumerate(textwrap.wrap(engine.error(), 38)):
            box.label(text=line, icon="ERROR" if k == 0 else "BLANK1")
        if m is None:
            box.operator("sm.preferences", icon="PREFERENCES")
    if obj is None:
        layout.label(text="Select an armature, or add one:", icon="INFO")
        layout.operator("sm.add_rig", icon="OUTLINER_OB_ARMATURE", text="Add SprongMotion Rig")
        return
    if why := data.why_not_editable(obj):
        layout.label(text=why, icon="LIBRARY_DATA_DIRECT")
        return
    r, s = obj.sprongmotion, blocks.settings(obj)
    row = layout.row(align=True)
    row.prop(r, "prompt", text="", placeholder="a person walks forward")
    row.operator("sm.block_add", text="Add Motion", icon="ADD")
    if s is None or not len(s.blocks):
        return
    row = layout.row()
    row.template_list("SM_UL_blocks", "", s, "blocks", s, "active", rows=3)
    col = row.column(align=True)
    col.operator("sm.block_remove", text="", icon="REMOVE")
    col.operator("sm.block_move", text="", icon="TRIA_UP").step = -1
    col.operator("sm.block_move", text="", icon="TRIA_DOWN").step = 1
    if s.blocks[-1].end > context.scene.frame_end:
        row = layout.row()
        row.label(text="Blocks run past the scene end", icon="INFO")
        row.operator("sm.fit_scene", text="Fit")
    if 0 <= s.active < len(s.blocks):
        _draw_active(layout, obj, s.blocks[s.active])
    _draw_frozen(layout, context)
    _draw_generate(layout, obj, r, s)


def _wrapped(layout, text, icon):
    for k, line in enumerate(textwrap.wrap(text, 38)):
        layout.label(text=line, icon=icon if k == 0 else "BLANK1")


def _draw_joins(box, obj, b):
    """Continue from Keys on the timeline's first and last block, and why it does not apply when it cannot."""
    line = [x.uid for x in blocks.settings(obj).blocks if x.enabled and x.text.strip()]
    if b.uid not in line[:1] + line[-1:]:
        return
    row = box.row()
    row.prop(b, "continues")
    why = anchor.why_not(obj, engine.loaded())
    row.active = why is None
    before, _, after = b.anchor_key.partition("|")
    if b.continues and why:
        _wrapped(box, why, "INFO")
    elif b.continues and (before or after):
        joined = " and ".join(w for w, on in (("the keys before", before), ("the keys after", after)) if on)
        _wrapped(box, f"Joins {joined}", "LINKED")
    if b.seam_note:
        _wrapped(box, b.seam_note, "ERROR")


def _draw_active(layout, obj, b):
    box = layout.box()
    box.prop(b, "text", text="")
    words = engine.unknown_words(b.text, look=False)
    near = {w: engine.nearest(w) for w in words} if engine.helper() else {}
    note = ", ".join(f"{w} (nearest: {', '.join(near[w])})" if near.get(w) else w for w in words)
    for k, line in enumerate(textwrap.wrap("Not understood: " + note, 38) if words else []):
        box.label(text=line, icon="INFO" if k == 0 else "BLANK1")
    for k, line in enumerate(textwrap.wrap(b.path_note, 38)):
        box.label(text=line, icon="CURVE_PATH" if k == 0 else "BLANK1")
    row = box.row(align=True)
    row.prop(b, "seconds")
    row.prop(b, "color", text="")
    row = box.row(align=True)
    row.prop(b, "seed")
    row.operator("sm.block_regenerate", text="", icon="MOD_NOISE").new_seed = True
    row.prop(b, "locked", text="", icon="LOCKED" if b.locked else "UNLOCKED")
    if b.variants > 1:
        row = box.row(align=True)
        row.operator("sm.variant", text="", icon="TRIA_LEFT").step = -1
        row.label(text=f"Take {b.variant + 1} of {b.variants}")
        row.operator("sm.variant", text="", icon="TRIA_RIGHT").step = 1
    _draw_joins(box, obj, b)
    row = box.row()
    row.prop(b, "in_place")
    row = box.row()
    row.prop(b, "back_to_start")
    why = paths.why_not_back(blocks.settings(obj), b)
    row.active = why is None
    if b.back_to_start and why:
        _wrapped(box, why, "INFO")
    row = box.row()
    row.prop(b, "fresh")
    row.active = not b.fresh or b.uid in engine.blend_uids(blocks.settings(obj))  # grey: a run's first block, or Loop
    row = box.row(align=True)
    row.operator("sm.block_regenerate", icon="FILE_REFRESH")
    row.operator("sm.block_split", text="Split", icon="CENTER_ONLY")


def _draw_frozen(layout, context):
    row = layout.row(align=True)
    row.enabled = context.mode == "POSE"
    row.operator("sm.freeze", text="Freeze", icon="FREEZE")
    row.operator("sm.freeze", text="Unfreeze", icon="LIGHT_SUN").unfreeze = True
    frozen = sorted(data.frozen(context.active_object))
    if frozen:
        box = layout.box().column(align=True)
        names = ", ".join(frozen)
        for k, line in enumerate(textwrap.wrap(f"Frozen: {names}", 38)):
            box.label(text=line, icon="FREEZE" if k == 0 else "BLANK1")
        op = box.operator("sm.freeze", text="Unfreeze All", icon="LIGHT_SUN")
        op.unfreeze, op.every = True, True


def _draw_generate(layout, obj, r, s):
    col = layout.column(align=True)
    row = col.row(align=True)
    row.prop(r, "auto", toggle=True, icon="AUTO")
    if engine.busy(obj):
        run = engine.progress()
        row.operator("sm.cancel", text=f"Cancel {run.done}/{run.total}" if run else "Cancel", icon="CANCEL")
    elif not r.auto:
        row.operator("sm.generate", icon="PLAY")
    row = col.row(align=True)
    row.prop(r, "variants")
    row = layout.row()
    row.prop(r, "foot_lock")
    row.prop(s, "in_place")
    row.prop(s, "loop")
    row = layout.row()
    row.prop(r, "hand_lock")
    row.prop(r, "floor_clamp")
    row = layout.row(align=True)
    row.prop(s, "path", icon="CURVE_PATH")
    row.operator("sm.path_from_stroke", text="", icon="OUTLINER_OB_GREASEPENCIL")
    row = layout.row(align=True)
    row.prop(r, "key_density", text="")
    if r.key_density == "UNBAKED":
        row.prop(r, "key_frames", text="")


def matched(obj):
    """The joints the rig's bones match: the stored map, else (before the first take) a match read from the rig."""
    s = obj.sprongmotion
    if len(s.bones):
        return set(rig.bone_map(obj))
    key = (obj.data.session_uid, s.preset, len(obj.data.bones))
    if key not in _matches:
        found = rig.auto_match(obj.data) if s.preset == "AUTO" else rig.match_preset(obj.data, s.preset)
        _matches[key] = {rig.JOINTS.index(j) for j in found}
    return _matches[key]


def draw_rig(layout, context):
    obj = blocks.armature(context)
    s = obj.sprongmotion
    row = layout.row(align=True)
    row.prop(s, "preset", text="")
    row.operator("sm.match_bones", text="", icon="FILE_REFRESH")
    found = matched(obj)
    ok = {0, 1, 2, 4, 5, 7, 8} <= found  # hips, knees, ankles
    layout.label(text=f"Rig: matched {len(found)} of {len(rig.JOINTS)}", icon="CHECKMARK" if ok else "ERROR")
    if any(name in obj.pose.bones for name in rig.RIGIFY_SWITCHES):
        layout.label(text="Takes set the Rigify arms and legs to FK, with full follow", icon="INFO")
    col = layout.column(align=True)
    for item in s.bones:
        row = col.row(align=True)
        row.label(text=item.joint)
        row.prop_search(item, "bone", obj.data, "bones", text="")


def draw_output(layout, _context):
    col = layout.column(align=True)
    col.operator("sm.bake", icon="KEYFRAME_HLT")
    col.operator("sm.export_bvh", icon="EXPORT")
    col.separator()
    col.operator("sm.remove_data", text="Clear Motion", icon="X").bake = False
    col.operator("sm.remove_data", icon="TRASH")
    col.operator("sm.remove_data", text="Remove from Whole File", icon="TRASH").every = True


_PANELS = [
    ("main", "SprongMotion", draw_main, set()),
    ("rig", "Rig", draw_rig, {"DEFAULT_CLOSED"}),
    ("output", "Output", draw_output, {"DEFAULT_CLOSED"}),
]


_reported = set()


def _safe(fn, layout, context):
    try:
        fn(layout, context)
    except Exception as e:  # shown in the panel; the traceback goes to the console once, not on every redraw
        if str(e) not in _reported:
            _reported.add(str(e))
            traceback.print_exc()
        layout.label(text=f"SprongMotion: {e}", icon="ERROR")


def _panel(key, label, fn, options, space):
    suffix = "view3d" if space == "VIEW_3D" else "dopesheet"
    attrs = {
        "bl_idname": f"SM_PT_{key}_{suffix}",
        "bl_label": label,
        "bl_space_type": space,
        "bl_region_type": "UI",
        "bl_category": "SprongMotion",
        "bl_options": options,
        "draw": lambda self, context: _safe(fn, self.layout, context),
    }
    if key != "main":
        attrs["poll"] = classmethod(lambda cls, context: blocks.armature(context) is not None)
    return type(attrs["bl_idname"], (bpy.types.Panel,), attrs)


def draw_pointer(layout, _context):
    layout.label(text="SprongMotion lives in the Dope Sheet / Timeline sidebar (N)", icon="ACTION")


CLASSES = (
    SM_UL_blocks,
    *[_panel(*p, space) for space in SPACES for p in _PANELS],
    _panel("pointer", "SprongMotion", draw_pointer, set(), "VIEW_3D"),
)


def _add_menu(self, _context):
    self.layout.operator("sm.add_rig", icon="OUTLINER_OB_ARMATURE")


def _pose_menu(self, context):
    if blocks.armature(context) is not None:
        self.layout.separator()
        self.layout.operator("sm.freeze", icon="FREEZE")
        self.layout.operator("sm.freeze", text="Unfreeze Bones", icon="LIGHT_SUN").unfreeze = True


class SM_MT_keys(bpy.types.Menu):
    bl_label = "SprongMotion"

    def draw(self, _context):
        self.layout.operator("sm.fill_gap", icon="KEYFRAME")


def _keys_menu(self, context):
    if blocks.armature(context) is not None:
        self.layout.separator()
        self.layout.menu(SM_MT_keys.__name__)


MENUS = ((bpy.types.VIEW3D_MT_armature_add, _add_menu), (bpy.types.VIEW3D_MT_pose_context_menu, _pose_menu),
         (bpy.types.VIEW3D_MT_pose, _pose_menu))  # fmt: skip
KEY_MENUS = ((bpy.types.DOPESHEET_MT_key, _keys_menu), (bpy.types.DOPESHEET_MT_context_menu, _keys_menu))


def register_menus():
    bpy.utils.register_class(SM_MT_keys)
    for menu, fn in MENUS + KEY_MENUS:
        menu.append(fn)


def unregister_menus():
    for menu, fn in MENUS + KEY_MENUS:
        menu.remove(fn)
    bpy.utils.unregister_class(SM_MT_keys)
