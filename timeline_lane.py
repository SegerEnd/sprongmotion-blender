"""The block lane on the Timeline / Dope Sheet: hit testing and the mouse operators."""

import blf
import bpy
from bpy.props import StringProperty

from . import block_operators, block_text, blocks, generate, operators, return_to_start, timeline_motion

HEIGHT, BOTTOM, EDGE, RADIUS, PAD = 34, 40, 6, 5, 8  # pixels at UI scale 1; BOTTOM clears the scrollbar and the markers
FONT, BASE = 12, 12  # text size, and its baseline above the block's bottom
STRIP = 3  # the path strip under the blocks
DRAG_START = 4
SNAP = 8  # how close a snap target must be to snap to it


class _State:
    def __init__(self):
        self.failed = False
        self.hover = (0, None, -1)  # region pointer, hit kind, block index
        self.range = None  # (block uid, frame, frame) while Alt+dragging a range to regenerate
        self.menu_frame = self.menu_gap = 0


_state = _State()


def ui_scale():
    return bpy.context.preferences.system.ui_scale or 1.0  # 0 without a window


def rect():
    """The lane's bottom and top in region pixels."""
    return BOTTOM * ui_scale(), (BOTTOM + HEIGHT) * ui_scale()


def frame_x(region, frame):
    return region.view2d.view_to_region(frame, 0, clip=False)[0]


def x_frame(region, x):
    return region.view2d.region_to_view(x, 0)[0]


def snapped(scene, frames, reach, edges=()):
    """How far to shift frames so the nearest one lands on a snap target within reach, or None."""
    kind = scene.tool_settings.snap_anim_element
    targets = [*edges, scene.frame_current]
    if kind == "MARKER":
        targets += [m.frame for m in scene.timeline_markers]
    best = None
    for f in frames:
        candidates = targets
        if kind == "SECOND":
            fps = blocks.fps(scene)
            candidates = targets + [round(round(f / fps) * fps)]
        for target in candidates:
            offset = target - f
            if abs(offset) <= reach and (best is None or abs(offset) < abs(best)):
                best = offset
    return best


def add_chip(region, s):
    """The "+" chip right after the last block, or "+ Add motion" at the playhead on an empty lane."""
    scale = ui_scale()
    blf.size(0, FONT * scale)
    if s is not None and len(s.blocks):
        x0, label = frame_x(region, s.blocks[-1].end) + 2 * scale, "+"
    else:
        x0, label = frame_x(region, bpy.context.scene.frame_current), "+ Add motion"
    return x0, x0 + blf.dimensions(0, label)[0] + 2 * PAD * scale, label


def _has_strips(s):
    return s is not None and (s.path or any(return_to_start.walks_back(s, b) for b in s.blocks))


def hit(region, s, x, y):
    """What is under the mouse: (kind, block index), kind one of add, left, right, body, gap, strip or None."""
    y0, y1 = rect()
    if _has_strips(s) and y0 - (STRIP + 4) * ui_scale() <= y < y0:
        for i, b in enumerate(s.blocks):
            if frame_x(region, b.start) <= x < frame_x(region, b.end):
                return "strip", i
        return None, -1
    if not y0 <= y <= y1:
        return None, -1
    x0, x1, _ = add_chip(region, s)
    if x0 <= x <= x1:
        return "add", -1
    spans = [(frame_x(region, b.start), frame_x(region, b.end)) for b in (s.blocks if s else ())]
    return _hit_edge(spans, x) or _hit_body(spans, x)


def _hit_edge(spans, x):
    edge = EDGE * ui_scale()
    for i, (start, end) in enumerate(spans):
        if abs(x - end) <= edge:
            return "right", i
        touches_previous = i > 0 and start - spans[i - 1][1] <= edge  # there the edge is the left block's end
        if abs(x - start) <= edge and not touches_previous:
            return "left", i
    return None


def _hit_body(spans, x):
    for i, (start, end) in enumerate(spans):
        if start < x < end:
            return "body", i
        if i and spans[i - 1][1] < x < start:
            return "gap", i - 1
    return None, -1


class _LaneOp:
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return block_text.in_lane(context) and block_operators.editable(context)  # read-only blocks only show

    def hit(self, context, event):
        return hit(context.region, blocks.settings(context.active_object), event.mouse_region_x, event.mouse_region_y)


class SM_OT_lane_hover(_LaneOp, bpy.types.Operator):
    bl_idname = "sm.lane_hover"
    bl_label = "Block Lane Hover"

    @classmethod
    def poll(cls, context):
        return block_text.in_lane(context) and blocks.armature(context) is not None

    def invoke(self, context, event):
        state = (context.region.as_pointer(), *self.hit(context, event))
        if state != _state.hover:
            _state.hover = state
            context.region.tag_redraw()
        return {"PASS_THROUGH"}


def _snapshot(s):
    """The blocks' places and edit flags, to drag from and to restore on cancel."""
    return [(b.uid, b.start, b.end, b.locked, b.dirty, b.remake) for b in s.blocks]


def _restore_places(s, snapshot):
    """The blocks' order and places as in snapshot."""
    for k, (uid, start, end, *_) in enumerate(snapshot):
        s.blocks.move(blocks.find(s, uid), k)
        s.blocks[k].start, s.blocks[k].end = start, end


def _drag_frames(op, frame):
    """The frames that may snap while dragging, and the snap targets for them."""
    spans = [(uid, start, end) for uid, start, end, *_ in op.snapshot]
    if op.kind == "body":
        moved = [e + frame - op.f0 for uid, a, b in spans if op.shift or uid == op.uid for e in (a, b)]
        targets = [] if op.shift else [e for uid, a, b in spans if uid != op.uid for e in (a, b)]
        return moved, targets
    own = next(span for span in spans if span[0] == op.uid)
    own_edge = own[2 if op.kind == "right" else 1]
    return [frame], [e for _, a, b in spans for e in (a, b) if e != own_edge]


class SM_OT_lane_click(_LaneOp, bpy.types.Operator):
    """Select a block; drag its middle to move it (Shift: move all, Alt: pick a range to regenerate), drag an edge
    to resize, the blocks touching it moving along (Alt: they stay, Ctrl: snap the other way)"""

    bl_idname = "sm.lane_click"
    bl_label = "Edit Blocks"
    bl_options = {"REGISTER", "UNDO", "INTERNAL"}

    def invoke(self, context, event):
        kind, i = self.hit(context, event)
        obj = context.active_object
        if kind in {None, "strip", "gap"}:
            return {"PASS_THROUGH"}
        if kind == "add":
            text = obj.sprongmotion.prompt
            blocks.add(obj, text)
            if not text.strip():
                bpy.ops.sm.block_edit("INVOKE_DEFAULT", index=len(blocks.settings(obj).blocks) - 1)
            return {"FINISHED"}
        s = blocks.settings(obj)
        s.active = i
        self.kind = "range" if kind == "body" and event.alt else kind
        self.uid, self.shift, self.moved = s.blocks[i].uid, event.shift, False
        self.before = blocks.continues(s)
        self.x0, self.f0 = event.mouse_region_x, x_frame(context.region, event.mouse_region_x)
        self.snapshot, self.region = _snapshot(s), context.region
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        obj = context.active_object
        if event.type == "MOUSEMOVE":
            self.moved |= abs(event.mouse_region_x - self.x0) >= DRAG_START * ui_scale()
            if self.moved:
                self.drag(obj, event.mouse_region_x, event.ctrl, event.alt)
                self.region.tag_redraw()
        elif event.type == "LEFTMOUSE" and event.value == "RELEASE":
            if not self.moved:
                return {"CANCELLED"}
            self.done(obj)
            return {"FINISHED"}
        elif event.type in {"ESC", "RIGHTMOUSE"}:
            self.restore(obj)
            self.region.tag_redraw()
            return {"CANCELLED"}
        return {"RUNNING_MODAL"}

    def drag(self, obj, x, ctrl=False, alt=False):
        """From the blocks as they were at the click: an edge to the mouse, or a block (Shift: all) moved."""
        s, scene, frame = blocks.settings(obj), bpy.context.scene, x_frame(self.region, x)
        _restore_places(s, self.snapshot)
        if self.kind == "range":
            _state.range = (self.uid, round(self.f0), round(frame))
            return
        i = blocks.find(s, self.uid)
        snap = None
        if scene.tool_settings.use_snap_anim != ctrl:  # Ctrl snaps the other way
            pixels_per_frame = frame_x(self.region, 1) - frame_x(self.region, 0)
            moving, targets = _drag_frames(self, frame)
            snap = snapped(scene, moving, SNAP * ui_scale() / pixels_per_frame, targets)
        if snap is None:
            to, by = round(frame), round(frame) - round(self.f0)
        else:
            to, by = round(frame + snap), round(frame - self.f0 + snap)
        if self.kind == "right":
            blocks.resize(obj, i, end=to, follow=not alt)  # Alt: the others stay
        elif self.kind == "left":
            blocks.resize(obj, i, start=to, follow=not alt)
        elif self.shift:
            blocks.shift(s, range(len(s.blocks)), by)
        else:
            s.active = blocks.place(s, i, s.blocks[i].start + by)

    def done(self, obj):
        s = blocks.settings(obj)
        if self.kind == "range":
            picked, _state.range = _state.range, None
            why = timeline_motion.remake_range(obj, blocks.find(s, self.uid), *sorted(picked[1:]))
            if why:
                self.report({"WARNING"}, why)
        elif self.shift and self.kind == "body":
            if len(s.blocks) == 1:
                blocks.snap(obj, 0, (s.blocks[0].start + s.blocks[0].end) // 2)
            generate.edited(obj)
        elif self.kind == "body":  # a block that now goes on from another one (or none) is made again
            now = blocks.continues(s)
            blocks.mark_dirty(obj, *(uid for uid, prev in now.items() if self.before.get(uid) != prev))
        else:
            blocks.mark_dirty(obj, self.uid)

    def restore(self, obj):
        s, _state.range = blocks.settings(obj), None
        _restore_places(s, self.snapshot)
        for b, (_, _, _, locked, dirty, remake) in zip(s.blocks, self.snapshot, strict=True):
            b.dirty, b.remake = dirty, remake
            if b.locked != locked:
                b["locked"] = locked  # an ID write: the RNA update would refit


class SM_OT_lane_open(_LaneOp, bpy.types.Operator):
    bl_idname = "sm.lane_open"
    bl_label = "Edit Block Text"

    def invoke(self, context, event):
        kind, i = self.hit(context, event)
        if kind != "body":
            return {"PASS_THROUGH"}
        bpy.ops.sm.block_edit("INVOKE_DEFAULT", index=i)
        return {"FINISHED"}


class SM_OT_lane_menu(_LaneOp, bpy.types.Operator):
    bl_idname = "sm.lane_menu"
    bl_label = "Block Menu"

    def invoke(self, context, event):
        kind, i = self.hit(context, event)
        if kind not in {"body", "gap"}:
            return {"PASS_THROUGH"}
        if kind == "body":
            blocks.settings(context.active_object).active = i
        _state.menu_frame, _state.menu_gap = round(x_frame(context.region, event.mouse_region_x)), i
        bpy.ops.wm.call_menu(name=(SM_MT_block if kind == "body" else SM_MT_gap).__name__)
        return {"FINISHED"}


class SM_OT_lane_delete(_LaneOp, bpy.types.Operator):
    bl_idname = "sm.lane_delete"
    bl_label = "Delete Block"
    bl_options = {"REGISTER", "UNDO", "INTERNAL"}

    def invoke(self, context, event):
        kind, i = self.hit(context, event)
        if kind in {None, "add", "strip", "gap"}:
            return {"PASS_THROUGH"}
        blocks.remove(context.active_object, i)
        return {"FINISHED"}


class SM_OT_lane_key(_LaneOp, bpy.types.Operator):
    """Copy, paste or duplicate the selected block with the mouse over the lane"""

    bl_idname = "sm.lane_key"
    bl_label = "Block Key"
    action: StringProperty()

    def invoke(self, context, event):
        y0, y1 = rect()
        over_lane = y0 <= event.mouse_region_y <= y1
        if not over_lane or (self.action != "paste" and not blocks.has_blocks(context)):
            return {"PASS_THROUGH"}
        return getattr(bpy.ops.sm, f"block_{self.action}")()


class SM_MT_block(bpy.types.Menu):
    bl_label = "Block"

    def draw(self, context):
        s = blocks.settings(context.active_object)
        layout, b = self.layout, s.blocks[s.active]
        layout.operator("sm.block_edit", text="Edit Text", icon="GREASEPENCIL")
        layout.operator("sm.block_edit", text="Edit in Popup…").popup = True
        layout.operator("sm.block_split", text="Split Here", icon="CENTER_ONLY").frame = _state.menu_frame
        layout.operator("sm.block_copy", text="Copy", icon="COPYDOWN")
        layout.operator("sm.block_paste", text="Paste", icon="PASTEDOWN")
        layout.operator("sm.block_duplicate", text="Duplicate", icon="DUPLICATE")
        layout.separator()
        layout.operator("sm.block_regenerate", text="Regenerate", icon="FILE_REFRESH")
        layout.operator("sm.block_regenerate", text="New Seed", icon="MOD_NOISE").new_seed = True
        layout.operator("sm.block_regenerate_range", text="Regenerate Preview Range")
        layout.prop(b, "locked", text="Lock")
        layout.prop(b, "fresh")
        layout.prop(b, "in_place")
        layout.prop(b, "back_to_start")
        if b.variants > 1:
            layout.operator("sm.variant", text="Next Take", icon="TRIA_RIGHT").step = 1
            layout.operator("sm.variant", text="Previous Take", icon="TRIA_LEFT").step = -1
        layout.separator()
        layout.operator("sm.block_remove", text="Delete", icon="X")


class SM_MT_gap(bpy.types.Menu):
    bl_label = "Gap"

    def draw(self, _context):
        self.layout.operator("sm.fill_gap", icon="KEYFRAME").after = _state.menu_gap


CLASSES = (block_text.SM_OT_block_edit, SM_OT_lane_hover, SM_OT_lane_click, SM_OT_lane_open, SM_OT_lane_menu,
           SM_OT_lane_delete, SM_OT_lane_key, SM_MT_block, SM_MT_gap)  # fmt: skip
KEYS = [  # Dope Sheet / Timeline
    ("sm.lane_hover", "MOUSEMOVE", "ANY", {}, {}),
    ("sm.lane_click", "LEFTMOUSE", "PRESS", {}, {}),
    ("sm.lane_click", "LEFTMOUSE", "PRESS", {"shift": True}, {}),
    ("sm.lane_click", "LEFTMOUSE", "PRESS", {"ctrl": True}, {}),
    ("sm.lane_click", "LEFTMOUSE", "PRESS", {"shift": True, "ctrl": True}, {}),
    ("sm.lane_click", "LEFTMOUSE", "PRESS", {"alt": True}, {}),
    ("sm.lane_open", "LEFTMOUSE", "DOUBLE_CLICK", {}, {}),
    ("sm.lane_menu", "RIGHTMOUSE", "PRESS", {}, {}),
    ("sm.lane_delete", "X", "PRESS", {}, {}),
    ("sm.lane_delete", "DEL", "PRESS", {}, {}),
    ("sm.lane_key", "C", "PRESS", {"ctrl": True}, {"action": "copy"}),
    ("sm.lane_key", "C", "PRESS", {"oskey": True}, {"action": "copy"}),  # Cmd on macOS, as Blender's own copy
    ("sm.lane_key", "V", "PRESS", {"ctrl": True}, {"action": "paste"}),
    ("sm.lane_key", "V", "PRESS", {"oskey": True}, {"action": "paste"}),
    ("sm.lane_key", "D", "PRESS", {"shift": True}, {"action": "duplicate"}),
]

_keymaps = []


def reset():
    _state.failed = False
    block_text.reset()


def register():
    global _state
    _state = _State()
    for c in CLASSES:
        bpy.utils.register_class(c)
    operators.add_keymaps([("Dopesheet", "DOPESHEET_EDITOR", KEYS)], _keymaps)


def unregister():
    global _state
    reset()
    _state = _State()
    operators.remove_keymaps(_keymaps)
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
