"""Typing a block's text in the block itself, with the typing help, or in a popup."""

import time

import bpy
from bpy.props import BoolProperty, IntProperty, StringProperty

from . import assist, blocks, model, ops

PAUSE = 0.3  # seconds without a key before the word list shows


class _Help:
    """The typing help for the text being typed: the ghost after the caret, or the word list."""

    def __init__(self, text=None):
        self.sentence, self.ghost, self.rows, self.pick, self.last = "", "", [], 0, 0.0
        self.quiet = text  # the text Esc hid the help for: nothing shows until it changes


class _State:
    def __init__(self):
        self.typing = None  # [block uid, text, caret] while typing in a block
        self.help = _Help()


_state = _State()


def in_lane(context):
    return bool(context.area and context.area.type == "DOPESHEET_EDITOR" and context.region.type == "WINDOW")


def type_key(text, caret, event, clipboard=""):
    """One key press on a single-line text field: (text, caret) after it."""
    key, command = event.type, event.ctrl or event.oskey
    moves = {"LEFT_ARROW": caret - 1, "RIGHT_ARROW": caret + 1, "HOME": 0, "END": len(text)}
    if key in moves:
        return text, max(0, min(len(text), moves[key]))
    if key == "BACK_SPACE" and caret:
        return text[: caret - 1] + text[caret:], caret - 1
    if key == "DEL":
        return text[:caret] + text[caret + 1 :], caret
    typed_text = "" if command else event.unicode
    if key == "V" and command:
        typed_text = (clipboard.splitlines() or [""])[0]
    if typed_text and typed_text.isprintable():
        return text[:caret] + typed_text + text[caret:], caret + len(typed_text)
    return text, caret


def begin(uid, text):
    """Start typing in a block: the help shows nothing until the text changes."""
    _state.typing, _state.help = [uid, text, len(text)], _Help(text)


def _help_takes(event):
    """What the typing help makes of a key: (text, caret, pick, hide), or None when it is the text field's."""
    help_, (_, text, caret) = _state.help, _state.typing
    return assist.take(assist.kind(event), text, caret, help_.ghost, help_.rows, help_.pick)


def typed(event, clipboard=""):
    """One key while typing in a block that does not end the edit."""
    help_, (_, text, caret) = _state.help, _state.typing
    took = _help_takes(event)
    if took is None:
        took = (*type_key(text, caret, event, clipboard), 0, False)
    new_text, new_caret, help_.pick, hide = took
    if hide:
        help_.quiet, help_.ghost, help_.rows = new_text, "", []
    elif (new_text, new_caret) != (text, caret):
        _state.typing[1:] = new_text, new_caret
        changed()


def help_key(event):
    """True when the typing help takes this key (Enter or Esc while it shows something)."""
    return _help_takes(event) is not None


def changed():
    """After the text or caret moved: the ghost now, the word list after a pause."""
    help_, (_, text, caret) = _state.help, _state.typing
    helper = model.helper()
    help_.rows, help_.pick, help_.last = [], 0, time.perf_counter()
    on = helper is not None and text != help_.quiet
    help_.sentence = helper.sentence(text, help_.sentence) if on and caret == len(text) else ""
    help_.ghost = help_.sentence[len(text) :]
    if on and not bpy.app.timers.is_registered(_pause):
        bpy.app.timers.register(_pause, first_interval=PAUSE)


def _pause():
    help_, helper = _state.help, model.helper()
    if _state.typing is None or helper is None:
        return None
    wait = PAUSE - (time.perf_counter() - help_.last)
    if wait > 0:
        return wait
    _, text, caret = _state.typing
    help_.rows = [] if help_.ghost or text == help_.quiet else helper.complete(text, caret)
    if help_.rows:
        blocks.redraw()
    return None


def _suggest(_op, _context, text):
    """The popup's list: the sentence the ghost would show first (a text field cannot draw it), then the words."""
    helper = model.helper()
    if helper is None:
        return []
    ghost = helper.ghost(text)
    found = [text + ghost] if ghost else []
    found += [assist.insert(text, len(text), word)[0] for word in helper.complete(text, len(text))]
    return found[: assist.ROWS]


class SM_OT_block_edit(bpy.types.Operator):
    """Edit the block's text: in the block itself or in a popup (the popup also takes input methods)"""

    bl_idname = "sm.block_edit"
    bl_label = "Edit Block Text"
    bl_options = {"REGISTER", "UNDO"}
    index: IntProperty(default=-1, options={"SKIP_SAVE"})
    popup: BoolProperty(options={"SKIP_SAVE"})
    text: StringProperty(name="Text", search=_suggest)

    @classmethod
    def poll(cls, context):
        return blocks.has_blocks(context) and ops.editable(context)

    def block(self, context):
        obj, i = blocks.pick(context, self.index)
        return blocks.settings(obj).blocks[i]

    def invoke(self, context, _event):
        b = self.block(context)
        self.text = b.text
        prefs = model.prefs()
        if self.popup or not in_lane(context) or (prefs and prefs.text_edit == "POPUP"):
            return context.window_manager.invoke_props_dialog(self, width=420, title="Motion", confirm_text="Apply")
        begin(b.uid, b.text)
        self.region = context.region
        context.window_manager.modal_handler_add(self)
        self.region.tag_redraw()
        return {"RUNNING_MODAL"}

    def draw(self, _context):
        self.layout.activate_init = True
        self.layout.prop(self, "text", text="")

    def modal(self, context, event):
        if event.value != "PRESS":
            return {"RUNNING_MODAL"}
        ends_edit = event.type in {"RET", "NUMPAD_ENTER", "ESC", "LEFTMOUSE", "RIGHTMOUSE"}
        if ends_edit and not help_key(event):
            self.text, _state.typing = _state.typing[1], None
            self.region.tag_redraw()
            if event.type == "ESC":
                return {"CANCELLED"}
            result = self.execute(context)
            return result | {"PASS_THROUGH"} if "MOUSE" in event.type else result
        typed(event, context.window_manager.clipboard)
        self.region.tag_redraw()
        return {"RUNNING_MODAL"}

    def execute(self, context):
        b = self.block(context)
        if b.text == self.text:
            return {"CANCELLED"}
        b.text = self.text
        return {"FINISHED"}


def reset():
    _state.typing = None
    if bpy.app.timers.is_registered(_pause):
        bpy.app.timers.unregister(_pause)
