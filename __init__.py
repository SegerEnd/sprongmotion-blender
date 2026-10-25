import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, PointerProperty, StringProperty

from . import continue_keys, generate, model, operators, panels, paths, properties, timeline_draw, timeline_lane


def _reload(_prefs, _context):
    model.load()


def _after_block(_prefs, _context):
    generate.reapply()


class SMPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    library_path: StringProperty(
        name="Library",
        subtype="FILE_PATH",
        update=_reload,
        description="The runtime: libsprongmotion (.dylib, .so, .dll). Empty: the one inside the add-on",
    )
    model_path: StringProperty(
        name="Model",
        subtype="FILE_PATH",
        update=_reload,
        description="A model file (.gguf). Empty: the one inside the add-on",
    )
    text_edit: EnumProperty(
        name="Edit Block Text",
        description="What a double-click on a block opens",
        items=[
            ("IN_PLACE", "In the Block", "Type straight into the block"),
            ("POPUP", "Popup", "A text field in a popup, which also takes input methods"),
        ],
    )

    typing_help: BoolProperty(
        name="Typing Help",
        default=True,
        description="While typing a block's text: the rest of a sentence in grey (Tab takes it) and known words to "
        "finish the word you type",
    )

    after_block: EnumProperty(
        name="After a Block",
        description="What the rig does where no block and none of your keys follow",
        items=[
            (
                "HOLD",
                "Hold Last Pose",
                "The last pose of the block stays (in a gap: until it eases into the next block)",
            ),
            ("REST", "Ease to Rest", "The bones the add-on animates ease back to their rest pose and stay there"),
        ],
        update=_after_block,
    )
    ease_seconds: FloatProperty(
        name="Ease Time",
        description="How long Ease to Rest takes, and the ease out of a held pose into a block after a gap",
        default=0.4,
        min=0.1,
        max=2.0,
        unit="TIME_ABSOLUTE",
        update=_after_block,
    )

    def draw(self, _context):
        col = self.layout.column()
        col.prop(self, "library_path")
        col.prop(self, "model_path")
        m = model.loaded()
        col.label(text=f"{m.id}, blocks up to {m.max_seconds:g} s" if m else model.error(),
                  icon="CHECKMARK" if m else "ERROR")  # fmt: skip
        col.prop(self, "text_edit")
        col.prop(self, "typing_help")
        col.prop(self, "after_block")
        col.prop(self, "ease_seconds")


@bpy.app.handlers.persistent
def _cancel(*_args):
    generate.cancel()
    timeline_lane.reset()


@bpy.app.handlers.persistent
def _on_file(*_args):
    _upgrade()
    paths.on_file()
    continue_keys.on_file()


@bpy.app.handlers.persistent
def _undone(*_args):
    generate.settle(redo=False)


@bpy.app.handlers.persistent
def _redone(*_args):
    generate.settle(redo=True)


HANDLERS = (bpy.app.handlers.load_pre, bpy.app.handlers.undo_post, bpy.app.handlers.redo_post)
FILE_HANDLERS = (bpy.app.handlers.load_post, bpy.app.handlers.undo_post, bpy.app.handlers.redo_post)
CLASSES = (
    SMPreferences,
    properties.SMBlock,
    properties.SMBone,
    properties.SMSettings,
    properties.SMTimeline,
    *operators.CLASSES,
    *panels.CLASSES,
)
_keymaps = []


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    bpy.types.Object.sprongmotion = PointerProperty(type=properties.SMSettings, override={"LIBRARY_OVERRIDABLE"})
    bpy.types.Action.sprongmotion = PointerProperty(type=properties.SMTimeline)
    for h in HANDLERS:
        h.append(_cancel)
    for h in FILE_HANDLERS:
        h.append(_on_file)
    bpy.app.handlers.undo_post.append(_undone)
    bpy.app.handlers.redo_post.append(_redone)
    operators.add_keymaps(operators.KEYMAPS, _keymaps)
    panels.register_menus()
    timeline_lane.register()
    timeline_draw.register()
    bpy.app.timers.register(_load, first_interval=0.1)


def _upgrade():
    for obj in bpy.data.objects:
        if obj.type == "ARMATURE" and obj.library is None:
            properties.upgrade(obj)


def _load():
    model.load()  # at startup, outside any draw
    _upgrade()
    paths.on_file()
    continue_keys.on_file()


def unregister():
    if bpy.app.timers.is_registered(_load):
        bpy.app.timers.unregister(_load)
    timeline_draw.unregister()
    timeline_lane.unregister()
    panels.unregister_menus()
    operators.remove_keymaps(_keymaps)
    for h in HANDLERS:
        if _cancel in h:
            h.remove(_cancel)
    for h in FILE_HANDLERS:
        if _on_file in h:
            h.remove(_on_file)
    for h, fn in ((bpy.app.handlers.undo_post, _undone), (bpy.app.handlers.redo_post, _redone)):
        if fn in h:
            h.remove(fn)
    paths.unregister()
    continue_keys.unregister()
    generate.shutdown()
    del bpy.types.Object.sprongmotion
    del bpy.types.Action.sprongmotion
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
