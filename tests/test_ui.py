import bpy
import rigs
from bl_ext.smtest.sprongmotion import panels, rig
from test_rigs import build


class _Layout:
    """Records what a draw puts in the panel: (kind, what)."""

    def __init__(self, seen=None):
        self.seen = [] if seen is None else seen

    def __getattr__(self, name):
        def call(*args, **kw):
            if name == "label":
                self.seen.append(("label", kw.get("text", "")))
            elif name in {"prop", "prop_search"}:
                self.seen.append(("prop", args[1]))
            elif name == "operator":
                self.seen.append(("operator", args[0]))
            return _Layout(self.seen)

        return call

    def __iter__(self):
        return iter((_Layout(self.seen), None))


def drawn(fn=panels.draw_main):
    layout = _Layout()
    fn(layout, bpy.context)
    return layout.seen


def test_panels_in_the_dope_sheet_and_a_pointer_in_the_3d_view():
    panels = [c for c in bpy.types.Panel.__subclasses__() if c.__module__.startswith("bl_ext.smtest.sprongmotion")]
    assert [c.bl_idname for c in panels if c.bl_space_type == "VIEW_3D"] == ["SM_PT_pointer_view3d"]
    assert len([c for c in panels if c.bl_space_type == "DOPESHEET_EDITOR"]) > 1


def test_rig_line_counts_before_any_take():
    obj = build("simple16")
    obj.sprongmotion.property_unset("bones")
    n = len(rigs.expected(rigs.RIGS["simple16"]))
    assert ("label", f"Rig: matched {n} of {len(rig.JOINTS)}") in drawn(panels.draw_rig), "read from the rig"
    assert not len(obj.sprongmotion.bones), "drawing writes nothing"
