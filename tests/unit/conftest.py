"""Unit tests in plain Python: the add-on's modules load as a package without Blender (bpy stubbed, __init__ not
run), so only the logic that needs no Blender data is under test here."""

import pathlib
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
for name in ("bpy", "bpy_extras", "bpy_extras.anim_utils", "mathutils"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["bpy_extras"].anim_utils = sys.modules["bpy_extras.anim_utils"]
sys.modules["mathutils"].Matrix = None
package = types.ModuleType("sprongmotion")
package.__path__ = [str(ROOT)]
sys.modules.setdefault("sprongmotion", package)
