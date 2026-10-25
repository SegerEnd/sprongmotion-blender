"""Run one test file inside Blender: blender -b --factory-startup --python tests/run.py -- tests/test_x.py"""

import importlib.util
import os
import sys
import tempfile
import traceback

import addon_utils
import bpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT = {"darwin": "dylib", "win32": "dll"}.get(sys.platform, "so")
RUNTIME_LIB = os.path.join(ROOT, "..", "sprongmotion-runtime", "build", f"libsprongmotion.{EXT}")
LIB = os.path.abspath(os.path.expanduser(os.environ.get("SM_LIB", RUNTIME_LIB)))
MODEL = os.path.expanduser(os.environ.get("SM_MODEL", ""))


def check():
    for name, path in (("SM_LIB", LIB), ("SM_MODEL", MODEL)):
        if not os.path.isfile(path):
            print(f"FAIL {name}: no file at {path}. Set {name} to the runtime library / model file (see README).")
            sys.exit(1)


def enable():
    repo = tempfile.mkdtemp()
    os.symlink(ROOT, os.path.join(repo, "sprongmotion"))
    bpy.context.preferences.extensions.repos.new(name="smtest", module="smtest", custom_directory=repo, source="USER")
    if addon_utils.enable("bl_ext.smtest.sprongmotion", default_set=True) is None:
        raise RuntimeError("the add-on did not enable: its traceback is above")
    prefs = bpy.context.preferences.addons["bl_ext.smtest.sprongmotion"].preferences
    prefs.library_path, prefs.model_path = LIB, MODEL
    model = sys.modules["bl_ext.smtest.sprongmotion.model"]
    model.THREADS = int(os.environ.get("SM_THREADS", "0"))  # make runs the test files side by side
    model.unload()


def report(head):
    print(head)
    print("> " + traceback.format_exc().strip().replace("\n", "\n> "))  # make shows the lines starting with "> "


def load(path):
    check()
    enable()
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    spec = importlib.util.spec_from_file_location(os.path.basename(path)[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(path):
    try:
        mod = load(path)
    except Exception:
        report(f"FAIL {path}: the add-on or the test file did not load")
        sys.exit(1)
    failed = 0
    for name in [n for n in vars(mod) if n.startswith("test_")]:
        try:
            bpy.ops.wm.read_homefile(use_empty=True)  # each test starts from an empty file
            getattr(mod, name)()
            print(f"PASS {mod.__name__}.{name}")
        except Exception:
            failed += 1
            report(f"FAIL {mod.__name__}.{name}")
    addon_utils.disable("bl_ext.smtest.sprongmotion")
    print(f"ok {mod.__name__}" if not failed else f"FAIL {mod.__name__}: {failed} failed")
    sys.exit(1 if failed else 0)


main(sys.argv[sys.argv.index("--") + 1])
