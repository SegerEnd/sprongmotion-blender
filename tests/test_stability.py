import ctypes as C
import os
import subprocess
import sys

import addon_utils
import bpy
from bl_ext.smtest.sprongmotion import runtime

PKG = "bl_ext.smtest.sprongmotion"
SLOW = os.environ.get("SM_SLOW") == "1"  # make test-slow: the full stress run


def leftovers():
    kc = bpy.context.window_manager.keyconfigs.addon
    found = [n for n in ("SM_OT_block_add", "SM_OT_lane_click", "SM_PT_main", "SM_PT_more") if hasattr(bpy.types, n)]
    found += [f"{t}.sprongmotion" for t in ("Object", "Action") if hasattr(getattr(bpy.types, t), "sprongmotion")]
    found += [
        h.__module__
        for hs in (bpy.app.handlers.load_pre, bpy.app.handlers.undo_post)
        for h in hs
        if "smtest" in h.__module__
    ]
    found += [k.idname for km in (kc.keymaps if kc else []) for k in km.keymap_items if k.idname.startswith("sm.")]
    return found


def test_enable_disable_20_times_also_mid_take():
    errors = []
    for i in range(20 if SLOW else 2):
        mod = sys.modules[PKG]
        if i % 2:
            bpy.ops.sm.add_rig()
            obj = bpy.context.active_object
            obj.sprongmotion.variants = 2
            sys.modules[PKG + ".blocks"].add(obj, "a person walks forward", 2)
            sys.modules[PKG + ".generate"]._state.due[obj.session_uid] = 0
            sys.modules[PKG + ".generate"]._tick()
            assert sys.modules[PKG + ".generate"].busy()
        generator, drawing = sys.modules[PKG + ".generate"], sys.modules[PKG + ".timeline_draw"]
        loader = sys.modules[PKG + ".model"]
        addon_utils.disable(PKG, handle_error=errors.append)
        assert not leftovers(), leftovers()
        assert loader._state.model is None and generator._state.job is None
        assert not bpy.app.timers.is_registered(generator._tick)
        assert drawing._handle is None
        addon_utils.enable(PKG, default_set=True, handle_error=errors.append)
        assert mod is not None
    assert not errors, errors
    assert sys.modules[PKG + ".model"].load() is not None


def test_a_failing_draw_reports_once(capsys=None):
    ln, bl = sys.modules[PKG + ".timeline_lane"], sys.modules[PKG + ".blocks"]
    drawing = sys.modules[PKG + ".timeline_draw"]
    armature, bl.armature = bl.armature, lambda context: 1 / 0
    try:
        drawing.draw()
        assert ln._state.failed
        drawing.draw()
    finally:
        bl.armature = armature
    bpy.ops.wm.read_homefile(use_empty=True)
    assert not ln._state.failed, "a new file draws the lane again"


def rss_mb():
    return int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())])) / 1024


class Counter:
    """sm_hooks with an allocator that counts live blocks."""

    def __init__(self):
        libc = C.CDLL(None)
        libc.posix_memalign.argtypes = [C.POINTER(C.c_void_p), C.c_size_t, C.c_size_t]
        libc.free.argtypes = [C.c_void_p]
        self.live = 0

        def alloc(_user, size, align):
            p = C.c_void_p()
            if libc.posix_memalign(C.byref(p), max(align, C.sizeof(C.c_void_p)), size):
                return None
            self.live += 1
            return p.value

        def free(_user, ptr):
            if ptr:
                self.live -= 1
                libc.free(ptr)

        self.alloc = C.CFUNCTYPE(C.c_void_p, C.c_void_p, C.c_size_t, C.c_size_t)(alloc)
        self.free = C.CFUNCTYPE(None, C.c_void_p, C.c_void_p)(free)

        threads = sys.modules[PKG + ".model"].THREADS
        self.hooks = runtime.Hooks(threads, C.cast(self.alloc, C.c_void_p).value, C.cast(self.free, C.c_void_p).value)


def cycle(m, seed):
    texts, seconds = ["a person walks forward", "a person turns around"], [2.0, 1.5]
    with m.blocks(texts, seconds, [seed, seed], [None, None]) as (motion, first):
        codes = motion.codes()
    keep = [codes[: first[1] // 2], None]
    with m.blocks(texts, seconds, [seed, seed + 1], keep, foot_lock=bool(seed % 2)) as (motion, _):
        motion.rotations()


def test_300_cycles_leak_nothing():
    count = Counter()
    loader = sys.modules[PKG + ".model"]
    m = runtime.Model(loader._state.lib, loader._state.paths[1], C.byref(count.hooks))
    try:
        cycle(m, 0)
        live, rss = count.live, rss_mb()
        for seed in range(1, 301 if SLOW else 11):
            cycle(m, seed)
            assert count.live == live, f"cycle {seed}: {count.live - live} blocks still allocated"
        grown = rss_mb() - rss
        print(f"> {seed} cycles: live allocations {live} -> {count.live}, RSS {rss:.0f} MB -> +{grown:.1f} MB")
        assert grown < 8, grown
    finally:
        m.free()
    assert count.live == 0, f"{count.live} blocks left after freeing the model"
