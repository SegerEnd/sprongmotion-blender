"""ctypes binding for the calls of libsprongmotion the add-on uses (include/sm.h)."""

import contextlib
import ctypes as C

import numpy as np

P_FOOT_LOCK, P_IN_PLACE, P_LOOP = 9, 10, 11
I_FPS, I_FRAMES_PER_TOKEN, I_MAX_FRAMES, I_JOINTS, I_CODE_LAYERS, I_ENCODER = 1, 2, 3, 4, 9, 11

_P, _U32, _I32, _SIZE = C.c_void_p, C.c_uint32, C.c_int32, C.c_size_t
_PU32, _PF = C.POINTER(C.c_uint32), C.POINTER(C.c_float)
_SIGS = {
    "sm_version": (_U32, []),
    "sm_status_message": (C.c_char_p, [_I32]),
    "sm_model_load_file": (_I32, [C.c_char_p, _P, C.POINTER(_P), C.c_char_p, _SIZE]),
    "sm_model_free": (None, [_P]),
    "sm_model_info": (_I32, [_P, _I32]),
    "sm_model_id": (C.c_char_p, [_P]),
    "sm_model_vocab": (C.c_char_p, [_P, _U32]),
    "sm_skeleton_offsets": (_I32, [_P, C.POINTER(C.c_float), _U32]),
    "sm_ctx_create": (_I32, [_P, C.POINTER(_P)]),
    "sm_ctx_free": (None, [_P]),
    "sm_ctx_last_error": (C.c_char_p, [_P]),
    "sm_request_create": (_I32, [_P, C.POINTER(_P)]),
    "sm_request_free": (None, [_P]),
    "sm_request_set_int": (_I32, [_P, _I32, _I32]),
    "sm_request_set_path": (_I32, [_P, _PF, _U32]),
    "sm_text_create": (_I32, [_P, C.POINTER(_P)]),
    "sm_text_free": (None, [_P]),
    "sm_ctx_encode_text": (_I32, [_P, C.c_char_p, _I32, _P]),
    "sm_text_vector": (_I32, [_P, C.POINTER(_PF), _PU32]),
    "sm_text_unknown_words": (_I32, [_P, C.c_char_p, _SIZE, C.POINTER(_SIZE), _PU32]),
    "sm_motion_create": (_I32, [_P, C.POINTER(_P)]),
    "sm_motion_free": (None, [_P]),
    "sm_ctx_generate_blocks": (
        _I32,
        [
            _P,
            _P,
            _U32,
            C.POINTER(C.c_char_p),
            _P,
            C.POINTER(C.c_double),
            C.POINTER(C.c_int32),
            C.POINTER(C.POINTER(C.c_int16)),
            _PU32,
            _P,
        ],
    ),
    "sm_ctx_encode_motion": (_I32, [_P, _P, _PF, _U32, _P]),
    "sm_motion_rotations": (C.POINTER(C.c_float), [_P, _PU32]),
    "sm_motion_joints": (C.POINTER(C.c_float), [_P, _PU32]),
    "sm_motion_codes": (C.POINTER(C.c_int16), [_P, _PU32]),
    "sm_motion_contacts": (C.POINTER(C.c_uint8), [_P, _PU32]),
    "sm_motion_speed": (_PF, [_P, _PU32]),
    "sm_motion_root": (_PF, [_P, _PU32]),
    "sm_motion_write_bvh": (_I32, [_P, C.c_double, C.c_char_p, _SIZE, C.POINTER(_SIZE)]),
}


class Hooks(C.Structure):
    """sm_hooks: the thread count (0 = every core) and optionally an allocator; the rest stays the library's."""

    _fields_ = [("struct_size", _U32), ("alloc", _P), ("free", _P), ("log", _P), ("parallel_for", _P),
                ("n_threads", _U32), ("user", _P)]  # fmt: skip

    def __init__(self, n_threads=0, alloc=None, free=None):
        super().__init__(C.sizeof(Hooks), alloc, free, None, None, n_threads, None)


class Error(RuntimeError):
    pass


class Lib:
    def __init__(self, path):
        self.path, self.c = path, C.CDLL(path)
        for name, (res, args) in _SIGS.items():
            fn = getattr(self.c, name)
            fn.restype, fn.argtypes = res, args
        if self.c.sm_version() >> 16 != 1:
            raise Error(f"{path}: runtime major version {self.c.sm_version() >> 16}, the add-on needs 1")

    def check(self, status, msg=b""):
        if status < 0:
            raise Error((msg or self.c.sm_status_message(status)).decode())
        return status


class Model:
    def __init__(self, lib, path, hooks=None):
        self.lib, c = lib, lib.c
        self.h, err = _P(), C.create_string_buffer(256)
        lib.check(c.sm_model_load_file(path.encode(), hooks, C.byref(self.h), err, 256), err.value)
        self.id = c.sm_model_id(self.h).decode()
        self.fps = c.sm_model_info(self.h, I_FPS)
        self.fpt, self.layers = c.sm_model_info(self.h, I_FRAMES_PER_TOKEN), c.sm_model_info(self.h, I_CODE_LAYERS)
        self.max_seconds = c.sm_model_info(self.h, I_MAX_FRAMES) / self.fps
        self.encoder = c.sm_model_info(self.h, I_ENCODER) == 1
        n = c.sm_model_info(self.h, I_JOINTS)
        off = (C.c_float * (n * 3))()
        lib.check(c.sm_skeleton_offsets(self.h, off, n * 3))
        self.offsets = np.array(off, np.float32).reshape(n, 3)
        self.ctx, self.req, self.text = _P(), _P(), _P()
        lib.check(c.sm_ctx_create(self.h, C.byref(self.ctx)))
        lib.check(c.sm_request_create(self.h, C.byref(self.req)))
        lib.check(c.sm_text_create(self.h, C.byref(self.text)))

    def free(self):
        c = self.lib.c
        c.sm_text_free(self.text)
        c.sm_request_free(self.req)
        c.sm_ctx_free(self.ctx)
        c.sm_model_free(self.h)

    def _options(self, foot_lock, in_place, loop, path):
        c = self.lib.c
        for p, v in ((P_FOOT_LOCK, foot_lock), (P_IN_PLACE, in_place), (P_LOOP, loop)):
            c.sm_request_set_int(self.req, p, int(v))
        pts = np.zeros((0, 4), np.float32) if path is None else np.ascontiguousarray(path, np.float32)
        self.lib.check(c.sm_request_set_path(self.req, pts.ctypes.data_as(_PF), len(pts)), self.error())

    @contextlib.contextmanager
    def blocks(self, texts, seconds, seeds, keep, foot_lock=True, in_place=False, loop=False, path=None):
        """Generate prompt blocks in one motion. Yields (motion, first frame of each block, end).

        keep[i]: a block's codes (int16 [tokens, layers]) to keep, None to make it. path: [n, 4] (t, x, z, heading).
        """
        n, c, out = len(texts), self.lib.c, Motion(self)
        keep = [None if k is None else np.ascontiguousarray(k, np.int16) for k in keep]
        ptrs = (C.POINTER(C.c_int16) * n)(*[k.ctypes.data_as(C.POINTER(C.c_int16)) if k is not None else None
                                            for k in keep])  # fmt: skip
        first = (C.c_uint32 * (n + 1))()
        try:
            self._options(foot_lock, in_place, loop, path)
            status = c.sm_ctx_generate_blocks(
                self.ctx, self.req, n, (C.c_char_p * n)(*[t.encode() for t in texts]), None,
                (C.c_double * n)(*seconds), (C.c_int32 * n)(*seeds), ptrs, first, out.h,
            )  # fmt: skip
            self.lib.check(status, self.error())
            yield out, list(first)
        finally:
            out.free()

    def encode(self, joints):
        """Motion to codes: joints [frames, 22, 3] (metres, Y up, model fps) to int16 [(frames - 1) // 2, layers]."""
        out, j = Motion(self), np.ascontiguousarray(joints, np.float32)
        try:
            status = self.lib.c.sm_ctx_encode_motion(self.ctx, None, j.ctypes.data_as(_PF), len(j), out.h)
            self.lib.check(status, self.error())
            return out.codes()
        finally:
            out.free()

    def vocab(self):
        """The whole words the model knows, sorted."""
        words, get = [], self.lib.c.sm_model_vocab
        while (w := get(self.h, len(words))) is not None:
            words.append(w.decode())
        return words

    def _encode(self, text):
        self.lib.check(self.lib.c.sm_ctx_encode_text(self.ctx, text.encode(), -1, self.text), self.error())

    def vector(self, text):
        """The text encoder's pooled vector."""
        v, n = _PF(), C.c_uint32()
        self._encode(text)
        self.lib.check(self.lib.c.sm_text_vector(self.text, C.byref(v), C.byref(n)))
        return np.ctypeslib.as_array(v, (n.value,)).copy()

    def unknown_words(self, text):
        c, need, n = self.lib.c, _SIZE(), C.c_uint32()
        self._encode(text)
        self.lib.check(c.sm_text_unknown_words(self.text, None, 0, C.byref(need), C.byref(n)))
        buf = C.create_string_buffer(need.value)
        self.lib.check(c.sm_text_unknown_words(self.text, buf, need.value, C.byref(need), C.byref(n)))
        return buf.value.decode().split()

    def error(self):
        return self.lib.c.sm_ctx_last_error(self.ctx)


class Motion:
    def __init__(self, model):
        self.model, self.h = model, _P()
        model.lib.check(model.lib.c.sm_motion_create(model.h, C.byref(self.h)))

    def free(self):
        self.model.lib.c.sm_motion_free(self.h)

    def _array(self, name, width):
        n = C.c_uint32()
        ptr = getattr(self.model.lib.c, name)(self.h, C.byref(n))
        return np.ctypeslib.as_array(ptr, (n.value,)).reshape(-1, *width).copy() if n.value else np.zeros((0, *width))

    def rotations(self):
        """[frames, joints, 4] x y z w, each relative to its parent."""
        return self._array("sm_motion_rotations", (len(self.model.offsets), 4))

    def joints(self):
        """[frames, joints, 3]: positions as output (in place when asked), the path the rotations go with."""
        return self._array("sm_motion_joints", (len(self.model.offsets), 3))

    def pelvis(self):
        return self.joints()[:, 0]

    def contacts(self):
        """[frames, 4] on the ground: l_ankle, l_foot, r_ankle, r_foot."""
        return self._array("sm_motion_contacts", (4,)).astype(bool)

    def speed(self):
        """[frames] m/s: the model's own root speed, before a path warp."""
        return self._array("sm_motion_speed", ())

    def root(self):
        """[frames, 3]: the root's full travel, also with In Place."""
        return self._array("sm_motion_root", (3,))

    def codes(self):
        return self._array("sm_motion_codes", (self.model.layers,)).astype(np.int16)

    def bvh(self, scale=100.0):
        c, need = self.model.lib.c, _SIZE()
        self.model.lib.check(c.sm_motion_write_bvh(self.h, scale, None, 0, C.byref(need)))
        buf = C.create_string_buffer(need.value)
        self.model.lib.check(c.sm_motion_write_bvh(self.h, scale, buf, need.value, C.byref(need)))
        return buf.value.decode()
