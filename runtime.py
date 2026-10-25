"""ctypes binding for the calls of libsprongmotion the add-on uses (sprongmotion.h)."""

import contextlib
import ctypes as C

import numpy as np

NEEDS = (1, 1)  # the oldest runtime version with this API

_FLOATS, _CODES = C.POINTER(C.c_float), C.POINTER(C.c_int16)


class PathPoint(C.Structure):
    _fields_ = [("time", C.c_float), ("x", C.c_float), ("z", C.c_float), ("heading", C.c_float)]


class ModelInfo(C.Structure):
    _fields_ = [
        ("id", C.c_char_p),
        ("fps", C.c_int),
        ("frames_per_token", C.c_int),
        ("max_seconds", C.c_float),
        ("code_layers", C.c_int),
        ("feature_size", C.c_int),
        ("text_vector_size", C.c_int),
        ("joint_count", C.c_int),
        ("joint_names", C.POINTER(C.c_char_p)),
        ("joint_parents", C.POINTER(C.c_int)),
        ("rest_offsets", _FLOATS),
        ("has_generator", C.c_bool),
        ("has_encoder", C.c_bool),
        ("follows_path", C.c_bool),
        ("internal", C.c_void_p),
    ]


class Options(C.Structure):
    _fields_ = [
        ("seconds", C.c_float),
        ("seed", C.c_int),
        ("steps", C.c_int),
        ("residual_layers", C.c_int),
        ("text_guidance", C.c_float),
        ("residual_guidance", C.c_float),
        ("temperature", C.c_float),
        ("top_k", C.c_int),
        ("foot_lock", C.c_bool),
        ("in_place", C.c_bool),
        ("loop", C.c_bool),
        ("path_warp", C.c_bool),
        ("path", C.POINTER(PathPoint)),
        ("path_count", C.c_int),
        ("previous", C.c_void_p),
        ("pinned_codes", _CODES),
        ("pinned_token_count", C.c_int),
        ("text", C.c_void_p),
        ("progress", C.c_void_p),
        ("user", C.c_void_p),
    ]


class Block(C.Structure):
    _fields_ = [
        ("prompt", C.c_char_p),
        ("text", C.c_void_p),
        ("seconds", C.c_float),
        ("seed", C.c_int),
        ("kept_codes", _CODES),
    ]


class Text(C.Structure):
    _fields_ = [
        ("vector", _FLOATS),
        ("vector_size", C.c_int),
        ("unknown_word_count", C.c_int),
        ("unknown_words", C.c_char_p),
        ("internal", C.c_void_p),
    ]


class MotionData(C.Structure):
    _fields_ = [
        ("frame_count", C.c_int),
        ("fps", C.c_int),
        ("seed", C.c_int),
        ("joints", _FLOATS),
        ("rotations", _FLOATS),
        ("offsets", _FLOATS),
        ("root", _FLOATS),
        ("speed", _FLOATS),
        ("contacts", C.POINTER(C.c_uint8)),
        ("features", _FLOATS),
        ("codes", _CODES),
        ("confidence", _FLOATS),
        ("token_count", C.c_int),
        ("path_error", C.c_float),
        ("block_start_frames", C.POINTER(C.c_int)),
        ("block_count", C.c_int),
        ("internal", C.c_void_p),
    ]


_SIGNATURES = {
    "sm_get_version": (C.c_char_p, []),
    "sm_get_error": (C.c_char_p, []),
    "sm_set_thread_count": (None, [C.c_int]),
    "sm_set_memory_functions": (None, [C.c_void_p, C.c_void_p]),
    "sm_load_model": (ModelInfo, [C.c_char_p]),
    "sm_is_model_valid": (C.c_bool, [ModelInfo]),
    "sm_unload_model": (None, [ModelInfo]),
    "sm_get_model_word": (C.c_char_p, [ModelInfo, C.c_int]),
    "sm_default_options": (Options, [ModelInfo]),
    "sm_generate_timeline": (MotionData, [ModelInfo, C.POINTER(Block), C.c_int, Options]),
    "sm_encode_motion": (MotionData, [ModelInfo, _FLOATS, C.c_int, Options]),
    "sm_is_motion_valid": (C.c_bool, [MotionData]),
    "sm_unload_motion": (None, [MotionData]),
    "sm_export_motion_bvh_text": (C.c_void_p, [MotionData, C.c_float]),
    "sm_encode_text": (Text, [ModelInfo, C.c_char_p]),
    "sm_is_text_valid": (C.c_bool, [Text]),
    "sm_unload_text": (None, [Text]),
    "sm_free": (None, [C.c_void_p]),
}


class Error(RuntimeError):
    pass


def version_tuple(text):
    """ "1.1.0" -> (1, 1, 0)."""
    return tuple(int(part) for part in text.split("."))


class Lib:
    def __init__(self, path):
        self.path, self.c = path, C.CDLL(path)
        if not hasattr(self.c, "sm_get_version"):
            raise Error(f"{path} is an older runtime, the add-on needs {NEEDS[0]}.{NEEDS[1]} or newer")
        for name, (result, args) in _SIGNATURES.items():
            function = getattr(self.c, name)
            function.restype, function.argtypes = result, args
        version = version_tuple(self.c.sm_get_version().decode())
        if version[0] != NEEDS[0] or version[:2] < NEEDS:
            raise Error(f"{path} is runtime {'.'.join(map(str, version))}, the add-on needs {NEEDS[0]}.{NEEDS[1]}")

    def error(self):
        return self.c.sm_get_error().decode()


class Model:
    def __init__(self, lib, path, threads=0):
        self.lib, c = lib, lib.c
        c.sm_set_thread_count(threads)
        self.info = c.sm_load_model(path.encode())
        if not c.sm_is_model_valid(self.info):
            raise Error(lib.error())
        info = self.info
        self.id = info.id.decode()
        self.fps, self.fpt, self.layers = info.fps, info.frames_per_token, info.code_layers
        self.max_seconds = info.max_seconds
        self.encoder = info.has_encoder
        self.offsets = np.ctypeslib.as_array(info.rest_offsets, (info.joint_count, 3)).copy()

    def free(self):
        self.lib.c.sm_unload_model(self.info)

    def _options(self, foot_lock, in_place, loop, path):
        """Options with the add-on's switches; the path array must stay alive while they are used."""
        options = self.lib.c.sm_default_options(self.info)
        options.foot_lock, options.in_place, options.loop = foot_lock, in_place, loop
        points = None
        if path is not None and len(path):
            points = np.ascontiguousarray(path, np.float32)
            options.path = points.ctypes.data_as(C.POINTER(PathPoint))
            options.path_count = len(points)
        return options, points

    def _checked(self, motion):
        if not self.lib.c.sm_is_motion_valid(motion):
            raise Error(self.lib.error())
        return Motion(self, motion)

    @contextlib.contextmanager
    def blocks(self, texts, seconds, seeds, keep, foot_lock=True, in_place=False, loop=False, path=None):
        """Generate prompt blocks in one motion. Yields (motion, first frame of each block, end).

        keep[i]: a block's codes (int16 [tokens, layers]) to keep, None to make it. path: [n, 4] (t, x, z, heading).
        """
        kept = [None if k is None else np.ascontiguousarray(k, np.int16) for k in keep]
        prompts = [t.encode() for t in texts]
        items = (Block * len(texts))()
        for item, prompt, length, seed, codes in zip(items, prompts, seconds, seeds, kept, strict=True):
            item.prompt, item.seconds, item.seed = prompt, length, seed
            item.kept_codes = codes.ctypes.data_as(_CODES) if codes is not None else None
        options, _points = self._options(foot_lock, in_place, loop, path)
        motion = self._checked(self.lib.c.sm_generate_timeline(self.info, items, len(texts), options))
        try:
            data = motion.data
            yield motion, list(np.ctypeslib.as_array(data.block_start_frames, (data.block_count + 1,)))
        finally:
            motion.free()

    def encode(self, joints):
        """Motion to codes: joints [frames, 22, 3] (metres, Y up, model fps) to int16 [(frames - 1) // 2, layers]."""
        joints = np.ascontiguousarray(joints, np.float32)
        options = self.lib.c.sm_default_options(self.info)
        motion = self._checked(
            self.lib.c.sm_encode_motion(self.info, joints.ctypes.data_as(_FLOATS), len(joints), options)
        )
        try:
            return motion.codes()
        finally:
            motion.free()

    def vocab(self):
        """The whole words the model knows, sorted."""
        words = []
        while (word := self.lib.c.sm_get_model_word(self.info, len(words))) is not None:
            words.append(word.decode())
        return words

    @contextlib.contextmanager
    def _text(self, prompt):
        text = self.lib.c.sm_encode_text(self.info, prompt.encode())
        if not self.lib.c.sm_is_text_valid(text):
            raise Error(self.lib.error())
        try:
            yield text
        finally:
            self.lib.c.sm_unload_text(text)

    def vector(self, prompt):
        """The text encoder's pooled vector."""
        with self._text(prompt) as text:
            return np.ctypeslib.as_array(text.vector, (text.vector_size,)).copy()

    def unknown_words(self, prompt):
        with self._text(prompt) as text:
            return text.unknown_words.decode().split()

    def error(self):
        return self.lib.error()


class Motion:
    def __init__(self, model, data):
        self.model, self.data = model, data

    def free(self):
        self.model.lib.c.sm_unload_motion(self.data)

    def _array(self, pointer, count, width):
        if not count:
            return np.zeros((0, *width))
        return np.ctypeslib.as_array(pointer, (count * int(np.prod(width)),)).reshape(-1, *width).copy()

    def rotations(self):
        """[frames, joints, 4] x y z w, each relative to its parent."""
        return self._array(self.data.rotations, self.data.frame_count, (len(self.model.offsets), 4))

    def joints(self):
        """[frames, joints, 3]: positions as output (in place when asked), the path the rotations go with."""
        return self._array(self.data.joints, self.data.frame_count, (len(self.model.offsets), 3))

    def pelvis(self):
        return self.joints()[:, 0]

    def contacts(self):
        """[frames, 4] on the ground: l_ankle, l_foot, r_ankle, r_foot."""
        return self._array(self.data.contacts, self.data.frame_count, (4,)).astype(bool)

    def speed(self):
        """[frames] m/s: the model's own root speed, before a path warp."""
        return self._array(self.data.speed, self.data.frame_count, ())

    def root(self):
        """[frames, 3]: the root's full travel, also with In Place."""
        return self._array(self.data.root, self.data.frame_count, (3,))

    def codes(self):
        return self._array(self.data.codes, self.data.token_count, (self.model.layers,)).astype(np.int16)

    def bvh(self, scale=100.0):
        c = self.model.lib.c
        pointer = c.sm_export_motion_bvh_text(self.data, scale)
        if not pointer:
            raise Error(self.model.lib.error())
        try:
            return C.string_at(pointer).decode()
        finally:
            c.sm_free(pointer)
