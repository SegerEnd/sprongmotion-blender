"""The runtime library and model the add-on generates with, and the word help built from the model."""

import os
import platform

import bpy

from . import blocks, generate, runtime, typing_help

THREADS = 0  # model threads, 0 = every core
EDGE_TOKENS = 12  # tokens sm_ctx_generate_blocks adds to a block: seam 3 + look-ahead 8, a lone loop 12
HERE = os.path.dirname(__file__)
BUNDLED_MODEL = os.path.join(HERE, "model", "SprongMotion-14M-v1.0-Q8_0.gguf")
ARCHITECTURES = {"arm64": "arm64", "aarch64": "arm64"}
LIBRARY_EXTENSIONS = {"Darwin": ".dylib", "Windows": ".dll"}


class _State:
    def __init__(self):
        self.lib = self.model = self.paths = self.help = None
        self.error = ""
        self.unknown = {}  # text -> the words in it the model does not know
        self.near = {}  # unknown word -> known words spelled like it


_state = _State()


def prefs():
    addon = bpy.context.preferences.addons.get(__package__)
    return addon.preferences if addon else None


def bundled():
    """The library and model shipped inside the extension, "" for a missing one."""
    system = platform.system()
    arch = ARCHITECTURES.get(platform.machine().lower(), "x64")
    tag = f"{system.lower().replace('darwin', 'macos')}-{arch}"
    lib = os.path.join(HERE, "lib", tag, "libsprongmotion" + LIBRARY_EXTENSIONS.get(system, ".so"))
    return (lib if os.path.isfile(lib) else ""), (BUNDLED_MODEL if os.path.isfile(BUNDLED_MODEL) else "")


def load():
    """The model, loaded again when the paths in the preferences changed. Never call it from a draw."""
    p, (lib, gguf) = prefs(), bundled()
    paths = (bpy.path.abspath(p.library_path) or lib, bpy.path.abspath(p.model_path) or gguf) if p else ("", "")
    if paths != _state.paths:
        unload()
        _state.paths, _state.error = paths, _load_files(*paths)
    return _state.model


def _load_files(lib_path, model_path):
    """Load the library and model; "" when done, else why not."""
    if not lib_path:
        return "Choose the SprongMotion library (libsprongmotion) in the add-on preferences"
    if not model_path:
        return "Choose a SprongMotion model file (.gguf) in the add-on preferences"
    if _state.lib and _state.lib.path != lib_path:
        return "Restart Blender to use another SprongMotion library"  # a loaded library cannot be unloaded safely
    try:
        _state.lib = _state.lib or runtime.Lib(lib_path)
    except (OSError, AttributeError, runtime.Error) as e:
        return f"Cannot load the library {lib_path}: {e}"
    try:
        _state.model = runtime.Model(_state.lib, model_path, THREADS)
    except runtime.Error as e:
        return f"Cannot load the model {model_path}: {e}"
    sentences = typing_help.lines("sentences.txt")
    known = set(_state.model.vocab()) & set(typing_help.lines("words.txt"))
    long_words = {w for s in sentences for w in s.split() if len(w) > 3}
    _state.help = typing_help.Help(sentences, known | long_words, _state.model.vector)
    return ""


def loaded():
    """The model as loaded now, for draws: loads nothing."""
    return _state.model


def error():
    return _state.error


def set_error(message):
    _state.error = message


def unload():
    generate.cancel()
    if _state.model:
        _state.model.free()
    _state.model, _state.paths, _state.help = None, None, None
    _state.unknown, _state.near = {}, {}


def max_seconds():
    """The longest block the runtime makes with whatever edges it has."""
    m = _state.model
    if m is None:
        return blocks.MAX_SECONDS - 1.2
    tokens = m.max_seconds * m.fps / m.fpt - EDGE_TOKENS
    return tokens * m.fpt / m.fps


def unknown_words(text, look=True):
    """The words of text the model does not know. look=False only reads the cache (for draws)."""
    if look and text not in _state.unknown and _state.model:
        try:
            _state.unknown[text] = _state.model.unknown_words(text)
        except runtime.Error:
            _state.unknown[text] = []
        for word in _state.unknown[text]:
            if word not in _state.near and _state.help:
                _state.near[word] = _state.help.near(word)
    return _state.unknown.get(text, [])


def nearest(word):
    """Known words spelled like an unknown one, as unknown_words found them."""
    return _state.near.get(word, [])


def helper():
    """The typing help, None without a model or when the preferences turn it off."""
    p = prefs()
    return _state.help if p is None or p.typing_help else None
