# Development

## Install from the repo

1. Build the runtime library: `make lib` in the `sprongmotion-runtime` repo gives `build/libsprongmotion.dylib`
   (`.so` on Linux, `.dll` on Windows). You also need a model file (`.gguf`), for example `SprongMotion-14M-v1.0-Q8_0.gguf`.
2. In Blender (5.0 or newer): Preferences > Get Extensions > Repositories > + > Add Local Repository, pointing at an
   empty folder. In that folder, symlink this repo as `sprongmotion`, then enable SprongMotion under Add-ons.
3. In the add-on's preferences, set Library and Model to the two files. Edits to the code apply after a Blender restart
   (or disabling and enabling the add-on).

`make build` writes a zip without the library and model to `dist/`. `make release` writes the three release zips
(`release.sh`): it takes the libraries from `RUNTIME` (default `../sprongmotion-runtime`) and the model and its
license files from `MODEL_DIR`, and skips a platform whose library is missing. `.github/workflows/release.yml` runs the
tests and builds the same zips on a `v*` tag.

## Tests

| Command | What runs |
|---|---|
| `make unit` | Pure-Python tests in `tests/unit` (pytest, Blender stubbed). Under a second. |
| `make test` | The unit tests, then every `tests/test_*.py` in its own headless Blender, side by side. About 15 s. |
| `make test-slow` | The same, with the long stress runs (`SM_SLOW=1`). |
| `make lint` | `ruff check` and `ruff format --check`. |

`PYTHON` must have pytest and numpy (`pip install pytest numpy`). Without them, `make test` says so and runs only the
Blender tests. `PYTHON` and `RUFF` default to `.venv/bin/python` and `.venv/bin/ruff` when those exist.

Settings: `SM_MODEL` (the model file, required for the Blender tests) and `SM_LIB` (the runtime library, default
`../sprongmotion-runtime/build`), then all optional: `BLENDER` (the Blender binary), `JOBS` (Blender test
files at once, 10), `SM_THREADS` (model threads per Blender, 2 under make), `SM_SLOW`. A Blender test file runs with
`blender -b --factory-startup --python tests/run.py -- tests/test_x.py`. Each test starts from an empty file.

## Modules

| File | Does |
|---|---|
| `runtime.py` | ctypes binding of the runtime (`include/sprongmotion.h`) |
| `model.py` | Loading the runtime library and the model, the add-on preferences, unknown words |
| `properties.py` | The data on each armature (bone map, settings) and on each action (its blocks, Loop, path), moving old files' blocks to the action |
| `blocks.py` | The block list: add, split, resize, move, place freely, runs, stored codes |
| `generate.py` | Making takes, at once or from a timer, and the undo steps around them |
| `timeline_motion.py` | Generate calls over the timeline's blocks, which blocks need a take, decoding the timeline |
| `takes.py` | Putting a decoded take on the rig: timing, contacts, easing to rest, per-block actions |
| `retarget.py` | Rest-pose maths from the model's 22 joints onto any rig, leg and arm IK, writing the keys |
| `quat.py` | Quaternion maths on numpy arrays |
| `keyframes.py` | F-curve maths: unbaked Bezier keys within a tolerance |
| `rig.py` | The 22-joint skeleton, rig presets, matching a rig by its structure, the add-on's own rig |
| `continue_keys.py` | The rig's own keys as context blocks: Continue from Keys, Fill Gap, reading a pose back to the 22 joints |
| `paths.py` | The rig's path: curve to model frame, fitting block lengths, the turn budget, re-applying after edits |
| `return_to_start.py` | Return to Start: the way back and its length |
| `rig_data.py` | Frozen bones, linked and override checks, Remove SprongMotion Data (per action, whole file) |
| `timeline_lane.py` | The block lane: hit testing and mouse operators |
| `timeline_draw.py` | Drawing the block lane |
| `block_text.py` | Typing a block's text in the block or in a popup |
| `typing_help.py` | Typing help: the rest of a sentence (`wordlists/sentences.txt`, ours) ranked by the text encoder, word completion from the model's words (`wordlists/words.txt`) |
| `block_operators.py` | The operators that edit blocks |
| `operators.py` | The other operators and the keymaps |
| `panels.py` | The sidebar panels (Dope Sheet / Timeline, a pointer in the 3D view), menu entries |
