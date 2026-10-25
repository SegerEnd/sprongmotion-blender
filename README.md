# SprongMotion for Blender

Text to motion for armatures. Type prompt blocks on a strip at the top of the Timeline ("walks forward", then "sits
down"): each block's motion is made on your CPU by [SprongMotion-14M](https://huggingface.co/SegerEnd/SprongMotion-14M),
retargeted onto your rig and written as keys, in a fraction of a second. No GPU, no account, nothing online.

- Prompt blocks on the timeline: move, resize, split, copy and paste them; an edit makes only the blocks it touches again.
- Any humanoid rig: Rigify, Mixamo-style and custom rigs are matched by their structure, with IK that keeps the feet
  on the ground and hands on the floor, and nothing below it.
- Takes per block (Try Again), Lock, Fresh Start, In Place, Return to Start.
- Paths: the rig follows a curve or a Grease Pencil stroke, with block lengths fitted to it.
- Continue from your own keys or mocap, and Fill Gap between them.
- Hold or ease to rest after a block, Loop, Bake to plain keys, Export BVH.
- Typing help: sentence and word completion from what the model knows.

## Install

1. Download the zip for your system from the [releases](https://github.com/SegerEnd/sprongmotion-blender/releases):
   `sprongmotion-1.0.0-macos-arm64.zip`, `-linux-x64` or `-windows-x64`.
2. In Blender 5.0 or newer: Edit > Preferences > Get Extensions > the arrow menu at the top right > Install from Disk,
   and pick the zip.
3. Select an armature, open the Dope Sheet or Timeline sidebar (N), type a motion and press Enter.

The runtime library and the model are inside the zip: nothing needs setting up. The Library and Model fields in the
add-on's preferences are optional overrides. [docs/usage.md](docs/usage.md) explains the lane, the sidebar, paths and
continuing from your keys.

## Compatibility

| | |
|---|---|
| Blender | 5.0 or newer |
| Systems | macOS on Apple Silicon, Linux x86-64, Windows x86-64 |
| Rigs | humanoid armatures; the model animates 22 joints, other bones keep their keys or rest pose |

## Troubleshooting

- **"Needs the hips and legs mapped"**: the rig's bones were not all matched. Pick the rig's preset, or set the bones,
  in the sidebar's Rig panel.
- **"Not understood" under a block**: a word the model does not know. The note suggests known words spelled alike.
- **macOS refuses to load the library**: if the files kept the download quarantine, run
  `xattr -dr com.apple.quarantine` on the extension's folder (Preferences > File Paths shows where extensions live)
  and restart Blender.
- **After installing a new version**: restart Blender before using it.

## Known limits

- Spins and pirouettes can glitch (a sudden jump in facing).
- The seam between two different prompts can jerk.
- Runs are fast (about 2.7 m/s), and a plain "runs" can float above the ground.
- Walks can slide the feet a little, more so on paths with corners, zigzags or small circles.
- A block is at most about 9 s: longer motion takes several blocks.

## Licenses

| Part | License | File in the zip |
|---|---|---|
| The add-on (this repo) | GPL-3.0-or-later | `licenses/ADDON-GPL-3.0-or-later.txt` (`LICENSE` here) |
| The runtime library (`lib/`), [sprongmotion-runtime](https://github.com/SegerEnd/sprongmotion-runtime) | Apache-2.0 | `licenses/RUNTIME-Apache-2.0.txt`, `licenses/RUNTIME-NOTICE.txt` |
| The model (`model/SprongMotion-14M-v1.0-Q8_0.gguf`) | Apache-2.0 | `licenses/MODEL-Apache-2.0.txt`, `licenses/MODEL-NOTICE.txt`, `licenses/MODEL_CARD.md` |

The model card has a request on use and the credit line.

Building, tests and the code layout: [docs/development.md](docs/development.md).
