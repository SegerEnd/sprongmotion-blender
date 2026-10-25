# Changelog

## Unreleased

- Hand Lock: hands on the floor in the motion (push-ups, crawling) stay put and on the floor, also on rigs whose arms
  are longer or shorter than the model's.
- Floor: no part of the body goes below the floor. Wrists and ankles by IK, toes by turning the foot, the rest by
  lifting the body.

## 1.0.0 (2026-10-21)

First release, with the SprongMotion runtime 1.0 and the SprongMotion-14M 1.0 model bundled per platform (macOS arm64,
Linux x86-64, Windows x86-64).

- Prompt blocks on a strip in the Timeline and Dope Sheet: add, move, resize, split, copy, paste, duplicate, snapping;
  blocks per action.
- Takes per block, Lock, Fresh Start, In Place, Return to Start; Hold Last Pose or Ease to Rest after a block.
- Retargeting onto any humanoid rig by its structure (Rigify and Mixamo presets), leg IK with planted feet.
- Paths from a curve or a Grease Pencil stroke, with block lengths fitted to the path.
- Continue from Keys, Fill Gap, Regenerate a range.
- Typing help: sentence and word completion.
- Bake (one action, or one per block), Export BVH, Clear Motion, Remove SprongMotion Data.
