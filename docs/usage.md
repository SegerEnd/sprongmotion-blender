# Using SprongMotion in Blender

## The lane and the sidebar

The blocks live in one place: the strip at the top of the Timeline. On an empty strip, "+ Add motion" starts typing a
new block. Click a block to select it, drag its middle to move it to any frame (Shift: all blocks together), drag
its edges to resize it, right-click for its menu (Lock, On, takes, Split, Fresh Start). Blocks never overlap: one
dropped onto another goes to that block's nearest free edge. Dragging an edge ripples the block's run: the blocks
touching it move along, so the run stays tight and the gaps around it keep their size (a block after a gap moves only
if it would be overlapped). Alt while dragging an edge resizes the block alone. The sidebar's up and down arrows swap the selected block with its neighbour. With the Timeline's snap magnet
on, a dragged block or edge snaps within 8 px to the other blocks' edges, the playhead and the Snap To choice (Frame,
Second or Nearest Marker). Ctrl inverts it. A lock and a
struck-through circle on a block show it locked or off. Fresh Start makes a block's motion as if it came first, without
the block before it, then blends into it from that block's last pose over 0.4 s (the block after it still continues from
it, and it does nothing on the first block or with Loop).
With the mouse over the strip, Ctrl+C copies the selected block with its takes (as text on the clipboard, so it pastes
into another rig or file), Ctrl+V pastes it after the selected block and Shift+D duplicates it (also in the right-click
menu): the blocks touching it after shift on. The copy keeps its take, so nothing is made (a take from another model is made again).
A block's In Place (sidebar or right-click menu) drops its horizontal root travel: the block after it goes on from
where it stands, facing the same way. On a path it stands where the path reaches it, and the later blocks follow the rest of the curve's
shape from there (moved back by the stretch it would have walked, none for a block that stands still anyway). The rig-wide
In Place sets or clears it on every block (it shows on only when all are in place).
A block's Return to Start (sidebar or right-click menu) walks it back to where its run started, on a U-turn of 2 m or more the add-on makes, with its length fitted like a path's (the lock keeps one, amber when it does not fit), unless the rig has its own Path.

Blocks that touch (up to 2 frames apart) are one run: each goes on from the one before it. A gap splits the timeline
into runs, and a block after a gap starts afresh where the rig stands, facing the same way (made as if it came first,
no blend: the gap is between them). In the gap the rig does what Preferences > After a Block says. A transition in a
gap comes only on request: right-click the gap in the lane > Fill Gap puts a block exactly in it (Text in the redo
panel) that goes on from the block before and leads into the block after, both keeping their takes. With Loop or a
path the timeline stays one run.

The SprongMotion panels are in the Dope Sheet / Timeline sidebar (N), next to the blocks; the 3D view's sidebar only
points there. The main panel has the motion field (Enter adds a block), the list of blocks and the selected block (text,
length, seed with Try Again, takes, Lock, Fresh Start, In Place, Continue from Keys when the rig's own keys are right next
to it). Below it: the rig (its preset and how many bones matched the 22 joints, counted before any take), Freeze, Path,
Keys, and Output with Bake, Export BVH, Clear Motion and Remove. Clear Motion is Remove without baking: the generated
keys and the blocks go, and the bones they animated go back to their rest pose (bones with your own keys keep them).

Preferences > After a Block: Hold Last Pose (the default), or Ease to Rest, where the bones the add-on animates ease
back to their rest pose (the rig stays where it walked to) over the Ease Time after the last block and in gaps between blocks (not with Loop, not
where your own keys follow, frozen bones stay as they are). In a gap, Hold Last Pose holds it and eases into the next
block over the Ease Time. Your own keys in a gap stay.

## Blocks belong to the action

The blocks (with their takes, Loop and the path) are stored on the action the rig plays, not on the rig. Nothing
changes in the everyday flow: select a rig, type a motion, it plays. A rig without an action gets one on its first
block, named after the rig, as the generated keys always did. The lane and the sidebar show the blocks of the action
the rig plays now, so picking another action in the Action Editor shows its blocks, and a new empty action gives an
empty lane. Blender's own Duplicate (or New with an action assigned) copies the blocks with the action: edits to the
copy leave the original alone. An action played by two rigs has one set of blocks, and each rig keys them with its
own bone map into its own slot. The bone map and rig preset, Frozen bones, Foot Lock, Takes, Auto and the key
settings stay on the rig.

Remove SprongMotion Data and Clear Motion work on the playing action: its blocks and generated keys go, other actions
keep theirs, and the rig keeps its settings while another action with a slot of its name still has blocks. Remove
from Whole File clears every action and every rig. The blocks of a linked or overridden action are read-only: the lane
shows them and the sidebar says why. A linked action without blocks gets a local copy on the first block.

## Paths

A rig can follow a Curve object over its whole timeline: set it in the sidebar's Path field (or draw a Grease Pencil
stroke and press the button next to it, which turns the last stroke into a curve). The root starts at the curve's
start and follows it; the body keeps the prompt's facing relative to the direction of travel, so a sidestep stays a
sidestep. Attaching the path, editing the curve so its length changes, or editing the blocks (text, add, remove,
resize, re-roll) fits the walking blocks' lengths to it at their own gait, in the same undo step (idle blocks keep
theirs, locked blocks too). A length you set by hand wins: resizing a block locks it, so the fit keeps that length
(amber if the path no longer fits it), and unlocking gives it back to the fit. Any other curve edit, or moving the rig, only re-applies the warp, about 20-80 ms after
0.3 s of quiet. A strip under the blocks turns amber where the path asks more than the motion follows (turning about
45 deg/s more than the motion itself turns, or more than 10 % off the block's gait); hover it for why. This works best on
gentle curves at a walk: corners, zigzags and small circles slide the feet. Baked as One Action per Block with root
motion, each action follows its own part of the curve, starting at the origin.

An edit whose take is made in the background (a typed Seconds value, a big timeline, several takes) is still one
Cmd+Z: the step that waits for the take is passed through on undo and redo, and nothing is made by an undo.

## Continue from your keys, fill a gap

When the first block starts where the rig's own keys end (within two frames: your animation, mocap, or a baked
take), its motion continues them: the last 2 s of your keys are read back to the model's 22 joints, encoded and
passed to the model as a block it keeps but never writes. A block whose end meets your keys again leads into them.
Your keys stay as they are; the take is written only between them, eases into your pose and its change per frame
at each seam over 0.4 s, and is placed where your keys are. A notch on the block's edge in the lane shows it; the
block's "Continue from Keys" switch turns it off. It needs the hips and legs mapped, and is off with Loop or a path.
Editing your keys next to the block makes it dirty (with Auto: a new take in its own undo step).

To fill a gap: select keys on both sides of it in the Dope Sheet, then Key > SprongMotion > Fill Gap (also in the
right-click menu). A block (several for a long gap) fills it exactly in one undo step; the Text in the redo panel
says what happens. Adding a block with the playhead in a gap, or Shift-dragging a lone block into one, also fills
it. If the keys after are further from where the motion arrives than it can absorb (0.25 m per second of gap plus
a tenth of the travel, or 30 deg/s of turning), the difference is still blended linearly into the gap's root, and
the block warns (amber notch, a note in the sidebar).

Regenerate a range: Alt+drag over part of a block in the lane, or set the preview range (Ctrl+Alt+P) and use
Regenerate Preview Range in the block's menu. Only that range is made again; the block's other codes stay exactly.

## Typing help

While you type in a block, the rest of a matching sentence shows in grey after the caret: Tab takes it, Ctrl/Cmd+Right
its next word, Esc hides it, typing on goes past it. With no sentence to finish, a pause of 0.3 s lists up to five known
words that finish the word you type (Up/Down, Tab or Enter). A popup text field cannot draw grey text, so its list holds
the sentence first, then the words. The selected block's "Not understood" note names known words spelled alike. Preferences >
Typing Help turns all of it off. `sentences.txt` is our own list of prompts; `words.txt` is the model's vocabulary
without word endings and fragments (macOS spell checker and a dictionary with plain inflections, plus a hand check),
intersected with the loaded model's vocabulary at load.
