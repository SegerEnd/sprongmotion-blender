"""Typing help for a block's text: the rest of a sentence from a fixed list (ghost text), ranked by the model's text
encoder, and the word at the caret completed from the words the model knows.
"""

import difflib
import pathlib
import re
from collections import Counter

import numpy as np

MIN_CHARS = 6  # no ghost for less typed text
MIN_LETTERS = 2  # of the word at the caret before the list
ROWS = 5
KINDS = {"TAB": "tab", "RET": "enter", "NUMPAD_ENTER": "enter", "ESC": "esc", "UP_ARROW": "up", "DOWN_ARROW": "down"}
_WORD = re.compile(r"[a-z]*$")
_REST = re.compile(r"[a-z]*")


def lines(name):
    return [s for s in (pathlib.Path(__file__).parent / name).read_text().splitlines() if s.strip()]


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


class Help:
    """sentences: whole prompts; words: the known words worth completing to; encode(text) -> the pooled vector."""

    def __init__(self, sentences, words, encode):
        self.sentences, self.encode = sentences, encode
        self.lower = [s.lower() for s in sentences]
        self.words = sorted(words)
        self.known = set(self.words)
        self.count = Counter(w for s in self.lower for w in s.split())
        self.vectors = _unit(np.stack([encode(s) for s in sentences]))
        self.cache = {}

    def sentence(self, text, keep=""):
        """The sentence that best goes on from text, or "". keep: the one shown so far, kept while it still fits."""
        low = text.lower()
        if len(text.strip()) < MIN_CHARS:
            return ""
        if len(keep) > len(text) and keep.lower().startswith(low):
            return keep
        if low not in self.cache:
            if len(self.cache) >= 512:
                self.cache.clear()
            self.cache[low] = self._best(text, low)
        return self.cache[low]

    def _best(self, text, low):
        """The candidate going on with the word most candidates continue with, nearest to text by the encoder."""
        hits = [i for i, s in enumerate(self.lower) if len(s) > len(low) and s.startswith(low)]
        if len(hits) > 1:
            ends = [_REST.match(self.lower[i], len(low)).group() for i in hits]
            end = Counter(ends).most_common(1)[0][0]
            hits, text = [i for i, e in zip(hits, ends, strict=True) if e == end], text + end
        if len(hits) > 1:
            hits = [hits[int(np.argmax(self.vectors[hits] @ _unit(self.encode(text))))]]
        return self.sentences[hits[0]] if hits else ""

    def ghost(self, text, keep=""):
        """What the ghost shows after text: the rest of sentence(text, keep)."""
        return self.sentence(text, keep)[len(text) :]

    def complete(self, text, caret):
        """Known words that finish the partly typed word before the caret, at most ROWS, our sentences' words first."""
        word = _WORD.search(text[:caret].lower()).group()
        if len(word) < MIN_LETTERS or word in self.known or text[caret : caret + 1].isalpha():
            return []
        hits = [w for w in self.words if w.startswith(word) and w != word]
        return sorted(hits, key=lambda w: (-self.count[w], len(w), w))[:ROWS]

    def near(self, word, n=3):
        """Known words spelled like an unknown word."""
        return difflib.get_close_matches(word.lower(), self.words, n, 0.75)


def next_word(ghost):
    """The ghost's leading spaces and first word: what Ctrl/Cmd+Right takes."""
    return re.match(r"\s*\S*", ghost).group()


def insert(text, caret, word):
    """The partly typed word before the caret replaced by word."""
    start = caret - len(_WORD.search(text[:caret].lower()).group())
    return text[:start] + word + text[caret:], start + len(word)


def kind(event):
    if event.type == "RIGHT_ARROW" and (event.ctrl or event.oskey):
        return "word"
    return KINDS.get(event.type)


def take(kind, text, caret, ghost, rows, pick):
    """What a key does while the help shows: (text, caret, pick, hide), None for the text field's own keys."""
    if rows and kind in {"tab", "enter"}:
        return (*insert(text, caret, rows[pick]), 0, False)
    if rows and kind in {"up", "down"}:
        return text, caret, (pick + (1 if kind == "down" else -1)) % len(rows), False
    if ghost and kind in {"tab", "word"}:
        add = ghost if kind == "tab" else next_word(ghost)
        return text + add, caret + len(add), 0, False
    if (ghost or rows) and kind == "esc":
        return text, caret, 0, True
    return None
