"""Split a transcript too long for one read into parts that one read returns whole."""

from typing import NamedTuple

from dreaming.render import _TEXT_LABEL

# The lines a part can spend on its note, section markers and thread header, blank lines included.
RESERVED_LINES = 8
# A part spends at most this fraction of its lines on the earlier conversation.
RECAP_DIVISOR = 5

EVERY_PART_NOTE = (
    "Do not write about parts or this split in memory: cite the session only. Leave any memory claim this part does "
    "not mention, because another part may support it."
)
FIRST_PART_NOTE = (
    "NOTE ABOUT THIS FILE: This is part 1 of {total} of one session, split because it is too long for one read. "
    "Analyze this part only: the later parts are analyzed after it, and can correct what you record. " + EVERY_PART_NOTE
)
LATER_PART_NOTE = (
    "NOTE ABOUT THIS FILE: This is part {number} of {total} of one session, split because it is too long for one "
    "read, and earlier parts have already been analyzed. The section marked as earlier conversation repeats what the "
    "user and the agent said in those parts, so that this part can be understood. It leaves out tool calls and "
    "results, and can leave out older messages to save room. Use it as context only, not as a source of new facts. "
    "If this part contradicts what memory says, fix memory. This part can begin partway through a block. "
    + EVERY_PART_NOTE
)


class Block(NamedTuple):
    label: str
    start: int
    stop: int
    primary: bool  # whether the block is in the primary thread


def _blocks(lines: list[str]) -> list[Block]:
    # The renderer indents any body line that reads like a label, so a line that matches starts a block.
    marks = [(index, match[1]) for index, line in enumerate(lines) if (match := _TEXT_LABEL.match(line))]
    stops = [index for index, _ in marks[1:]] + [len(lines)]
    blocks, primary = [], True
    for (start, label), stop in zip(marks, stops):
        if label == "=== ":
            primary = lines[start].startswith("=== primary")
        blocks.append(Block(label, start, stop, primary))
    return blocks


def _recap(lines: list[str], blocks: list[Block], before: int, budget: int) -> list[str]:
    """What the user and the agent said before a line: the user first, then the newest agent messages that fit."""
    talk = [block for block in blocks if block.primary and block.stop <= before and block.label in ("USER:", "AGENT:")]
    kept, used = set(), 0
    for label in ("USER:", "AGENT:"):
        for block in reversed(talk):
            size = block.stop - block.start
            if block.label == label and used + size <= budget:
                kept.add(block.start)
                used += size
    return [line for block in talk if block.start in kept for line in lines[block.start : block.stop]]


def _cut(blocks: list[Block], start: int, room: int) -> int:
    """Where a part that starts at a line ends: at a user turn if one is near the end, else at any block."""
    earliest = start + room // 2
    in_reach = [block for block in blocks if earliest < block.start <= start + room]
    for wanted in ([block for block in in_reach if block.label == "USER:"], in_reach):
        if wanted:
            return wanted[-1].start
    return start + room  # one block fills the part, so it has to be cut through


def split_transcript(text: str, limit: int) -> list[str]:
    """The transcript as one text, or as parts of at most limit lines when it has more lines than that."""
    lines = text.splitlines()
    if len(lines) <= limit:
        return [text]
    blocks = _blocks(lines)
    first = next((block.start for block in blocks if block.label == "=== "), 0)
    head = lines[:first]

    pieces, start = [], first  # each piece is (recap, first line, line after the last)
    while start < len(lines):
        recap = _recap(lines, blocks, start, limit // RECAP_DIVISOR) if pieces else []
        room = limit - RESERVED_LINES - len(head) - len(recap)
        end = len(lines) if len(lines) - start <= room else _cut(blocks, start, room)
        pieces.append((recap, start, end))
        start = end

    parts = []
    for number, (recap, start, end) in enumerate(pieces, 1):
        if number == 1:
            out = [*head, FIRST_PART_NOTE.format(total=len(pieces)), ""]
        else:
            out = [*head, LATER_PART_NOTE.format(number=number, total=len(pieces)), ""]
            out += ["=== earlier conversation, for context only ===", "", *recap]
            out += [f"=== part {number} starts here ===", ""]
            if not lines[start].startswith("=== "):
                out += [next(line for line in reversed(lines[:start]) if line.startswith("=== ")), ""]
        parts.append("\n".join([*out, *lines[start:end]]) + "\n")
    return parts
