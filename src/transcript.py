"""
Turning YouTube's caption payloads into usable text.

YouTube serves captions in three different shapes depending on which client
asked and whether `fmt` was honoured, so all three are parsed here rather than
assuming the one that happened to come back during development:

  * `json3`   — the modern format: events with millisecond offsets.
  * `srv3`    — `<timedtext format="3">` with `<p>` timing and `<s>` runs.
  * `legacy`  — `<transcript>` with plain `<text start dur>` elements.

Everything downstream works from the parsed segment list, so adding a format
never touches the formatters.
"""

from __future__ import annotations

import html as html_lib
import json
import re
import xml.etree.ElementTree as ET

# YouTube double-escapes legacy caption text: the XML carries `&amp;#39;`,
# which one unescape turns into `&#39;` and a second turns into an apostrophe.
_MAX_UNESCAPE_PASSES = 2


def _tidy(text: str) -> str:
    """Unescape entities and collapse the newlines YouTube uses for line wrap."""
    for _ in range(_MAX_UNESCAPE_PASSES):
        if "&" not in text:
            break
        text = html_lib.unescape(text)
    return re.sub(r"\s+", " ", text.replace("\n", " ")).strip()


def _from_json3(payload: str) -> list[dict]:
    data = json.loads(payload)
    segments = []
    for event in data.get("events") or []:
        runs = event.get("segs")
        if not runs:
            continue
        text = _tidy("".join(run.get("utf8", "") for run in runs))
        if not text:
            continue
        start = event.get("tStartMs", 0) / 1000
        duration = event.get("dDurationMs", 0) / 1000
        segments.append(
            {
                "start": round(start, 3),
                "duration": round(duration, 3),
                "end": round(start + duration, 3),
                "text": text,
            }
        )
    return segments


def _from_srv3(root: ET.Element) -> list[dict]:
    segments = []
    for node in root.iter("p"):
        pieces = [node.text or ""] + [
            (child.text or "") + (child.tail or "") for child in node
        ]
        text = _tidy("".join(pieces))
        if not text:
            continue
        start = int(node.get("t", 0)) / 1000
        duration = int(node.get("d", 0)) / 1000
        segments.append(
            {
                "start": round(start, 3),
                "duration": round(duration, 3),
                "end": round(start + duration, 3),
                "text": text,
            }
        )
    return segments


def _from_legacy(root: ET.Element) -> list[dict]:
    segments = []
    for node in root.iter("text"):
        text = _tidy(node.text or "")
        if not text:
            continue
        start = float(node.get("start", 0))
        duration = float(node.get("dur", 0))
        segments.append(
            {
                "start": round(start, 3),
                "duration": round(duration, 3),
                "end": round(start + duration, 3),
                "text": text,
            }
        )
    return segments


def parse(payload: str) -> list[dict]:
    """Parse any caption payload YouTube returns into timed segments."""
    body = (payload or "").strip()
    if not body:
        return []
    if body.startswith("{"):
        try:
            return _from_json3(body)
        except (ValueError, KeyError):
            return []
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        return []
    # srv3 nests runs inside <p>; the legacy format is flat <text> elements.
    return _from_srv3(root) if root.find(".//p") is not None else _from_legacy(root)


def to_text(segments: list[dict]) -> str:
    return " ".join(s["text"] for s in segments)


def _clock(seconds: float, comma: bool) -> str:
    whole = int(seconds)
    millis = int(round((seconds - whole) * 1000))
    if millis == 1000:  # rounding can carry into the next second
        whole += 1
        millis = 0
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    sep = "," if comma else "."
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{millis:03d}"


def to_srt(segments: list[dict]) -> str:
    """SubRip — what video editors and most caption tools import."""
    blocks = []
    for index, seg in enumerate(segments, 1):
        blocks.append(
            f"{index}\n"
            f"{_clock(seg['start'], True)} --> {_clock(seg['end'], True)}\n"
            f"{seg['text']}\n"
        )
    return "\n".join(blocks)


def to_vtt(segments: list[dict]) -> str:
    """WebVTT — what HTML5 <track> elements need."""
    blocks = ["WEBVTT\n"]
    for seg in segments:
        blocks.append(
            f"{_clock(seg['start'], False)} --> {_clock(seg['end'], False)}\n"
            f"{seg['text']}\n"
        )
    return "\n".join(blocks)


def to_chunks(segments: list[dict], size: int, overlap: int) -> list[dict]:
    """Group segments into overlapping passages for embedding.

    A raw caption segment is a few words long — far too small to embed on its
    own, and splitting on a fixed character count alone would cut sentences in
    half and lose the timestamp. Chunks are therefore built by accumulating
    whole segments up to `size` characters, so every chunk keeps the real start
    and end time of the passage it covers and can be cited back to the video.
    """
    if size <= 0 or not segments:
        return []
    overlap = max(0, min(overlap, size - 1))

    chunks: list[dict] = []
    current: list[dict] = []
    length = 0

    def flush() -> None:
        if not current:
            return
        chunks.append(
            {
                "index": len(chunks),
                "start": current[0]["start"],
                "end": current[-1]["end"],
                "text": " ".join(s["text"] for s in current),
            }
        )

    for segment in segments:
        addition = len(segment["text"]) + 1
        if current and length + addition > size:
            flush()
            if overlap:
                # Walk back from the end of the chunk we just emitted until we
                # have `overlap` characters of context to prepend to the next.
                tail: list[dict] = []
                carried = 0
                for previous in reversed(current):
                    carried += len(previous["text"]) + 1
                    tail.insert(0, previous)
                    if carried >= overlap:
                        break
                current = tail
                length = carried
            else:
                current = []
                length = 0
        current.append(segment)
        length += addition

    flush()
    return chunks
