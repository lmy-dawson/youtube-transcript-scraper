"""Caption parsing and the output formats built from it."""
import json

from src import transcript

JSON3 = json.dumps({"events": [
    {"tStartMs": 0, "dDurationMs": 1500, "segs": [{"utf8": "Hello "}, {"utf8": "world"}]},
    {"tStartMs": 1500, "dDurationMs": 2000, "segs": [{"utf8": "\n"}]},   # blank: skipped
    {"tStartMs": 3500, "dDurationMs": 1000},                              # no runs: skipped
    {"tStartMs": 3725, "dDurationMs": 2275, "segs": [{"utf8": "it&amp;#39;s\nfine"}]},
]})
SRV3 = '<timedtext format="3"><body><p t="1000" d="500">Hi <s>there</s></p></body></timedtext>'
LEGACY = '<transcript><text start="2.5" dur="1.25">it&amp;amp;#39;s legacy</text></transcript>'


def test_json3_skips_empty_events_and_keeps_millisecond_timing():
    segs = transcript.parse(JSON3)
    assert [s["text"] for s in segs] == ["Hello world", "it's fine"]
    assert segs[1]["start"] == 3.725 and segs[1]["end"] == 6.0


def test_srv3_joins_inline_runs():
    assert transcript.parse(SRV3) == [
        {"start": 1.0, "duration": 0.5, "end": 1.5, "text": "Hi there"}]


def test_legacy_format_is_double_unescaped():
    (seg,) = transcript.parse(LEGACY)
    assert seg["text"] == "it's legacy" and seg["end"] == 3.75


def test_garbage_and_empty_payloads_return_nothing():
    for payload in ("", "   ", "{not json", "<broken", None):
        assert transcript.parse(payload) == []


def test_srt_uses_comma_milliseconds_and_numbers_blocks():
    srt = transcript.to_srt(transcript.parse(JSON3))
    assert srt.startswith("1\n00:00:00,000 --> 00:00:01,500\nHello world\n")
    assert "2\n00:00:03,725 --> 00:00:06,000\nit's fine" in srt


def test_vtt_has_header_and_dot_milliseconds():
    vtt = transcript.to_vtt(transcript.parse(SRV3))
    assert vtt.startswith("WEBVTT\n")
    assert "00:00:01.000 --> 00:00:01.500\nHi there" in vtt


def test_clock_carries_rounding_into_the_next_second():
    assert transcript._clock(59.9996, comma=True) == "00:01:00,000"
    assert transcript._clock(3661.25, comma=False) == "01:01:01.250"


def _segs(n, text="abcd"):
    return [{"start": float(i), "end": i + 1.0, "duration": 1.0, "text": text}
            for i in range(n)]


def test_chunks_respect_size_and_keep_real_timestamps():
    chunks = transcript.to_chunks(_segs(6), size=10, overlap=0)
    assert [c["text"] for c in chunks] == ["abcd abcd"] * 3
    assert (chunks[1]["start"], chunks[1]["end"]) == (2.0, 4.0)
    assert [c["index"] for c in chunks] == [0, 1, 2]


def test_chunks_carry_overlap_into_the_next_chunk():
    chunks = transcript.to_chunks(_segs(4), size=10, overlap=5)
    assert chunks[1]["start"] == chunks[0]["end"] - 1.0, "last segment repeats as context"


def test_chunks_guard_bad_input():
    assert transcript.to_chunks([], 100, 10) == []
    assert transcript.to_chunks(_segs(3), 0, 0) == []
