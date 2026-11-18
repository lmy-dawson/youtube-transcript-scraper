"""Turning anything a user pastes into a video ID or a listing to expand."""
import pytest

from src.resolve import classify, video_id_from

VID = "dQw4w9WgXcQ"


@pytest.mark.parametrize("url", [
    VID,
    f"https://www.youtube.com/watch?v={VID}&t=42s",
    f"youtube.com/watch?v={VID}",
    f"https://youtu.be/{VID}?si=abc",
    f"https://m.youtube.com/shorts/{VID}",
    f"https://www.youtube.com/embed/{VID}",
    f"https://www.youtube.com/live/{VID}",
])
def test_every_video_link_shape_resolves(url):
    assert video_id_from(url) == VID


@pytest.mark.parametrize("url", ["", "https://vimeo.com/12345", "https://youtu.be/short", "hello"])
def test_non_videos_return_none(url):
    assert video_id_from(url) is None


@pytest.mark.parametrize("raw,kind,norm", [
    ("@TED", "channel", "https://www.youtube.com/@TED/videos"),
    ("https://www.youtube.com/@TED/featured", "channel", "https://www.youtube.com/@TED/videos"),
    ("https://www.youtube.com/channel/UC123/about", "channel",
     "https://www.youtube.com/channel/UC123/videos"),
    ("https://www.youtube.com/playlist?list=PL9", "playlist",
     "https://www.youtube.com/playlist?list=PL9"),
    ("https://www.youtube.com/results?search_query=python+asyncio", "search", "python asyncio"),
    ("python asyncio", "search", "python asyncio"),
    ("https://example.com/x", "unknown", "https://example.com/x"),
])
def test_classify(raw, kind, norm):
    assert classify(raw) == (kind, norm)


def test_a_video_link_inside_a_playlist_is_treated_as_the_video():
    url = f"https://www.youtube.com/watch?v={VID}&list=PL9"
    assert classify(url)[0] == "video"
