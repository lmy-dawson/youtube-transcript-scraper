"""
Turning whatever the user pasted into a list of video IDs.

People paste channel links, playlist links, Shorts links, `youtu.be` links,
bare IDs and `@handles`, and a tool that only accepts one of those shapes gets
returned to the store. Everything here answers the same question: which videos
did they mean?
"""

from __future__ import annotations

from urllib.parse import parse_qs, urlparse

import httpx

from . import innertube

# YouTube IDs are exactly 11 characters of URL-safe base64.
_ID_LEN = 11
_ID_CHARS = set(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
)

# Paths that carry the video ID as the segment straight after the marker.
_ID_PATHS = ("shorts", "embed", "live", "v")

_PAGE_SIZE = 30  # what YouTube returns per listing page


def looks_like_id(value: str) -> bool:
    return len(value) == _ID_LEN and all(c in _ID_CHARS for c in value)


def video_id_from(url: str) -> str | None:
    """Extract a single video ID from a URL, or None if it is not a video link."""
    text = (url or "").strip()
    if not text:
        return None
    if looks_like_id(text):
        return text

    if "://" not in text:
        text = "https://" + text
    parsed = urlparse(text)
    host = parsed.netloc.lower().removeprefix("www.").removeprefix("m.")

    if host == "youtu.be":
        candidate = parsed.path.lstrip("/").split("/")[0]
        return candidate if looks_like_id(candidate) else None

    if "youtube.com" not in host:
        return None

    query = parse_qs(parsed.query)
    if "v" in query and looks_like_id(query["v"][0]):
        return query["v"][0]

    parts = [p for p in parsed.path.split("/") if p]
    for marker in _ID_PATHS:
        if marker in parts:
            index = parts.index(marker)
            if index + 1 < len(parts) and looks_like_id(parts[index + 1]):
                return parts[index + 1]
    return None


def classify(url: str) -> tuple[str, str]:
    """Return (kind, normalised url) for one input string.

    kind is one of: video, playlist, channel, search, unknown.
    """
    text = (url or "").strip()
    if not text:
        return "unknown", text

    if video_id_from(text):
        return "video", text

    if text.startswith("@"):
        return "channel", f"https://www.youtube.com/{text}/videos"

    if "://" not in text:
        # A bare word that is not an ID or a handle is a search term.
        if "." not in text:
            return "search", text
        text = "https://" + text

    parsed = urlparse(text)
    host = parsed.netloc.lower().removeprefix("www.").removeprefix("m.")
    if "youtube.com" not in host and host != "youtu.be":
        return "unknown", text

    query = parse_qs(parsed.query)
    parts = [p for p in parsed.path.split("/") if p]

    if "list" in query:
        return "playlist", f"https://www.youtube.com/playlist?list={query['list'][0]}"
    if parts and parts[0] == "playlist":
        return "playlist", text
    if parts and parts[0] == "results":
        return "search", (query.get("search_query") or [""])[0]

    if parts and (parts[0].startswith("@") or parts[0] in ("channel", "c", "user")):
        # Point every channel form at the Videos tab so the listing is videos,
        # not the channel's curated home page.
        base = "/".join(parts[:1] if parts[0].startswith("@") else parts[:2])
        return "channel", f"https://www.youtube.com/{base}/videos"

    return "unknown", text


def _walk(node, key: str) -> list:
    """Collect every value stored under `key`, in the order it appears.

    Document order is not cosmetic here. A channel page carries several
    continuation tokens — the video grid's, plus ones belonging to side
    shelves — and only the first is the "next page of videos". Order also
    decides whether the caller gets a channel's videos newest-first, the way
    the page presents them. Children are therefore pushed reversed so that
    popping walks the structure front to back.
    """
    found: list = []
    stack = [node]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            value = current.get(key)
            if isinstance(value, str):
                found.append(value)
            stack.extend(reversed(list(current.values())))
        elif isinstance(current, list):
            stack.extend(reversed(current))
    return found


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    return [v for v in values if not (v in seen or seen.add(v))]


def _video_ids(payload) -> list[str]:
    return _unique([v for v in _walk(payload, "videoId") if looks_like_id(v)])


def _continuation(payload) -> str:
    tokens = _unique(_walk(payload, "token"))
    return tokens[0] if tokens else ""


async def expand(
    http: httpx.AsyncClient, kind: str, target: str, limit: int
) -> list[str]:
    """Resolve a channel, playlist or search into video IDs, paging as needed."""
    if limit <= 0:
        return []

    if kind == "search":
        context, _ = await innertube.fetch_watch_page(http, "https://www.youtube.com/")
        payload = await innertube.search_videos(http, context, target)
        is_search = True
    else:
        context, payload = await innertube.fetch_watch_page(http, target)
        is_search = False

    ids = _video_ids(payload)
    token = _continuation(payload)

    # Guard against a listing that keeps handing back a token but no new IDs,
    # which would otherwise spin until the run times out.
    while token and len(ids) < limit:
        page = await innertube.continue_listing(http, context, token, search=is_search)
        fresh = _video_ids(page)
        if not fresh:
            break
        before = len(ids)
        ids = _unique(ids + fresh)
        if len(ids) == before:
            break
        token = _continuation(page)

    return ids[:limit]
