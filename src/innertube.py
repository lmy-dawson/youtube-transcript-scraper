"""
Talking to YouTube's internal player API.

The public `WEB` client stopped being usable for captions. It still answers
200, but the caption URLs it hands out now return an **empty body** unless the
request carries a proof-of-origin token, which can only be produced by running
YouTube's own obfuscated JavaScript. That single change is why so many
transcript tools now fail intermittently and return blank rows.

Two clients are still exempt from that requirement: the iOS app client and the
Meta Quest ("ANDROID_VR") client. Both hand back complete, signed caption URLs
that download normally from an ordinary datacenter IP, with no token, no
cookies, no browser and no proxy. This module speaks to those two, in that
order, and falls back to the watch page only if both refuse.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
from dataclasses import dataclass, field

import httpx

PLAYER_URL = "https://www.youtube.com/youtubei/v1/player?prettyPrint=false"
BROWSE_URL = "https://www.youtube.com/youtubei/v1/browse?prettyPrint=false"
SEARCH_URL = "https://www.youtube.com/youtubei/v1/search?prettyPrint=false"

WEB_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)


@dataclass(frozen=True)
class Client:
    """One InnerTube client identity."""

    label: str
    client_id: int
    user_agent: str
    context: dict = field(default_factory=dict)

    def payload(self, video_id: str) -> dict:
        return {
            "context": {"client": {**self.context, "hl": "en", "gl": "US"}},
            "videoId": video_id,
            "contentCheckOk": True,
            "racyCheckOk": True,
        }

    def headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "User-Agent": self.user_agent,
            "X-YouTube-Client-Name": str(self.client_id),
            "X-YouTube-Client-Version": self.context["clientVersion"],
            "Origin": "https://www.youtube.com",
        }


IOS = Client(
    label="ios",
    client_id=5,
    user_agent=(
        "com.google.ios.youtube/20.03.02 (iPhone16,2; U; CPU iOS 18_2_1 like Mac OS X)"
    ),
    context={
        "clientName": "IOS",
        "clientVersion": "20.03.02",
        "deviceMake": "Apple",
        "deviceModel": "iPhone16,2",
        "osName": "iPhone",
        "osVersion": "18.2.1.22C161",
    },
)

ANDROID_VR = Client(
    label="android_vr",
    client_id=28,
    user_agent=(
        "com.google.android.apps.youtube.vr.oculus/1.60.19 "
        "(Linux; U; Android 12; GB) gzip"
    ),
    context={
        "clientName": "ANDROID_VR",
        "clientVersion": "1.60.19",
        "deviceMake": "Oculus",
        "deviceModel": "Quest 3",
        "osName": "Android",
        "osVersion": "12",
        "androidSdkVersion": 32,
    },
)

# Order matters: iOS returns the richer caption track list, ANDROID_VR is the
# stand-in for the occasional video iOS declines to serve.
PLAYER_CLIENTS = (IOS, ANDROID_VR)


class Unavailable(Exception):
    """The video ID resolves but cannot be read: private, deleted or blocked."""


class RateLimited(Exception):
    """YouTube asked us to slow down, and kept asking."""


class BotChallenge(Unavailable):
    """YouTube challenged the address we called from, not the video itself."""


class Blocked(Unavailable):
    """Captions exist but every address tried returned an empty body."""


# YouTube's bot challenge wording. Unlike "this video is private", it says
# nothing about the video — the same ID succeeds from a different address a
# second later — so it is retried elsewhere rather than reported to the user.
BOT_CHECK = "not a bot"


def _is_bot_check(reason: str) -> bool:
    return BOT_CHECK in (reason or "").lower()


async def _retrying(call, *, attempts: int = 4, base_delay: float = 1.5):
    """Retry a request on 429 and 5xx with exponential backoff and jitter.

    YouTube starts answering 429 long before it answers anything permanent, so
    backing off politely is the difference between a run that completes and a
    run that returns half its rows blank.
    """
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            response = await call()
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last = exc
        else:
            if response.status_code == 200:
                return response
            if response.status_code in (429, 500, 502, 503, 504):
                last = RateLimited(f"HTTP {response.status_code}")
            else:
                return response
        if attempt < attempts - 1:
            await asyncio.sleep(base_delay * (2**attempt) + random.uniform(0, 0.75))
    raise last or RateLimited("request failed")


async def fetch_player(http: httpx.AsyncClient, video_id: str) -> tuple[dict, Client]:
    """Return the player response for a video, plus the client that obtained it.

    Raises Unavailable carrying YouTube's own wording, so the real reason
    ("This video is private") reaches the user instead of a generic failure.

    A bot challenge is not such a reason: it describes the address we called
    from, not the video, and the same ID succeeds from a different address a
    moment later. It is raised separately as BotChallenge so the caller can
    rotate proxies and try again, which this layer has no way to do itself.
    """
    reason = "This video is unavailable"
    challenged = False

    for client in PLAYER_CLIENTS:
        try:
            response = await _retrying(
                lambda c=client: http.post(
                    PLAYER_URL, json=c.payload(video_id), headers=c.headers()
                )
            )
        except (RateLimited, httpx.HTTPError):
            challenged = True
            continue
        if response.status_code != 200:
            continue

        data = response.json()
        status = (data.get("playabilityStatus") or {}).get("status")
        if status == "OK":
            return data, client

        told = (data.get("playabilityStatus") or {}).get("reason") or ""
        if _is_bot_check(told):
            challenged = True
        elif told:
            reason = told

    if challenged:
        raise BotChallenge("YouTube challenged this request")
    raise Unavailable(reason)


async def fetch_caption_text(
    http: httpx.AsyncClient,
    base_url: str,
    client: Client,
    translate_to: str = "",
    *,
    attempts: int = 2,
) -> str:
    """Download one caption track.

    The signed URL carries an expiry, so this runs immediately after the player
    response rather than being queued for later.

    A blocked address answers 200 with a zero-length body rather than an error
    status, which is precisely the failure that leaves other transcript tools
    emitting rows with an empty transcript field. An empty body is therefore
    treated as a failure worth retrying, not as a video without captions.
    """
    url = base_url + "&fmt=json3"
    if translate_to:
        url += f"&tlang={translate_to}"

    for attempt in range(attempts):
        try:
            response = await _retrying(
                lambda: http.get(url, headers={"User-Agent": client.user_agent})
            )
        except (RateLimited, httpx.HTTPError):
            response = None
        if response is not None and response.status_code == 200 and response.text.strip():
            return response.text
        if attempt < attempts - 1:
            await asyncio.sleep(0.8 * (attempt + 1) + random.uniform(0, 0.4))
    return ""


async def fetch_watch_page(http: httpx.AsyncClient, url: str) -> tuple[dict, dict]:
    """Load a YouTube page and return (innertube context, ytInitialData).

    Used for listings: channels, playlists, search. Those are ordinary page
    reads that the WEB client still serves without complaint. Only the caption
    URLs are gated, which is why listings and transcripts take different paths.
    """
    response = await _retrying(
        lambda: http.get(
            url,
            headers={"User-Agent": WEB_UA, "Accept-Language": "en-US,en;q=0.9"},
            follow_redirects=True,
        )
    )
    html = response.text

    context: dict = {}
    marker = re.search(r'"INNERTUBE_CONTEXT":', html)
    if marker:
        try:
            context, _ = json.JSONDecoder().raw_decode(html, marker.end())
        except ValueError:
            context = {}

    data: dict = {}
    found = re.search(r"ytInitialData\s*=\s*(\{.+?\})\s*;\s*</script>", html, re.S)
    if found:
        try:
            data = json.loads(found.group(1))
        except ValueError:
            data = {}
    return context, data


def _listing_headers(context: dict) -> dict:
    version = (context.get("client") or {}).get("clientVersion", "2.20240726.00.00")
    return {
        "Content-Type": "application/json",
        "User-Agent": WEB_UA,
        "X-YouTube-Client-Name": "1",
        "X-YouTube-Client-Version": version,
        "Origin": "https://www.youtube.com",
    }


async def continue_listing(
    http: httpx.AsyncClient, context: dict, token: str, *, search: bool = False
) -> dict:
    """Fetch the next page of a listing using a continuation token."""
    try:
        response = await _retrying(
            lambda: http.post(
                SEARCH_URL if search else BROWSE_URL,
                json={"context": context, "continuation": token},
                headers=_listing_headers(context),
            )
        )
    except (RateLimited, httpx.HTTPError):
        return {}
    return response.json() if response.status_code == 200 else {}


async def search_videos(http: httpx.AsyncClient, context: dict, query: str) -> dict:
    """Run a YouTube search restricted to videos."""
    try:
        response = await _retrying(
            lambda: http.post(
                SEARCH_URL,
                # EgIQAQ%3D%3D is YouTube's own "type: video" filter. Without it
                # the response is padded with channel, playlist and shelf
                # results that carry video IDs we never asked for.
                json={"context": context, "query": query, "params": "EgIQAQ%3D%3D"},
                headers=_listing_headers(context),
            )
        )
    except (RateLimited, httpx.HTTPError):
        return {}
    return response.json() if response.status_code == 200 else {}
