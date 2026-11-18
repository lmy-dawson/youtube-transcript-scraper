"""
YouTube Transcript Scraper.

Give it video, channel, playlist or search URLs and it returns the transcript
of every video behind them, with timings, in whatever format the next step
needs: plain text, timestamped segments, SRT, WebVTT, or overlapping chunks
ready to embed.

Nothing here needs an API key, a login cookie or a headless browser. A proxy
is required on the platform and is configured by default.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import httpx
from apify import Actor

from . import innertube, resolve, session, transcript

# Billing is done by the platform, not by this code. The Actor is priced on
# Apify's built-in events: a fixed fee when a run starts, and one charge for
# every item written to the default dataset. The default dataset is therefore
# the bill, and the rule this file keeps is that only a delivered transcript is
# ever pushed there. Videos without one go to the key-value store instead, where
# they cost the user nothing, which is what the README promises.
FAILED_VIDEOS_KEY = "VIDEOS_WITHOUT_TRANSCRIPT"


def _int(value, default=0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _track_name(track: dict) -> str:
    name = track.get("name") or {}
    if "simpleText" in name:
        return name["simpleText"]
    return "".join(run.get("text", "") for run in name.get("runs") or [])


def pick_track(tracks: list[dict], wanted: list[str], prefer_manual: bool) -> dict | None:
    """Choose the caption track that best matches the requested languages.

    Two things make this fiddly. YouTube labels the same language
    inconsistently — `en`, `en-US`, `en-GB` are all "English" to a user asking
    for `en` — so a request matches on the language prefix as well as the exact
    code. And auto-generated captions are markedly worse than human ones.

    Those two rules interact: a video can carry `en` (auto-generated) alongside
    `en-GB` (written by a human). Treating the exact code as the stronger
    signal would hand back the auto track and quietly ignore the caller's
    stated preference, so within one requested language every regional variant
    is pooled first and the manual/auto preference is applied across the pool.
    Exact-code matches still lead the pool, and the sort is stable, so they win
    every tie that the manual preference does not decide.
    """
    if not tracks:
        return None

    def is_auto(track: dict) -> bool:
        return track.get("kind") == "asr"

    def best(candidates: list[dict]) -> dict:
        return sorted(candidates, key=is_auto)[0] if prefer_manual else candidates[0]

    for language in wanted:
        code = language.strip().lower()
        if not code or code == "any":
            continue
        base = code.split("-")[0]
        exact = [t for t in tracks if (t.get("languageCode") or "").lower() == code]
        related = [
            t
            for t in tracks
            if (t.get("languageCode") or "").lower().split("-")[0] == base
            and t not in exact
        ]
        pool = exact + related
        if pool:
            return best(pool)

    # Either "any" was asked for, or none of the preferred languages exist.
    # Returning some transcript beats returning none.
    if not wanted or "any" in [w.strip().lower() for w in wanted]:
        return best(tracks)
    return None


class Options:
    """The user's input, read once."""

    def __init__(self, cfg: dict):
        raw = cfg.get("languages") or ["en"]
        self.languages = [str(x).strip() for x in raw if str(x).strip()]
        self.prefer_manual = cfg.get("preferManualCaptions", True)
        self.fallback_any = cfg.get("fallbackToAnyLanguage", True)
        self.translate_to = (cfg.get("translateTo") or "").strip()

        self.include_segments = cfg.get("includeTimestampedSegments", True)
        self.include_srt = cfg.get("includeSrt", False)
        self.include_vtt = cfg.get("includeVtt", False)
        self.include_metadata = cfg.get("includeVideoMetadata", True)
        self.include_failed = cfg.get("includeVideosWithoutTranscript", False)

        self.chunk_size = _int(cfg.get("chunkSize"), 0)
        self.chunk_overlap = _int(cfg.get("chunkOverlap"), 0)

        self.max_per_source = max(1, _int(cfg.get("maxVideosPerSource"), 50))
        self.concurrency = min(15, max(1, _int(cfg.get("concurrency"), 5)))
        # How many different proxy addresses to try before giving up on a
        # video. Each retry costs one player call, so this stays modest.
        self.block_retries = min(8, max(1, _int(cfg.get("blockRetries"), 4)))


class Attempt:
    """What one pass at a video produced."""

    def __init__(self, player, client, track, segments, translated):
        self.player = player
        self.client = client
        self.track = track
        self.segments = segments
        self.translated = translated


async def _read_video(
    fetcher: session.Fetcher, video_id: str, opts: Options
) -> Attempt:
    """Fetch the player response and the caption track as one retryable unit.

    These cannot be retried separately. An address that YouTube has decided to
    block answers the caption URL with 200 and a zero-length body — and since
    the track list we just read proves captions exist, an empty body is
    evidence about the address, not about the video. Retrying the download on
    that same address just fails more slowly, which is exactly what it did.

    So the whole pass is repeated from a fresh address, and only two things end
    it early: a video with no usable caption track at all, which no address will
    fix, and a successful read.
    """
    challenged = False

    for attempt in range(opts.block_retries):
        if attempt:
            await fetcher.rotate()
            await asyncio.sleep(0.5 * attempt)

        try:
            player, client = await innertube.fetch_player(fetcher.client, video_id)
        except innertube.BotChallenge:
            challenged = True
            continue
        challenged = False

        tracklist = (player.get("captions") or {}).get(
            "playerCaptionsTracklistRenderer"
        ) or {}
        tracks = tracklist.get("captionTracks") or []

        wanted = list(opts.languages)
        if opts.fallback_any:
            wanted.append("any")
        track = pick_track(tracks, wanted, opts.prefer_manual)
        if track is None:
            return Attempt(player, client, None, [], False)

        payload = await innertube.fetch_caption_text(
            fetcher.client, track["baseUrl"], client, opts.translate_to
        )
        segments = transcript.parse(payload)
        translated = bool(opts.translate_to)

        if not segments and opts.translate_to:
            # Machine translation is rate-limited far harder than the source
            # track, so falling back to the original beats returning nothing.
            Actor.log.info(f"{video_id}: translation unavailable, using original")
            payload = await innertube.fetch_caption_text(
                fetcher.client, track["baseUrl"], client
            )
            segments = transcript.parse(payload)
            translated = False

        if segments:
            return Attempt(player, client, track, segments, translated)

    if challenged:
        raise innertube.BotChallenge("YouTube challenged this request")
    raise innertube.Blocked("Caption download returned no text")


async def scrape_video(
    fetcher: session.Fetcher, video_id: str, opts: Options, source: dict
) -> tuple[dict, str]:
    """Fetch and shape one video's transcript.

    Always returns a row plus the outcome that produced it, so the caller can
    both decide whether to keep unsuccessful rows and count why they failed.
    Only an "ok" row is pushed to the default dataset, which is what is billed.
    """
    base = {
        "videoId": video_id,
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "sourceType": source["type"],
        "sourceInput": source["input"],
        "scrapedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    try:
        attempt = await _read_video(fetcher, video_id, opts)
    except innertube.Blocked as exc:
        why = (
            f"{exc}. YouTube served this video's captions as empty from every "
            "address tried. Switching the proxy configuration to residential "
            "usually fixes it."
        )
        Actor.log.warning(f"{video_id}: {why}")
        return {**base, "error": why, "transcript": None}, "blocked"
    except innertube.BotChallenge:
        why = (
            "YouTube blocked this request as automated. Switch the proxy "
            "configuration to residential, or lower the concurrency."
        )
        Actor.log.warning(f"{video_id}: {why}")
        return {**base, "error": why, "transcript": None}, "blocked"
    except innertube.Unavailable as exc:
        Actor.log.warning(f"{video_id}: {exc}")
        return {**base, "error": str(exc), "transcript": None}, "unavailable"
    except Exception as exc:  # noqa: BLE001 - one bad video must not end the run
        Actor.log.warning(f"{video_id}: {exc}")
        return {**base, "error": str(exc), "transcript": None}, "error"

    player = attempt.player
    track = attempt.track
    segments = attempt.segments

    details = player.get("videoDetails") or {}
    if opts.include_metadata:
        thumbnails = (details.get("thumbnail") or {}).get("thumbnails") or []
        base.update(
            {
                "title": details.get("title"),
                "channelName": details.get("author"),
                "channelId": details.get("channelId"),
                "channelUrl": (
                    f"https://www.youtube.com/channel/{details['channelId']}"
                    if details.get("channelId")
                    else None
                ),
                "durationSeconds": _int(details.get("lengthSeconds"), 0) or None,
                "viewCount": _int(details.get("viewCount"), 0) or None,
                "description": details.get("shortDescription"),
                "keywords": details.get("keywords") or [],
                "thumbnailUrl": thumbnails[-1]["url"] if thumbnails else None,
                "isLiveContent": bool(details.get("isLiveContent")),
            }
        )

    tracklist = (player.get("captions") or {}).get(
        "playerCaptionsTracklistRenderer"
    ) or {}
    tracks = tracklist.get("captionTracks") or []
    base["availableLanguages"] = [t.get("languageCode") for t in tracks]

    if track is None:
        why = (
            "This video has no captions"
            if not tracks
            else f"No captions in {', '.join(opts.languages)}"
        )
        Actor.log.info(f"{video_id}: {why}")
        return {**base, "error": why, "transcript": None}, "no_captions"

    text = transcript.to_text(segments)
    base.update(
        {
            "language": opts.translate_to or track.get("languageCode"),
            "languageName": _track_name(track),
            "isAutoGenerated": track.get("kind") == "asr",
            "isTranslated": attempt.translated,
            "transcript": text,
            "wordCount": len(text.split()),
            "characterCount": len(text),
            "segmentCount": len(segments),
        }
    )
    if opts.include_segments:
        base["segments"] = segments
    if opts.include_srt:
        base["srt"] = transcript.to_srt(segments)
    if opts.include_vtt:
        base["vtt"] = transcript.to_vtt(segments)
    if opts.chunk_size > 0:
        base["chunks"] = transcript.to_chunks(
            segments, opts.chunk_size, opts.chunk_overlap
        )

    base["error"] = None
    return base, "ok"


# Used only when a run arrives with no videos and no search terms at all —
# typically an API call made without input, or an automated check running the
# Actor bare. Transcribing one known video shows the output format instead of
# ending the run with nothing.
SAMPLE_VIDEO = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


async def collect_targets(
    http: httpx.AsyncClient, cfg: dict, opts: Options
) -> list[tuple[str, dict]]:
    """Resolve every input into a de-duplicated list of (video id, source)."""
    # Search terms are tagged as searches here rather than re-classified later.
    # Classification guesses from the text, and a term like "node.js tutorial"
    # contains a dot, looks like a malformed URL, and would be silently dropped.
    raw: list[tuple[str, str]] = []
    for entry in cfg.get("videoUrls") or []:
        if isinstance(entry, dict):
            entry = entry.get("url", "")
        if str(entry).strip():
            raw.append(("url", str(entry).strip()))
    for query in cfg.get("searchQueries") or []:
        if str(query).strip():
            raw.append(("search", str(query).strip()))

    if not raw:
        Actor.log.info(
            "No video URLs or search terms were given, so this run transcribes "
            f"one sample video to show the output: {SAMPLE_VIDEO}. Add your own "
            "under 'YouTube URLs' or 'Search terms'."
        )
        raw.append(("url", SAMPLE_VIDEO))

    targets: list[tuple[str, dict]] = []
    seen: set[str] = set()

    for origin, item in raw:
        kind, target = ("search", item) if origin == "search" else resolve.classify(item)
        if kind == "unknown":
            Actor.log.warning(f"Not a YouTube video, channel, playlist or search: {item}")
            continue

        if kind == "video":
            ids = [resolve.video_id_from(target)]
        else:
            Actor.log.info(f"Expanding {kind}: {target}")
            try:
                ids = await resolve.expand(http, kind, target, opts.max_per_source)
            except Exception as exc:  # noqa: BLE001
                Actor.log.warning(f"Could not expand {item}: {exc}")
                continue
            Actor.log.info(f"  found {len(ids)} videos")

        source = {"type": kind, "input": item}
        for video_id in ids:
            if video_id and video_id not in seen:
                seen.add(video_id)
                targets.append((video_id, source))

    return targets


async def proxy_configuration(cfg: dict):
    """Resolve the run's proxy configuration, degrading rather than failing.

    YouTube refuses its player API outright from cloud datacenter ranges, which
    is where this Actor runs, so on the platform a proxy is effectively
    required. The default asks for residential proxies, but not every account
    can use them, and the SDK raises ConnectionError at start-up when the
    requested group is unavailable. Dropping straight to a direct connection
    would turn that into a run that returns nothing, so the order is: what the
    input asked for, then Apify's automatic proxy selection (datacenter, which
    delivered 24 of 25 in testing), and only then a direct connection — which
    is also what a local run without proxy access needs.
    """
    requested = cfg.get("proxyConfiguration")
    try:
        # Returns None, without raising, when the input turned proxies off.
        return await Actor.create_proxy_configuration(actor_proxy_input=requested)
    except Exception as exc:  # noqa: BLE001
        Actor.log.warning(f"Requested proxy unavailable ({exc}).")

    if (requested or {}).get("apifyProxyGroups"):
        try:
            fallback = await Actor.create_proxy_configuration(
                actor_proxy_input={"useApifyProxy": True}
            )
        except Exception as exc:  # noqa: BLE001
            Actor.log.warning(f"Automatic proxy selection unavailable too ({exc}).")
        else:
            Actor.log.warning(
                "Falling back to Apify's automatic proxy selection. Expect a few "
                "more blocked videos than with residential proxies."
            )
            return fallback

    Actor.log.warning("Connecting directly; YouTube may block this.")
    return None


async def main() -> None:
    async with Actor:
        cfg = await Actor.get_input() or {}

        opts = Options(cfg)

        configuration = await proxy_configuration(cfg)
        Actor.log.info(
            "Using Apify Proxy."
            if configuration
            else "No proxy configured — connecting directly."
        )

        # Listings (channel, playlist, search) are ordinary page reads that
        # YouTube still serves to datacenter addresses, so they go direct and
        # leave the proxy for the calls that genuinely need it.
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=15.0), follow_redirects=True
        ) as browser:
            targets = await collect_targets(browser, cfg, opts)

        if not targets:
            Actor.log.warning(
                "Nothing to scrape. Add a YouTube video, channel, playlist "
                "or search URL to videoUrls, or a term to searchQueries."
            )
            return

        total = len(targets)
        Actor.log.info(f"Scraping {total} videos, {opts.concurrency} at a time...")

        done = 0
        delivered = 0
        blocked = 0
        unfinished: list[dict] = []
        progress = asyncio.Lock()

        async with session.Pool(configuration, opts.concurrency) as pool:
            # The queue is both the concurrency limit and the worker's proxy
            # address: taking a fetcher is taking a slot, so no separate
            # semaphore is needed and two workers never share an address.
            free: asyncio.Queue[session.Fetcher] = asyncio.Queue()
            for fetcher in pool.fetchers:
                free.put_nowait(fetcher)

            async def worker(video_id: str, source: dict) -> None:
                nonlocal done, delivered, blocked
                fetcher = await free.get()
                try:
                    record, outcome = await scrape_video(fetcher, video_id, opts, source)
                finally:
                    free.put_nowait(fetcher)

                if outcome == "ok":
                    await Actor.push_data(record)
                elif opts.include_failed:
                    unfinished.append(record)
                async with progress:
                    done += 1
                    if outcome == "ok":
                        delivered += 1
                    elif outcome == "blocked":
                        blocked += 1
                    if done % 25 == 0 or done == total:
                        Actor.log.info(
                            f"  {done}/{total} processed, {delivered} transcripts"
                        )

            await asyncio.gather(
                *(worker(video_id, source) for video_id, source in targets)
            )

        Actor.log.info(f"Finished. {delivered} transcripts from {total} videos.")
        if unfinished:
            await Actor.set_value(FAILED_VIDEOS_KEY, unfinished)
            Actor.log.info(
                f"{len(unfinished)} videos had no transcript and were not billed. "
                f"Each is listed with its reason in the key-value store record "
                f"{FAILED_VIDEOS_KEY}."
            )
        if blocked:
            Actor.log.warning(
                f"{blocked} videos were blocked as automated. Set the proxy "
                "configuration to residential proxies for a much higher success "
                "rate on large runs."
            )
        if delivered == 0 and not blocked:
            Actor.log.warning(
                "No transcripts found. The videos may have captions disabled, "
                "or none match your language filter — try enabling "
                "'fallbackToAnyLanguage'."
            )
