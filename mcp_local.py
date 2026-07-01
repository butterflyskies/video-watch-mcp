"""
Video Watch - Local MCP Server
Runs locally — no Modal, no cloud. Video requests come from your home IP.

Same three tools as the cloud version:
- video_listen: Transcript only (lightweight, no download needed for YouTube)
- video_see: Frames only (visual content)
- watch_video: Both (full experience)

No GPU required — uses youtube-transcript-api and yt-dlp subtitles instead of Whisper.

Install:
    pip install 'mcp[cli]' youtube-transcript-api yt-dlp
    # Also needs: ffmpeg (system package)

Run:
    python mcp_local.py
"""

import base64
import json
import logging
import re
import subprocess
import tempfile
from pathlib import Path

from mcp.server.fastmcp import FastMCP, Image

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("video-watch-local")

mcp = FastMCP("video-watch-local")


# ---------------------------------------------------------------------------
# Helpers — extracted from mcp_remote.py, stripped of Modal
# ---------------------------------------------------------------------------


def download_video(url: str, video_path: str) -> dict:
    """Download video via yt-dlp with browser impersonation."""
    cmd = [
        "yt-dlp",
        "-f", "best[height<=720]/best",
        "-o", video_path,
        "--no-playlist",
    ]

    # Only add --impersonate if curl_cffi is available
    try:
        import curl_cffi  # noqa: F401
        cmd += ["--impersonate", "chrome"]
    except ImportError:
        pass

    # YouTube-specific extractor args
    if extract_youtube_id(url):
        cmd += ["--extractor-args", "youtube:player_client=android_creator,mediaconnect"]

    cmd.append(url)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return {"success": False, "error": result.stderr}
    return {"success": True}


def extract_youtube_id(url: str) -> str | None:
    """Extract YouTube video ID from URL, or None if not YouTube."""
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/shorts/)([a-zA-Z0-9_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def fetch_transcript(url: str) -> dict:
    """Get transcript without downloading the video.

    Strategy:
    1. YouTube → youtube-transcript-api (fast, no download)
    2. Any platform → yt-dlp subtitle extraction (--skip-download)
    """
    # --- YouTube transcript API (preferred for YouTube) ---
    video_id = extract_youtube_id(url)
    if video_id:
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            ytt_api = YouTubeTranscriptApi()
            transcript = ytt_api.fetch(video_id)
            snippets = list(transcript)
            full_text = " ".join(snippet.text for snippet in snippets)
            duration = snippets[-1].start + snippets[-1].duration if snippets else 0
            return {
                "success": True,
                "transcript": full_text,
                "duration_seconds": duration,
                "method": "youtube-transcript-api",
            }
        except Exception as e:
            log.info(f"YouTube transcript API failed: {e}")

    # --- yt-dlp subtitle extraction (works for many platforms) ---
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            sub_base = f"{tmpdir}/subs"
            cmd = [
                "yt-dlp",
                "--write-subs", "--write-auto-subs",
                "--sub-lang", "en",
                "--skip-download",
                "-o", sub_base,
            ]
            try:
                import curl_cffi  # noqa: F401
                cmd += ["--impersonate", "chrome"]
            except ImportError:
                pass
            cmd.append(url)

            subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            # yt-dlp writes subs as subs.en.vtt, subs.en.srv3, etc.
            sub_files = sorted(Path(tmpdir).glob("subs*"))
            for sf in sub_files:
                if sf.suffix in (".vtt", ".srv3", ".json3", ".srt", ".ass"):
                    text = sf.read_text(errors="replace")
                    # Strip VTT/SRT formatting cruft for readability
                    if sf.suffix == ".vtt":
                        text = _clean_vtt(text)
                    return {
                        "success": True,
                        "transcript": text,
                        "duration_seconds": 0,
                        "method": "yt-dlp-subtitles",
                    }
    except Exception as e:
        log.info(f"yt-dlp subtitle extraction failed: {e}")

    return {"success": False, "error": "No transcript available for this video"}


def _clean_vtt(text: str) -> str:
    """Strip VTT headers and timestamps, return plain text."""
    lines = text.splitlines()
    out = []
    for line in lines:
        line = line.strip()
        # Skip VTT header, blank lines, timestamp lines, NOTE lines
        if not line or line.startswith("WEBVTT") or line.startswith("Kind:") or line.startswith("Language:"):
            continue
        if line.startswith("NOTE"):
            continue
        if re.match(r"^\d{2}:\d{2}", line) and "-->" in line:
            continue
        if re.match(r"^\d+$", line):  # cue index numbers
            continue
        # Strip inline tags like <c> </c> <00:00:01.234>
        line = re.sub(r"<[^>]+>", "", line)
        if line and (not out or line != out[-1]):  # deduplicate consecutive identical lines
            out.append(line)
    return " ".join(out)


def get_duration(video_path: str) -> float:
    """Get video duration in seconds via ffprobe."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", video_path],
        capture_output=True, text=True,
    )
    try:
        return float(json.loads(probe.stdout)["format"]["duration"])
    except Exception:
        return 0


def extract_frames(video_path: str, output_dir: str, fps: float = 0.5, max_frames: int = 5) -> list[str]:
    """Extract frames from video, return as base64 JPEG list."""
    output_pattern = f"{output_dir}/frame_%04d.jpg"

    # Try with timestamp overlay first, fall back to plain if drawtext unavailable
    cmd_with_text = [
        "ffmpeg", "-i", video_path,
        "-vf", f"fps={fps},scale=480:-1,drawtext=text='%{{pts\\:hms}}':x=10:y=10:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.5",
        "-q:v", "6",
        output_pattern,
    ]
    result = subprocess.run(cmd_with_text, capture_output=True)

    if result.returncode != 0:
        # Drawtext filter might not be available — fall back to plain frames
        log.info("drawtext filter failed, extracting frames without timestamps")
        cmd_plain = [
            "ffmpeg", "-i", video_path,
            "-vf", f"fps={fps},scale=480:-1",
            "-q:v", "6",
            output_pattern,
        ]
        subprocess.run(cmd_plain, capture_output=True, check=True)

    frame_paths = sorted(Path(output_dir).glob("frame_*.jpg"))

    if len(frame_paths) > max_frames:
        step = len(frame_paths) / max_frames
        frame_paths = [frame_paths[int(i * step)] for i in range(max_frames)]

    frames_b64 = []
    for fp in frame_paths:
        with open(fp, "rb") as f:
            frames_b64.append(base64.b64encode(f.read()).decode("utf-8"))

    return frames_b64


def format_duration(seconds: float) -> str:
    """Format seconds as m:ss."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def video_listen(url: str) -> str:
    """Get transcript of a video. Lightweight — no video download needed for YouTube.

    Best for: talking head videos, podcasts, commentary, interviews, tutorials with narration.
    """
    result = fetch_transcript(url)
    if result["success"]:
        duration = result.get("duration_seconds", 0)
        dur_str = format_duration(duration) if duration else "unknown"
        method = result.get("method", "")
        note = f"\n*(via {method})*" if method else ""
        return (
            f"**Video:** {url}\n"
            f"**Duration:** {dur_str}{note}\n\n"
            f"**Transcript:**\n{result['transcript']}"
        )

    return (
        f"**Video:** {url}\n\n"
        f"Could not extract transcript: {result.get('error', 'unknown error')}\n\n"
        f"Tip: YouTube videos usually have transcripts available. "
        f"For other platforms, try `video_see` for visual content."
    )


@mcp.tool()
def video_see(url: str, max_frames: int = 5) -> list:
    """Get visual frames from a video. Returns key frames as images, no transcript.

    Best for: dance videos, visual art, scenery, silent clips, memes, anything where visuals matter more than audio.
    """
    max_frames = min(max_frames, 10)

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = f"{tmpdir}/video.mp4"
        frames_dir = f"{tmpdir}/frames"
        Path(frames_dir).mkdir()

        dl = download_video(url, video_path)
        if not dl["success"]:
            return [f"Error downloading video: {dl['error']}"]

        duration = get_duration(video_path)
        frames = extract_frames(video_path, frames_dir, fps=0.5, max_frames=max_frames)

        content: list = [
            f"**Video:** {url}\n"
            f"**Duration:** {format_duration(duration)}\n"
            f"**Frames:** {len(frames)}"
        ]
        for frame_b64 in frames:
            content.append(Image(data=base64.b64decode(frame_b64), format="jpeg"))

        return content


@mcp.tool()
def watch_video(url: str, max_frames: int = 5) -> list:
    """Full video experience — frames AND transcript. Uses more context but gives the complete picture.

    Use when both visuals and audio matter.
    """
    max_frames = min(max_frames, 10)

    # Fetch transcript first (may not need a full download for YouTube)
    transcript_result = fetch_transcript(url)
    transcript = transcript_result.get("transcript", "[No transcript available]") if transcript_result["success"] else "[No transcript available]"
    transcript_note = ""
    if transcript_result["success"]:
        method = transcript_result.get("method", "")
        if method:
            transcript_note = f" *(via {method})*"

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = f"{tmpdir}/video.mp4"
        frames_dir = f"{tmpdir}/frames"
        Path(frames_dir).mkdir()

        dl = download_video(url, video_path)
        if not dl["success"]:
            if transcript_result["success"]:
                # Have transcript but no video — return what we have
                duration = transcript_result.get("duration_seconds", 0)
                dur_str = format_duration(duration) if duration else "unknown"
                return [
                    f"**Video:** {url}\n"
                    f"**Duration:** {dur_str}\n"
                    f"**Frames:** 0\n"
                    f"**Note:** Video download failed — transcript only{transcript_note}\n\n"
                    f"**Transcript:**\n{transcript}"
                ]
            return [
                f"Error: Download failed: {dl['error']}\n"
                f"Transcript also unavailable: {transcript_result.get('error', 'unknown')}"
            ]

        duration = get_duration(video_path)
        frames = extract_frames(video_path, frames_dir, fps=0.5, max_frames=max_frames)

        content: list = [
            f"**Video:** {url}\n"
            f"**Duration:** {format_duration(duration)}\n"
            f"**Frames:** {len(frames)}{transcript_note}\n\n"
            f"**Transcript:**\n{transcript}"
        ]
        for frame_b64 in frames:
            content.append(Image(data=base64.b64decode(frame_b64), format="jpeg"))

        return content


if __name__ == "__main__":
    mcp.run()
