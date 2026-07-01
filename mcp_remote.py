"""
Video Watch - Remote MCP Server on Modal
Fully cloud-hosted MCP server that lets Claude "watch" videos.

Three tools:
- video_listen: Transcript only (lightweight)
- video_see: Frames only (visual content)
- watch_video: Both (full experience)
"""

import modal
import base64
import json
import re
import subprocess
import tempfile
from pathlib import Path

# Image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "curl")
    .run_commands(
        "curl -fsSL https://deno.land/install.sh | sh",
        "ln -s /root/.deno/bin/deno /usr/local/bin/deno",
    )
    .pip_install(
        "yt-dlp>=2025.06.09",  # Pin recent version — bump date to bust Modal image cache
        "curl_cffi",  # For browser impersonation (TikTok, Instagram, etc.)
        "brotli",     # For compression support
        "youtube-transcript-api>=1.0.0",  # v1.0+ API (fetch instead of get_transcript)
        "openai-whisper",
        "torch",
        "mcp[cli]",
        "starlette",
        "sse-starlette",
        "uvicorn",
    )
)

app = modal.App("video-watch-mcp", image=image)


def download_video(url: str, video_path: str) -> dict:
    """Download video, return success/error."""
    result = subprocess.run([
        "yt-dlp",
        "-f", "best[height<=720]/best",
        "-o", video_path,
        "--no-playlist",
        "--impersonate", "chrome",
        "--extractor-args", "youtube:player_client=android_creator,mediaconnect",
        url
    ], capture_output=True, text=True)

    if result.returncode != 0:
        return {"success": False, "error": result.stderr}
    return {"success": True}


def extract_youtube_id(url: str) -> str | None:
    """Extract YouTube video ID from URL, or None if not a YouTube URL."""
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/shorts/)([a-zA-Z0-9_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def fetch_transcript_fallback(url: str) -> dict:
    """Try to get transcript via youtube-transcript-api when yt-dlp fails.
    Only works for YouTube URLs. Returns dict with success/transcript/error."""
    video_id = extract_youtube_id(url)
    if not video_id:
        return {"success": False, "error": "Not a YouTube URL — transcript fallback unavailable"}

    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id)
        snippets = list(transcript)
        full_text = " ".join(snippet.text for snippet in snippets)
        duration = snippets[-1].start + snippets[-1].duration if snippets else 0
        return {"success": True, "transcript": full_text, "duration_seconds": duration}
    except Exception as e:
        return {"success": False, "error": f"Transcript API fallback failed: {e}"}


def get_duration(video_path: str) -> float:
    """Get video duration in seconds."""
    probe = subprocess.run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json", video_path
    ], capture_output=True, text=True)

    try:
        return float(json.loads(probe.stdout)["format"]["duration"])
    except:
        return 0


def extract_frames(video_path: str, output_dir: str, fps: float = 0.5, max_frames: int = 5) -> list[str]:
    """Extract frames from video, return as base64 list."""
    output_pattern = f"{output_dir}/frame_%04d.jpg"

    # Smaller frames (480px width), more compression
    subprocess.run([
        "ffmpeg", "-i", video_path,
        "-vf", f"fps={fps},scale=480:-1,drawtext=text='%{{pts\\:hms}}':x=10:y=10:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.5",
        "-q:v", "6",
        output_pattern
    ], capture_output=True, check=True)

    frame_paths = sorted(Path(output_dir).glob("frame_*.jpg"))

    # Limit frames
    if len(frame_paths) > max_frames:
        step = len(frame_paths) / max_frames
        frame_paths = [frame_paths[int(i * step)] for i in range(max_frames)]

    frames_b64 = []
    for fp in frame_paths:
        with open(fp, "rb") as f:
            frames_b64.append(base64.b64encode(f.read()).decode("utf-8"))

    return frames_b64


def transcribe_audio(video_path: str, audio_path: str) -> str:
    """Extract and transcribe audio."""
    import whisper

    # Extract audio
    subprocess.run([
        "ffmpeg", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        audio_path
    ], capture_output=True, check=True)

    # Transcribe
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["text"]


@app.function(gpu="T4", timeout=300)
def process_listen(url: str):
    """Audio/transcript only - lightweight."""
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = f"{tmpdir}/video.mp4"
        audio_path = f"{tmpdir}/audio.wav"

        dl = download_video(url, video_path)
        if not dl["success"]:
            # Fallback: try youtube-transcript-api (no download needed)
            fallback = fetch_transcript_fallback(url)
            if fallback["success"]:
                return {
                    "success": True,
                    "duration_seconds": fallback.get("duration_seconds", 0),
                    "transcript": fallback["transcript"],
                    "url": url,
                    "note": "Used transcript API fallback (video download failed)"
                }
            return {"success": False, "error": f"Download failed: {dl['error']}\nTranscript fallback also failed: {fallback['error']}"}

        duration = get_duration(video_path)

        try:
            transcript = transcribe_audio(video_path, audio_path)
        except Exception as e:
            transcript = f"[Transcription failed: {e}]"

        return {
            "success": True,
            "duration_seconds": duration,
            "transcript": transcript,
            "url": url
        }


@app.function(timeout=300)
def process_see(url: str, max_frames: int = 5):
    """Frames only - no GPU needed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = f"{tmpdir}/video.mp4"
        frames_dir = f"{tmpdir}/frames"
        Path(frames_dir).mkdir()

        dl = download_video(url, video_path)
        if not dl["success"]:
            return {"success": False, "error": dl["error"]}

        duration = get_duration(video_path)
        frames = extract_frames(video_path, frames_dir, fps=0.5, max_frames=max_frames)

        return {
            "success": True,
            "duration_seconds": duration,
            "frame_count": len(frames),
            "frames": frames,
            "url": url
        }


@app.function(gpu="T4", timeout=300)
def process_watch(url: str, max_frames: int = 5):
    """Full experience - frames + transcript."""
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = f"{tmpdir}/video.mp4"
        audio_path = f"{tmpdir}/audio.wav"
        frames_dir = f"{tmpdir}/frames"
        Path(frames_dir).mkdir()

        dl = download_video(url, video_path)
        if not dl["success"]:
            # Fallback: try transcript-only via API (no frames, but better than nothing)
            fallback = fetch_transcript_fallback(url)
            if fallback["success"]:
                return {
                    "success": True,
                    "duration_seconds": fallback.get("duration_seconds", 0),
                    "frame_count": 0,
                    "frames": [],
                    "transcript": fallback["transcript"],
                    "url": url,
                    "note": "Used transcript API fallback — no frames available (video download failed)"
                }
            return {"success": False, "error": f"Download failed: {dl['error']}\nTranscript fallback also failed: {fallback['error']}"}

        duration = get_duration(video_path)
        frames = extract_frames(video_path, frames_dir, fps=0.5, max_frames=max_frames)

        try:
            transcript = transcribe_audio(video_path, audio_path)
        except Exception as e:
            transcript = f"[Transcription failed: {e}]"

        return {
            "success": True,
            "duration_seconds": duration,
            "frame_count": len(frames),
            "frames": frames,
            "transcript": transcript,
            "url": url
        }


@app.function()
@modal.asgi_app()
def mcp_server():
    """ASGI app that serves MCP over SSE."""
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.responses import JSONResponse
    from sse_starlette.sse import EventSourceResponse

    async def handle_sse(request):
        async def event_generator():
            yield {
                "event": "endpoint",
                "data": json.dumps({
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized"
                })
            }
        return EventSourceResponse(event_generator())

    def format_duration(seconds: float) -> str:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}:{secs:02d}"

    async def handle_mcp(request):
        body = await request.json()
        method = body.get("method", "")
        params = body.get("params", {})
        request_id = body.get("id")

        if method == "initialize":
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "video-watch", "version": "2.0.0"}
                }
            })

        elif method == "tools/list":
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "tools": [
                        {
                            "name": "video_listen",
                            "description": "Get transcript of a video. Lightweight - returns only the spoken/audio content as text. Best for: talking head videos, podcasts, commentary, interviews, tutorials with narration.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "url": {"type": "string", "description": "Video URL"}
                                },
                                "required": ["url"]
                            }
                        },
                        {
                            "name": "video_see",
                            "description": "Get visual frames from a video. Returns key frames as images, no audio transcription. Best for: dance videos, visual art, scenery, silent clips, memes, anything where visuals matter more than audio.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "url": {"type": "string", "description": "Video URL"},
                                    "max_frames": {"type": "integer", "description": "Max frames (default 5)", "default": 5}
                                },
                                "required": ["url"]
                            }
                        },
                        {
                            "name": "watch_video",
                            "description": "Full video experience - frames AND transcript. Uses more context but gives complete picture. Use when both visuals and audio matter.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "url": {"type": "string", "description": "Video URL"},
                                    "max_frames": {"type": "integer", "description": "Max frames (default 5)", "default": 5}
                                },
                                "required": ["url"]
                            }
                        }
                    ]
                }
            })

        elif method == "tools/call":
            tool_name = params.get("name")
            args = params.get("arguments", {})
            url = args.get("url")

            if not url:
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"content": [{"type": "text", "text": "Error: No URL provided"}]}
                })

            # Route to appropriate processor
            if tool_name == "video_listen":
                result = await process_listen.remote.aio(url)

                if not result.get("success"):
                    return JSONResponse({
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {"content": [{"type": "text", "text": f"Error: {result.get('error')}"}]}
                    })

                note = f"\n**Note:** {result['note']}" if result.get("note") else ""
                content = [{
                    "type": "text",
                    "text": f"**Video:** {url}\n**Duration:** {format_duration(result.get('duration_seconds', 0))}{note}\n\n**Transcript:**\n{result.get('transcript', '[No transcript]')}"
                }]

            elif tool_name == "video_see":
                max_frames = min(args.get("max_frames", 5), 10)
                result = await process_see.remote.aio(url, max_frames)

                if not result.get("success"):
                    return JSONResponse({
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {"content": [{"type": "text", "text": f"Error: {result.get('error')}"}]}
                    })

                content = [{
                    "type": "text",
                    "text": f"**Video:** {url}\n**Duration:** {format_duration(result.get('duration_seconds', 0))}\n**Frames:** {result.get('frame_count', 0)}"
                }]

                for frame_b64 in result.get("frames", []):
                    content.append({"type": "image", "data": frame_b64, "mimeType": "image/jpeg"})

            elif tool_name == "watch_video":
                max_frames = min(args.get("max_frames", 5), 10)
                result = await process_watch.remote.aio(url, max_frames)

                if not result.get("success"):
                    return JSONResponse({
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {"content": [{"type": "text", "text": f"Error: {result.get('error')}"}]}
                    })

                note = f"\n**Note:** {result['note']}" if result.get("note") else ""
                content = [{
                    "type": "text",
                    "text": f"**Video:** {url}\n**Duration:** {format_duration(result.get('duration_seconds', 0))}\n**Frames:** {result.get('frame_count', 0)}{note}\n\n**Transcript:**\n{result.get('transcript', '[No transcript]')}"
                }]

                for frame_b64 in result.get("frames", []):
                    content.append({"type": "image", "data": frame_b64, "mimeType": "image/jpeg"})
            else:
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"}
                })

            return JSONResponse({
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"content": content}
            })

        return JSONResponse({
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"}
        })

    async def health(request):
        return JSONResponse({"status": "ok", "service": "video-watch-mcp", "version": "2.0.0"})

    routes = [
        Route("/", handle_mcp, methods=["POST"]),
        Route("/sse", handle_sse),
        Route("/health", health),
    ]

    return Starlette(routes=routes)
