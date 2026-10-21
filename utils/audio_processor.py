import os
import shutil
import logging
import subprocess
from pathlib import Path

import yt_dlp
from pydub import AudioSegment

logger = logging.getLogger(__name__)

DOWNLOAD_DIR = Path("downloads")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)


def find_deno():
    """Find Deno if it is installed and available on PATH."""
    deno_path = shutil.which("deno")

    if not deno_path:
        return None

    try:
        result = subprocess.run(
            [deno_path, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

        if result.returncode == 0:
            logger.info("Deno runtime detected: %s", deno_path)
            return deno_path

    except (OSError, subprocess.SubprocessError):
        logger.exception("Unable to verify Deno runtime")

    return None


def find_ffmpeg():
    """Find FFmpeg from PATH or imageio-ffmpeg."""
    ffmpeg_path = shutil.which("ffmpeg")

    if ffmpeg_path:
        return ffmpeg_path

    try:
        import imageio_ffmpeg

        candidate = imageio_ffmpeg.get_ffmpeg_exe()

        if candidate and Path(candidate).is_file():
            return candidate

    except (ImportError, OSError, RuntimeError):
        logger.exception("Could not locate FFmpeg")

    return None


def download_youtube_audio(url: str) -> str:
    """Download audio from a YouTube URL and return its file path."""
    if not url or not url.strip():
        raise ValueError("Please provide a valid YouTube URL.")

    url = url.strip()
    deno_path = find_deno()
    ffmpeg_path = find_ffmpeg()

    output_template = str(DOWNLOAD_DIR / "%(id)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "noplaylist": True,
        "cachedir": False,
        "retries": 3,
        "fragment_retries": 3,
        "socket_timeout": 30,
        "quiet": False,
        "no_warnings": False,
        "remote_components": ["ejs:github"],
    }

    if deno_path:
        ydl_opts["js_runtimes"] = {
            "deno": {"path": deno_path}
        }
    else:
        logger.warning(
            "Deno is not installed or not on PATH. "
            "YouTube extraction may fail on JavaScript challenges."
        )

    if ffmpeg_path:
        ydl_opts["ffmpeg_location"] = ffmpeg_path

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

            if not info:
                raise RuntimeError(
                    "YouTube did not return video information."
                )

            downloaded_path = Path(ydl.prepare_filename(info))

            if downloaded_path.is_file():
                return str(downloaded_path.resolve())

            # Account for a changed file extension after downloading.
            video_id = info.get("id", "")
            candidates = [
                path
                for path in DOWNLOAD_DIR.glob(f"{video_id}.*")
                if path.is_file()
                and not path.name.endswith((".part", ".ytdl"))
            ]

            if candidates:
                return str(candidates[0].resolve())

            raise FileNotFoundError(
                "yt-dlp finished but the downloaded file was not found."
            )

    except Exception as exc:
        logger.exception("YouTube download failed")
        raise RuntimeError(
            f"Unable to download YouTube audio: {exc}"
        ) from exc


def convert_to_wav(input_path: str) -> str:
    """Convert an audio/video file to mono 16 kHz WAV."""
    source = Path(input_path)

    if not source.is_file():
        raise FileNotFoundError(f"Input file not found: {source}")

    output_path = source.with_name(
        f"{source.stem}_converted.wav"
    )

    ffmpeg_path = find_ffmpeg()

    if ffmpeg_path:
        AudioSegment.converter = ffmpeg_path

    try:
        audio = AudioSegment.from_file(str(source))
        audio = audio.set_channels(1).set_frame_rate(16000)

        audio.export(str(output_path), format="wav")

        return str(output_path.resolve())

    except Exception as exc:
        logger.exception("Audio conversion failed")
        raise RuntimeError(
            "Could not convert the input to WAV. "
            "Check FFmpeg availability and the input format. "
            f"Details: {exc}"
        ) from exc


def chunk_audio(
    wav_path: str,
    chunk_minutes: int = 10,
) -> list[str]:
    """Split WAV audio into chunks of the requested duration."""
    source = Path(wav_path)

    if not source.is_file():
        raise FileNotFoundError(f"WAV file not found: {source}")

    if chunk_minutes <= 0:
        raise ValueError("chunk_minutes must be greater than zero.")

    try:
        audio = AudioSegment.from_wav(str(source))
        chunk_ms = chunk_minutes * 60 * 1000
        chunks = []

        for index, start in enumerate(range(0, len(audio), chunk_ms)):
            chunk = audio[start : start + chunk_ms]

            chunk_path = source.with_name(
                f"{source.stem}_chunk_{index}.wav"
            )

            chunk.export(str(chunk_path), format="wav")
            chunks.append(str(chunk_path.resolve()))

        return chunks

    except Exception as exc:
        logger.exception("Audio chunking failed")
        raise RuntimeError(
            f"Could not split audio into chunks: {exc}"
        ) from exc


def process_input(
    source: str,
    chunk_minutes: int = 10,
) -> list[str]:
    """
    Process a local audio/video file or a URL.
    Returns WAV chunk paths for transcription.
    """
    if not source or not source.strip():
        raise ValueError("Please upload a file or enter a URL.")

    source = source.strip()

    if source.startswith(("https://", "http://")):
        logger.info("Detected URL. Downloading audio...")
        input_path = download_youtube_audio(source)
    else:
        input_path = source

        if not Path(input_path).is_file():
            raise FileNotFoundError(
                f"Input file does not exist: {input_path}"
            )

    logger.info("Converting input to WAV...")
    wav_path = convert_to_wav(input_path)

    logger.info(
        "Splitting audio into %s-minute chunks...",
        chunk_minutes,
    )

    chunks = chunk_audio(wav_path, chunk_minutes)

    logger.info("Created %s audio chunks.", len(chunks))
    return chunks