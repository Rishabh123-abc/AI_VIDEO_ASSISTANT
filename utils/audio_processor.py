import os
import shutil
import logging
from pathlib import Path

import yt_dlp
from pydub import AudioSegment

logger = logging.getLogger(__name__)

# Use a writable directory on Streamlit Cloud
DOWNLOAD_DIR = Path("downloads")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)


def find_ffmpeg():
    """Find FFmpeg executable if available."""
    return shutil.which("ffmpeg")


def download_youtube_audio(url: str) -> str:
    """
    Download audio from a YouTube URL.
    Returns the downloaded audio file path.
    """

    if not url or not url.strip():
        raise ValueError("Please provide a valid YouTube URL.")

    url = url.strip()

    output_template = str(DOWNLOAD_DIR / "%(id)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "noplaylist": True,
        "quiet": False,
        "no_warnings": False,
        "retries": 3,
        "fragment_retries": 3,
        "socket_timeout": 30,
        "cachedir": False,

        # Let yt-dlp use Deno when installed and detected.
        "js_runtimes": {
            "deno": {}
        },

        # Enable yt-dlp's supported YouTube challenge components.
        "remote_components": ["ejs:github"],

        # Avoid pretending a downloaded file is already WAV.
        "postprocessors": [],
    }

    ffmpeg_path = find_ffmpeg()
    if ffmpeg_path:
        ydl_opts["ffmpeg_location"] = ffmpeg_path

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

            if not info:
                raise RuntimeError("YouTube did not return video information.")

            downloaded_path = ydl.prepare_filename(info)
            downloaded_file = Path(downloaded_path)

            # Some formats may use a different extension after downloading.
            if not downloaded_file.exists():
                possible_files = list(
                    DOWNLOAD_DIR.glob(f"{info.get('id', '')}.*")
                )
                possible_files = [
                    p for p in possible_files
                    if p.is_file() and not p.name.endswith(".part")
                ]

                if not possible_files:
                    raise FileNotFoundError(
                        "yt-dlp completed without producing an audio file."
                    )

                downloaded_file = possible_files[0]

            return str(downloaded_file.resolve())

    except Exception as exc:
        logger.exception("YouTube audio download failed")
        raise RuntimeError(
            f"Unable to download YouTube audio: {exc}"
        ) from exc


def convert_to_wav(input_path: str) -> str:
    """
    Convert an audio/video file to mono 16 kHz WAV.
    Requires FFmpeg for video and many audio formats.
    """

    source = Path(input_path)

    if not source.exists():
        raise FileNotFoundError(f"Input file not found: {source}")

    output_path = source.with_name(f"{source.stem}_converted.wav")

    try:
        audio = AudioSegment.from_file(str(source))
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export(str(output_path), format="wav")

        return str(output_path.resolve())

    except Exception as exc:
        logger.exception("Audio conversion failed")
        raise RuntimeError(
            f"Could not convert audio to WAV. Check FFmpeg installation: {exc}"
        ) from exc


def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list[str]:
    """
    Split WAV audio into chunks of the requested duration.
    Returns a list of chunk file paths.
    """

    source = Path(wav_path)

    if not source.exists():
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
        raise RuntimeError(f"Could not split audio: {exc}") from exc


def process_input(source: str, chunk_minutes: int = 10) -> list[str]:
    """
    Process a local audio/video file or a YouTube URL.
    Returns WAV chunk paths for transcription.
    """

    if not source or not source.strip():
        raise ValueError("Please upload a file or provide a URL.")

    source = source.strip()

    if source.startswith(("http://", "https://")):
        print("Detected URL. Downloading audio...")
        input_path = download_youtube_audio(source)
    else:
        input_path = source

        if not Path(input_path).exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

    print("Converting audio to WAV...")
    wav_path = convert_to_wav(input_path)

    print(f"Splitting audio into {chunk_minutes}-minute chunks...")
    chunks = chunk_audio(wav_path, chunk_minutes)

    print(f"Audio processing complete. Created {len(chunks)} chunks.")
    return chunks