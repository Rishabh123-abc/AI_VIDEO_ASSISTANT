import os
import subprocess
import tempfile
from pathlib import Path

import whisper


# ==========================================
# CONFIGURATION
# ==========================================

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

# Keep the loaded model in memory for reuse.
model = None


# ==========================================
# LOAD WHISPER MODEL
# ==========================================

def load_model():
    global model

    if model is None:
        print(f"Loading local Whisper model: {WHISPER_MODEL}")
        print("The first run may download the model.")

        model = whisper.load_model(WHISPER_MODEL)

        print("Whisper model loaded successfully.")

    return model


# ==========================================
# SPLIT AUDIO USING FFMPEG
# ==========================================

def chunk_audio(
    audio_path: str,
    output_dir: str,
    chunk_duration: int = 300,
) -> list[str]:

    audio_path = str(Path(audio_path).expanduser().resolve())

    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if chunk_duration <= 0:
        raise ValueError("chunk_duration must be greater than zero.")

    os.makedirs(output_dir, exist_ok=True)

    # Remove previous chunks to prevent duplicate transcription.
    for old_chunk in Path(output_dir).glob("chunk_*.wav"):
        old_chunk.unlink()

    output_pattern = str(
        Path(output_dir) / "chunk_%03d.wav"
    )

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", audio_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        "-f", "segment",
        "-segment_time", str(chunk_duration),
        "-reset_timestamps", "1",
        output_pattern,
    ]

    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )

    except FileNotFoundError as exc:
        raise RuntimeError(
            "FFmpeg is not installed or is missing from PATH."
        ) from exc

    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"FFmpeg failed to split audio: {exc.stderr}"
        ) from exc

    chunks = sorted(
        str(path)
        for path in Path(output_dir).glob("chunk_*.wav")
    )

    if not chunks:
        raise RuntimeError("No audio chunks were generated.")

    print(f"Created {len(chunks)} audio chunk(s).")

    return chunks


# ==========================================
# TRANSCRIBE ONE CHUNK USING LOCAL WHISPER
# ==========================================

def transcribe_chunk_whisper(
    chunk_path: str,
    language: str = "english",
) -> str:

    whisper_model = load_model()

    language = language.lower().strip()

    # Force Hindi or English when explicitly requested.
    # Hinglish uses automatic language detection because speech
    # may contain a mixture of Hindi and English.
    if language in {"hindi", "hi", "hi-in"}:
        language_code = "hi"

    elif language in {"english", "en", "en-us", "en-in"}:
        language_code = "en"

    elif language in {
        "hinglish",
        "hindi-english",
        "hindi_english",
    }:
        language_code = None

    else:
        raise ValueError(
            "Unsupported language. Choose english, hindi, or hinglish."
        )

    options = {
        "task": "transcribe",
        "fp16": False,
        "verbose": False,
    }

    if language_code is not None:
        options["language"] = language_code

    result = whisper_model.transcribe(
        str(chunk_path),
        **options,
    )

    return result.get("text", "").strip()


# ==========================================
# TRANSCRIBE ONE CHUNK
# ==========================================

def transcribe_chunk(
    chunk_path: str,
    language: str,
) -> str:

    # All languages now use local Whisper.
    return transcribe_chunk_whisper(
        chunk_path,
        language=language,
    )


# ==========================================
# TRANSCRIBE COMPLETE AUDIO
# ==========================================

def transcribe_audio(
    audio_path: str | list[str],
    language: str = "english",
) -> str:

    language = language.lower().strip()

    supported_languages = {
        "english",
        "en",
        "en-us",
        "en-in",
        "hindi",
        "hi",
        "hi-in",
        "hinglish",
        "hindi-english",
        "hindi_english",
    }

    if language not in supported_languages:
        raise ValueError(
            "Unsupported language. Choose english, hindi, or hinglish."
        )

    # Accept either a file path or already-created chunk paths.
    if isinstance(audio_path, list):
        if not audio_path:
            raise ValueError("No audio chunks were provided.")

        chunks = [
            str(Path(path).expanduser().resolve())
            for path in audio_path
        ]

        for chunk in chunks:
            if not os.path.isfile(chunk):
                raise FileNotFoundError(
                    f"Audio chunk not found: {chunk}"
                )

        temp_dir = None

    else:
        audio_path = str(
            Path(audio_path).expanduser().resolve()
        )

        if not os.path.isfile(audio_path):
            raise FileNotFoundError(
                f"Audio file not found: {audio_path}"
            )

        # Local Whisper can handle longer chunks, reducing repeated
        # model calls compared with the previous 25-second chunks.
        chunk_duration = 600

        with tempfile.TemporaryDirectory(
            prefix="transcription_chunks_"
        ) as temp_dir:
            chunks = chunk_audio(
                audio_path=audio_path,
                output_dir=temp_dir,
                chunk_duration=chunk_duration,
            )

            transcripts = []

            for index, chunk_path in enumerate(chunks, start=1):
                print(
                    f"Transcribing chunk {index}/{len(chunks)}..."
                )

                try:
                    text = transcribe_chunk(
                        chunk_path=chunk_path,
                        language=language,
                    )

                    if text:
                        transcripts.append(text)

                except Exception as exc:
                    raise RuntimeError(
                        f"Chunk {index}/{len(chunks)} failed: {exc}"
                    ) from exc

            full_transcript = " ".join(transcripts).strip()

            print("Transcription completed successfully.")

            return full_transcript

    # This section handles an existing list of chunks.
    transcripts = []

    for index, chunk_path in enumerate(chunks, start=1):
        print(
            f"Transcribing chunk {index}/{len(chunks)}..."
        )

        try:
            text = transcribe_chunk(
                chunk_path=chunk_path,
                language=language,
            )

            if text:
                transcripts.append(text)

        except Exception as exc:
            raise RuntimeError(
                f"Chunk {index}/{len(chunks)} failed: {exc}"
            ) from exc

    full_transcript = " ".join(transcripts).strip()

    print("Transcription completed successfully.")

    return full_transcript


# ==========================================
# RUN DIRECTLY
# ==========================================

if __name__ == "__main__":
    audio_file = input("Enter audio file path: ").strip().strip('"')

    language = input(
        "Enter language (english/hindi/hinglish): "
    ).strip().lower()

    try:
        transcript = transcribe_audio(
            audio_path=audio_file,
            language=language,
        )

        print("\n========== TRANSCRIPT ==========\n")
        print(transcript)

        output_file = (
            Path(audio_file).with_suffix("")
        )

        output_file = Path(
            str(output_file) + "_transcript.txt"
        )

        output_file.write_text(
            transcript,
            encoding="utf-8",
        )

        print(f"\nTranscript saved to: {output_file}")

    except Exception as exc:
        print(f"\nTranscription error: {exc}")