import yt_dlp
from pydub import AudioSegment
import os
DOWNLOAD_DIR = 'downloades'
os.makedirs(DOWNLOAD_DIR,exist_ok=True)
def download_youtube_audio(url:str) -> str:
    output_path = os.path.join(DOWNLOAD_DIR,'%(title)s.%(ext)s')
    ydl_opts = {
    "format": "bestaudio/best",
    "outtmpl": "downloads/%(title)s.%(ext)s",
    "noplaylist": True,
    "cachedir": False,
    "quiet": False,
}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url,download=True)
        filename = ydl.prepare_filename(info).replace('.webm','.wav').replace('.m4a','.wav')
    return filename
def convert_to_wav(input_path:str) -> str:
    '''Convert any audio/video file to WAV format using pydub.'''
    output_path = os.path.splitext(input_path)[0] + '_converted.wav'
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_path,format='wav')
    return output_path
def chunk_audio(wav_path:str,chunk_minutes:int=10) -> list:
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000
    chunks = []
    for i,start in enumerate(range(0,len(audio),chunk_ms)):
        chunk = audio[start:start + chunk_ms]
        chunk_path = f'{wav_path}_chunk_{i}.wav'
        chunk.export(chunk_path,format='wav')
        chunks.append(chunk_path)
    return chunks
def process_input(source: str) -> list[str]:
    if source.startswith(("http://", "https://")):
        print("Detected YouTube URL. Downloading audio...")
        wav_path = download_youtube_audio(source)
    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    wav_path = convert_to_wav(wav_path) if not wav_path.lower().endswith(".wav") else wav_path
    chunks = chunk_audio(wav_path, chunk_minutes=10)

    print(f"Audio ready - {len(chunks)} chunk(s) created")
    return chunks