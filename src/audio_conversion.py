# ai_data_preprocessor/src/audio_conversion.py

import os
import subprocess
from pathlib import Path

def convert_mp4a_to_wav_if_needed(audio_dir: Path):
    """
    Looks for all .mp4a files in audio_dir. If a matching .wav file 
    does not exist, convert the .mp4a to .wav using ffmpeg.
    
    We do NOT delete or overwrite the .mp4a source; we keep both files.
    """
    # Make sure ffmpeg is installed in the container/environment
    # e.g. apt-get install ffmpeg in your Dockerfile
    
    audio_files = list(audio_dir.glob("*.mp4*"))
    
    for audio_file in audio_files:
        base_name = audio_file.stem  # e.g. "interview1" from "interview1.mp4a"
        wav_file = audio_dir / f"{base_name}.wav"
        
        if not wav_file.exists():
            print(f"Converting {audio_file.name} --> {wav_file.name}")
            # ffmpeg command: ffmpeg -i input.mp4a output.wav
            try:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",            # overwrite if needed
                        "-i", 
                        str(audio_file),
                        str(wav_file)
                    ],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                print(f"Conversion complete: {wav_file.name}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to convert {audio_file.name}:\n{e.stderr.decode('utf-8')}")
        else:
            print(f"WAV file already exists for {audio_file.name}; skipping conversion.")
