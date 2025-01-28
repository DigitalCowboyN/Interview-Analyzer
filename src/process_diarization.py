# code for diatization processing of audio
# src/process_diarization.py

from pyannote.audio import Pipeline
from pyannote.core import Annotation
import os

def run_speaker_diarization(audio_path):
    """
    Run speaker diarization on the audio file and return speaker labels.

    Args:
        audio_path (str): Path to the audio file.

    Returns:
        List[str]: A list of speaker labels in sequential order.
    """
    try:
        print(f"Running diarization on: {audio_path}")
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token="your_huggingface_token")
        diarization = pipeline(audio_path)

        # Extract speaker labels sequentially (ignoring timestamps)
        speakers = []
        for _, _, speaker in diarization.itertracks(yield_label=True):
            speakers.append(speaker)

        # Debug: Print speaker labels
        print(f"Identified speakers: {speakers}")
        return speakers

    except Exception as e:
        print(f"Error during diarization: {e}")
        return None
