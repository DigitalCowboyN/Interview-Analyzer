# code for diatization processing of audio
# src/process_diarization.py

from pyannote.audio import Pipeline
from pyannote.core import Annotation
import os

def run_speaker_diarization(audio_path: str):
    """
    Run pyannote.audio speaker diarization on the given audio file.
    Returns a pyannote.core.Annotation with speaker segments.
    """
    # 1 Retrieve HF token from environment
    HF_TOKEN = os.environ.get("HF_TOKEN", None)
    print("HF_TOKEN is:", HF_TOKEN)

    # 2 Instantiate the pipeline using the token (if any)
    if HF_TOKEN:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=HF_TOKEN)
    else:
        # If no token is set, the pipeline might fail if the model is gated
        # or succeed if the model is freely available
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

    # 3 Run diarization
    diarization_result = pipeline(audio_path)
    return diarization_result
