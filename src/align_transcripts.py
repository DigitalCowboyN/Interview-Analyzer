# code for aligning result of audio and transcript processing
# src/align_transcript.py

import pandas as pd
from typing import Optional
from pyannote.core import Annotation
from datetime import datetime
from .process_transcripts import convert_time_to_seconds

def align_diarization_with_transcript(
    df_transcript: pd.DataFrame,
    diarization_result: Annotation
) -> pd.DataFrame:
    """
    Align the transcript DataFrame (with columns [Start, End, Text]) to speaker segments.
    Adds a 'Speaker' column indicating which speaker (SPEAKER_00, SPEAKER_01, etc.).
    """
    # Convert start/end to numeric seconds
    df_transcript["StartSec"] = df_transcript["Start"].apply(convert_time_to_seconds)
    df_transcript["EndSec"] = df_transcript["End"].apply(convert_time_to_seconds)

    # Flatten the diarization annotation into a list of segments
    diarized_segments = []
    for segment, _, speaker_label in diarization_result.itertracks(yield_label=True):
        diarized_segments.append({
            "start": segment.start,
            "end": segment.end,
            "speaker": speaker_label
        })

    # For each transcript row, find which speaker segment covers its midpoint
    def find_speaker(row):
        midpoint = (row["StartSec"] + row["EndSec"]) / 2.0
        for seg in diarized_segments:
            if seg["start"] <= midpoint <= seg["end"]:
                return seg["speaker"]
        return "Unknown"

    df_transcript["Speaker"] = df_transcript.apply(find_speaker, axis=1)
    return df_transcript
