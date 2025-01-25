# code for diarization and processing transcripts
# src/process_transcripts.py

import webvtt
import pandas as pd
from pathlib import Path
from datetime import datetime

def vtt_to_dataframe(vtt_file: Path) -> pd.DataFrame:
    """
    Convert a .vtt file into a pandas DataFrame with columns: [Start, End, Text].
    """
    captions = webvtt.read(vtt_file)
    rows = []

    for caption in captions:
        start_time = caption.start
        end_time = caption.end
        text = caption.text.replace('\n', ' ')  # remove line breaks in the caption
        rows.append([start_time, end_time, text])

    df = pd.DataFrame(rows, columns=["Start", "End", "Text"])
    return df

def convert_time_to_seconds(timestr: str) -> float:
    """
    Convert a VTT time string 'HH:MM:SS.xxx' or 'MM:SS.xxx' to total seconds as a float.
    """
    # If there's only one colon, it might be 'MM:SS.xxx'; prepend "00:"
    if timestr.count(':') == 1:
        timestr = "00:" + timestr

    fmt = "%H:%M:%S.%f"
    dt = datetime.strptime(timestr, fmt)
    total_seconds = dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6
    return total_seconds
