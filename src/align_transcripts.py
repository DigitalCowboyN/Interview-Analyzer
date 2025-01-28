# code for aligning result of audio and transcript processing
# src/align_transcript.py

import pandas as pd
from typing import List

def align_diarization_with_transcript(transcript_df, speakers):
    """
    Aligns speaker labels with transcript text lines.

    Args:
        transcript_df (pd.DataFrame): The transcript as a DataFrame with a 'Text' column.
        speakers (List[str]): A list of speaker labels in sequential order.

    Returns:
        pd.DataFrame: DataFrame with 'Text' and 'Speaker' columns.
    """
    if "Text" not in transcript_df.columns:
        raise ValueError("Transcript DataFrame must contain a 'Text' column.")

    # Assign speakers to text lines sequentially
    transcript_df["Speaker"] = speakers[: len(transcript_df)]
    return transcript_df


# def align_diarization_with_transcript(transcript_df: pd.DataFrame, diarization_results: List[dict]) -> pd.DataFrame:
#     """
#     Aligns diarized speaker segments with transcript text lines.

#     Args:
#         transcript_df (pd.DataFrame): The transcript as a DataFrame with a 'Text' column.
#         diarization_results (List[dict]): List of diarization segments, each containing 'start', 'end', and 'speaker'.

#     Returns:
#         pd.DataFrame: DataFrame with 'Start', 'End', 'Text', and 'Speaker' columns.
#     """
#     if "Text" not in transcript_df.columns:
#         raise ValueError("Transcript DataFrame must contain a 'Text' column.")

#     # Initialize alignment result list
#     aligned_rows = []

#     # Flatten diarization results for easier processing
#     diarization_segments = [
#         {"start": segment["start"], "end": segment["end"], "speaker": segment["speaker"]}
#         for segment in diarization_results
#     ]

#     # Initialize transcript and diarization pointers
#     transcript_index = 0

#     for segment in diarization_segments:
#         segment_start = segment["start"]
#         segment_end = segment["end"]
#         speaker = segment["speaker"]

#         # Group transcript lines into this diarization segment
#         combined_text = []

#         while transcript_index < len(transcript_df):
#             text_line = transcript_df.iloc[transcript_index]
#             text = text_line["Text"]

#             # Add the text line to the current segment
#             combined_text.append(text)
