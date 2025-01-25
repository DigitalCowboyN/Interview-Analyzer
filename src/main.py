#main python code
# src/main.py

import os
from pathlib import Path
import pandas as pd

# Import functions
from .process_transcripts import vtt_to_dataframe
from .process_diarization import run_speaker_diarization
from .align_transcripts import align_diarization_with_transcript
from .audio_conversion import convert_mp4a_to_wav_if_needed

#HF_TOKEN = os.environ.get("HF_TOKEN")

def process_interview(interview_id: str, interviews_dir: Path):
    """
    1) Find audio file and transcript file
    2) Convert mp4a to wav if needed
    3) Run diarization if audio is found
    4) Align transcript with diarization
    5) Save the result as CSV
    """

    # Example: "data/transcripts/interview1.vtt"
    transcript_file = interviews_dir / "transcripts" / f"{interview_id}.vtt"
    audio_folder = interviews_dir / "audio"
    audio_wav_file = audio_folder / f"{interview_id}.wav"

    # (You might need to convert .mp4a to .wav with ffmpeg first)
    # audio_file = data_dir / "audio" / f"{interview_id}.wav"

    # Step 2: Convert mp4a to wav if needed
    convert_mp4a_to_wav_if_needed(audio_folder)

    # STEP 3: Parse Transcripts .vtt -> DataFrame
    if not transcript_file.exists():
        print(f"Transcript file not found: {transcript_file}. Skipping.")
        return

    df_transcript = vtt_to_dataframe(transcript_file)

    # Step 4: Run diarization if .wav file exists
    if audio_wav_file.exists():
        # STEP B: Run diarization
        diarization_result = run_speaker_diarization(str(audio_wav_file))
        # STEP C: Align transcript with diarization
        df_aligned = align_diarization_with_transcript(df_transcript, diarization_result)
    else:
        print(f"Warning: No audio found for {interview_id}. Skipping diarization.")
        df_aligned = df_transcript.copy()
        df_aligned["Speaker"] = "Unknown"

    # 4) Save output in interviews/processed/
    processed_dir = interviews_dir / "processed"
    processed_dir.mkdir(exist_ok=True)
    output_file = processed_dir / f"{interview_id}_aligned.csv"

    df_aligned.to_csv(output_file, index=False)
    print(f"Processed interview: {interview_id} -> {output_file}")

if __name__ == "__main__":
    # Root data folder
    base_dir = Path(__file__).resolve().parent.parent  # ai_data_preprocessor/
    data_dir = base_dir / "data"
    
    # The "interviews" folder within data/
    interviews_dir = data_dir / "interviews"
    
    # STEP A: Collect all .vtt from data/interviews/transcripts/
    transcripts_dir = interviews_dir / "transcripts"
    if not transcripts_dir.exists():
        print(f"No transcripts directory found at {transcripts_dir}")
        exit(0)

    # Find all .vtt files
    transcript_files = sorted(transcripts_dir.glob("*.vtt"))
    if not transcript_files:
        print(f"No .vtt files found in {transcripts_dir}")
        exit(0)

    # STEP B: For each file, parse the interview_id from the filename
    for transcript_path in transcript_files:
        interview_id = transcript_path.stem  # e.g. "interview1"
        print(f"Processing interview ID: {interview_id}")
        process_interview(interview_id, interviews_dir)
