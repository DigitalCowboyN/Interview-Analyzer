import os
from pathlib import Path
import pandas as pd
from .process_transcripts import vtt_to_dataframe
from .process_diarization import run_speaker_diarization
from .align_transcripts import align_diarization_with_transcript
from .audio_conversion import convert_mp4a_to_wav_if_needed
from src.classify_relevance import (
    load_zero_shot_classifier,
    classify_csv
)

def ensure_directories_exist(*paths):
    """
    Ensure that all directories for the given paths exist.
    Creates parent directories if they do not already exist.

    Args:
        *paths: Variable length argument list of file paths.
    """
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)

def needs_rediarization(transcript_file, audio_file, aligned_csv_path):
    """
    Check if diarization needs to be rerun based on file modification times.

    Args:
        transcript_file (Path): Path to the transcript file.
        audio_file (Path): Path to the audio file.
        aligned_csv_path (Path): Path to the aligned CSV file.

    Returns:
        bool: True if diarization needs to be rerun, False otherwise.
    """
    if not aligned_csv_path.exists():
        return True
    aligned_mtime = aligned_csv_path.stat().st_mtime
    transcript_mtime = transcript_file.stat().st_mtime if transcript_file.exists() else 0
    audio_mtime = audio_file.stat().st_mtime if audio_file.exists() else 0
    return (transcript_mtime > aligned_mtime) or (audio_mtime > aligned_mtime)

def needs_reclassification(aligned_csv_path, classified_csv_path):
    """
    Check if classification needs to be rerun based on file modification times.

    Args:
        aligned_csv_path (Path): Path to the aligned CSV file.
        classified_csv_path (Path): Path to the classified CSV file.

    Returns:
        bool: True if classification needs to be rerun, False otherwise.
    """
    if not classified_csv_path.exists():
        return True
    aligned_mtime = aligned_csv_path.stat().st_mtime
    classified_mtime = classified_csv_path.stat().st_mtime
    return aligned_mtime > classified_mtime

def convert_audio_if_needed(audio_folder):
    """
    Convert audio files from .mp4a to .wav if needed.

    Args:
        audio_folder (Path): Path to the directory containing audio files.
    """
    print("Checking and converting audio files if needed...")
    convert_mp4a_to_wav_if_needed(audio_folder)

def run_diarization_if_needed(transcript_file, audio_wav_file, aligned_csv):
    """
    Run diarization and alignment if necessary, and save the aligned CSV.

    Args:
        transcript_file (Path): Path to the transcript file.
        audio_wav_file (Path): Path to the .wav audio file.
        aligned_csv (Path): Path to save the aligned CSV.

    Returns:
        pd.DataFrame: The aligned DataFrame.
    """
    if not transcript_file.exists():
        print(f"No transcript found: {transcript_file}")
        return None

    # Convert the transcript file to a DataFrame
    df_transcript = vtt_to_dataframe(transcript_file)

    if audio_wav_file.exists():
        # Perform speaker diarization on the audio file
        diarization_result = run_speaker_diarization(str(audio_wav_file))
        # Align the transcript with the diarization results
        df_aligned = align_diarization_with_transcript(df_transcript, diarization_result)
    else:
        print(f"Warning: No audio found for {audio_wav_file}. Skipping diarization.")
        df_aligned = df_transcript.copy()
        df_aligned["Speaker"] = "Unknown"

    # Save the aligned DataFrame to a CSV file
    df_aligned.to_csv(aligned_csv, index=False)
    print(f"Saved aligned CSV to: {aligned_csv}")
    return df_aligned

def classify_if_needed(aligned_csv, classified_csv, threshold=0.8):
    """
    Run classification if necessary and save the classified CSV.

    Args:
        aligned_csv (Path): Path to the aligned CSV file.
        classified_csv (Path): Path to save the classified CSV file.
        threshold (float): Confidence threshold for classification.
    """
    if not aligned_csv.exists():
        print(f"Aligned CSV not found: {aligned_csv}")
        return None

    # Load the zero-shot classifier
    classifier = load_zero_shot_classifier("facebook/bart-large-mnli")
    # Perform classification and save results
    classify_csv(
        csv_path=aligned_csv,
        classifier=classifier,
        threshold=threshold,
        output_path=classified_csv
    )
    print(f"Saved classified CSV to: {classified_csv}")

def process_interview(interview_id: str, interviews_dir: Path):
    """
    Process a single interview by:
    1. Converting audio (if needed).
    2. Running diarization (if needed).
    3. Aligning transcripts and diarization.
    4. Running classification (if needed).

    Args:
        interview_id (str): Unique identifier for the interview.
        interviews_dir (Path): Path to the base directory containing interview data.
    """
    print(f"Starting processing for interview ID: {interview_id}")

    # Define paths for transcript, audio, aligned CSV, and classified CSV
    transcript_file = interviews_dir / "transcripts" / f"{interview_id}.vtt"
    audio_folder = interviews_dir / "audio"
    audio_wav_file = audio_folder / f"{interview_id}.wav"
    aligned_csv = interviews_dir / "aligned" / f"{interview_id}_aligned.csv"
    classified_csv = interviews_dir / "classified" / f"{interview_id}_classified.csv"

    # Ensure output directories exist
    ensure_directories_exist(aligned_csv, classified_csv)

    # Step 1: Convert audio if needed
    convert_audio_if_needed(audio_folder)

    # Step 2: Run diarization and alignment if needed
    if needs_rediarization(transcript_file, audio_wav_file, aligned_csv):
        run_diarization_if_needed(transcript_file, audio_wav_file, aligned_csv)
    else:
        print(f"Skipping diarization, using existing: {aligned_csv}")

    # Step 3: Run classification if needed
    if needs_reclassification(aligned_csv, classified_csv):
        classify_if_needed(aligned_csv, classified_csv)
    else:
        print(f"Skipping classification, using existing: {classified_csv}")

    print(f"Processing complete for interview ID: {interview_id}\n")

def main():
    """
    Main entry point for processing all interviews.
    """
    base_dir = Path(__file__).resolve().parent.parent
    interviews_dir = base_dir / "data" / "interviews"

    # Collect all transcripts in the transcripts directory
    transcripts_dir = interviews_dir / "transcripts"
    all_transcripts = sorted(transcripts_dir.glob("*.vtt"))

    for transcript_path in all_transcripts:
        # Extract the interview ID from the filename
        interview_id = transcript_path.stem
        process_interview(interview_id, interviews_dir)

if __name__ == "__main__":
    main()
