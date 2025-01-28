import os
from pathlib import Path
import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import whisper
from .audio_conversion import convert_mp4a_to_wav_if_needed
from typing import List

def ensure_directories_exist(*paths):
    """
    Ensure that all directories for the given paths exist.
    Creates parent directories if they do not already exist.

    Args:
        *paths: Variable length argument list of file paths.
    """
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)

# Load SpaCy English model
nlp = spacy.load("en_core_web_sm")

def validate_audio(audio_file):
    """
    Validate that the audio file is in a proper format and readable.

    Args:
        audio_file (Path): Path to the audio file to validate.

    Returns:
        bool: True if the audio file is valid, False otherwise.
    """
    try:
        import soundfile as sf
        with sf.SoundFile(audio_file) as f:
            print(f"Valid audio file: {audio_file}")
    except Exception as e:
        print(f"Invalid audio file: {audio_file}, error: {e}")
        return False
    return True

def transcribe_audio(audio_file, transcript_path):
    """
    Transcribe audio to text using Whisper and save the result.

    Args:
        audio_file (Path): Path to the .wav audio file.
        transcript_path (Path): Path to save the generated transcript file.

    Returns:
        Path: Path to the saved transcript file.
    """
    print(f"Transcribing audio: {audio_file}")
    model = whisper.load_model("small")
    result = model.transcribe(str(audio_file))
    with open(transcript_path, "w") as f:
        f.write(result["text"])
    print(f"Saved transcript to: {transcript_path}")
    return transcript_path

def split_text_into_chunks_with_punctuation(
    text: str, max_words_per_chunk: int = 30
) -> List[str]:
    """
    Split the text into chunks using end-of-sentence punctuation.

    Args:
        text (str): Unstructured text blob.
        max_words_per_chunk (int): Maximum number of words per chunk.

    Returns:
        List[str]: List of text chunks.
    """
    import re

    # Split the text into sentences using punctuation as delimiters
    sentences = re.split(r"(?<=[.!?]) +", text.strip())

    chunks, current_chunk, current_word_count = [], [], 0
    for sentence in sentences:
        word_count = len(sentence.split())

        # Add sentence to the current chunk if it fits
        if current_word_count + word_count <= max_words_per_chunk:
            current_chunk.append(sentence)
            current_word_count += word_count
        else:
            # Start a new chunk
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_word_count = word_count

    # Add the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    # Debug: Print chunk sizes
    print(f"Generated {len(chunks)} chunks with sizes: {[len(chunk.split()) for chunk in chunks]}")
    return chunks

def generate_embeddings(chunks):
    """
    Generate embeddings for text chunks using Sentence Transformers.

    Args:
        chunks (List[str]): List of text chunks.

    Returns:
        List[np.ndarray]: List of embedding vectors.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(chunks)

def cluster_embeddings(embeddings, num_speakers=2):
    """
    Cluster embeddings into speaker groups using K-Means clustering.

    Args:
        embeddings (List[np.ndarray]): List of embedding vectors.
        num_speakers (int): Number of speakers to cluster.

    Returns:
        List[int]: Cluster labels for each chunk.
    """
    kmeans = KMeans(n_clusters=num_speakers, random_state=42)
    return kmeans.fit_predict(embeddings)

def label_speakers(chunks, labels):
    """
    Label text chunks with speaker identities based on clustering results.

    Args:
        chunks (List[str]): List of text chunks.
        labels (List[int]): Cluster labels for each chunk.

    Returns:
        pd.DataFrame: DataFrame with text chunks and corresponding speaker labels.
    """
    return pd.DataFrame({"Text": chunks, "Speaker": [f"Speaker_{label}" for label in labels]})

def process_interview(interview_id: str, interviews_dir: Path, num_speakers=2):
    """
    Process a single interview by clustering text chunks into speaker groups.

    Args:
        interview_id (str): Unique identifier for the interview.
        interviews_dir (Path): Path to the base directory containing interview data.
        num_speakers (int): Number of speakers to identify.
    """
    print(f"Starting processing for interview ID: {interview_id}")

    # Paths for audio, transcript, and output files
    audio_folder, audio_file = interviews_dir / "audio", interviews_dir / "audio" / f"{interview_id}.wav"
    transcript_file, aligned_csv = interviews_dir / "transcripts" / f"{interview_id}.txt", interviews_dir / "aligned" / f"{interview_id}_aligned.csv"

    # Ensure output directory exists
    ensure_directories_exist(aligned_csv)

    # Convert audio to .wav if needed
    convert_mp4a_to_wav_if_needed(audio_folder)

    # Validate audio file
    if not validate_audio(audio_file):
        print(f"Skipping invalid audio file: {audio_file}")
        return

    # Transcribe audio if transcript does not exist
    if not transcript_file.exists():
        transcribe_audio(audio_file, transcript_file)

    # Read and process the transcript
    try:
        with open(transcript_file, "r") as f:
            text = f.read()
        chunks = split_text_into_chunks_with_punctuation(text)
    except Exception as e:
        print(f"Error reading transcript file {transcript_file}: {e}")
        return

    # Generate embeddings and cluster them
    try:
        embeddings = generate_embeddings(chunks)
        labels = cluster_embeddings(embeddings, num_speakers=num_speakers)
        df_aligned = label_speakers(chunks, labels)
    except Exception as e:
        print(f"Error during clustering: {e}")
        return

    # Save the aligned DataFrame to a CSV file
    try:
        df_aligned.to_csv(aligned_csv, index=False)
        print(f"Aligned CSV saved to: {aligned_csv}")
    except Exception as e:
        print(f"Error saving aligned CSV: {e}")

def main():
    """
    Main entry point for processing all interviews.
    Iterates over all transcript files in the directory and processes each.
    """
    base_dir, interviews_dir = Path(__file__).resolve().parent.parent, Path(__file__).resolve().parent.parent / "data" / "interviews"
    transcripts_dir, all_transcripts = interviews_dir / "transcripts", sorted((interviews_dir / "transcripts").glob("*.txt"))

    for transcript_file in all_transcripts:
        process_interview(transcript_file.stem, interviews_dir)

if __name__ == "__main__":
    main()
