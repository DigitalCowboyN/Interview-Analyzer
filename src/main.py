import os
from pathlib import Path
import pandas as pd
import spacy
import whisper
from sentence_transformers import SentenceTransformer
import hdbscan
from sklearn.metrics.pairwise import cosine_similarity
# from ctransformers import AutoModelForCausalLM
from .audio_conversion import convert_mp4a_to_wav_if_needed
import subprocess

def ensure_directories_exist(*paths):
    """
    Ensure that all directories for the given paths exist.
    """
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)

# Load SpaCy English model
nlp = spacy.load("en_core_web_sm")

# Load local LLaMA model - no longer needed as the image is loaded directly
# llama_model = AutoModelForCausalLM.from_pretrained("llama-2-7b.gguf", model_type="llama")

def validate_audio(audio_file):
    """
    Validate that the audio file is in a proper format and readable.
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
    """
    print(f"Transcribing audio: {audio_file}")
    model = whisper.load_model("small")
    result = model.transcribe(str(audio_file))
    with open(transcript_path, "w") as f:
        f.write(result["text"])
    print(f"Saved transcript to: {transcript_path}")
    return transcript_path

def split_text_into_chunks_with_punctuation(text: str):
    """
    Split the text into chunks using end-of-sentence punctuation.
    """
    import re
    chunks = re.split(r"(?<=[.!?])\s+", text.strip())
    print(f"Generated {len(chunks)} chunks.")
    return chunks

def create_context_windows(chunks, window_size=2):
    """
    Creates contextual windows by including neighboring sentences.
    """
    context_chunks = []
    for i in range(len(chunks)):
        start = max(0, i - window_size)
        end = min(len(chunks), i + window_size + 1)
        context_window = " ".join(chunks[start:end])
        context_chunks.append(context_window)
    return context_chunks

def generate_embeddings(chunks):
    """
    Generate embeddings for text chunks.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(chunks)

# def cluster_embeddings(embeddings):
#     """
#     Cluster embeddings dynamically using HDBSCAN.
#     """
#     clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric="euclidean")
#     labels = clusterer.fit_predict(embeddings)
#     confidence_scores = clusterer.probabilities_
#     return labels, confidence_scores

# def assign_cluster_tags(chunks, labels, confidence_scores, threshold=0.7):
#     """
#     Assign meaningful tags to clusters.
#     """
#     from collections import Counter
#     cluster_text = {label: [] for label in set(labels) if label != -1}
#     for chunk, label in zip(chunks, labels):
#         if label != -1:
#             cluster_text[label].extend(chunk.split())
#     cluster_tags = {
#         label: " ".join([word for word, _ in Counter(words).most_common(3)])
#         for label, words in cluster_text.items()
#     }
#     tagged_chunks = []
#     for i, chunk in enumerate(chunks):
#         label = labels[i]
#         conf = confidence_scores[i]
#         if label == -1:
#             tag = "Excluded"
#         else:
#             tag = cluster_tags.get(label, "Uncategorized")
#             if conf < threshold:
#                 tag = f"might be {tag}"
#         tagged_chunks.append((chunk, tag, conf))
#     return tagged_chunks

def group_related_chunks(embeddings, threshold=0.8):
    """
    Groups related text chunks based on semantic similarity.
    """
    similarity_matrix = cosine_similarity(embeddings)
    labels = [-1] * len(embeddings)
    cluster_id = 0
    for i in range(len(embeddings)):
        if labels[i] == -1:
            labels[i] = cluster_id
            for j in range(i + 1, len(embeddings)):
                if similarity_matrix[i][j] > threshold:
                    labels[j] = cluster_id
            cluster_id += 1
    return labels

import subprocess
import json

def classify_with_llama(text):
    """
    Generate a JSON object classification for the given text using LLaMA 2.
    Expected output: a JSON object with keys "Category1", "Category2", and "FunctionalRole"
    with no extra text or formatting.
    If output is not valid JSON, return the fallback JSON string.
    """
    prompt_text = f"""You are a text classifier. For the sentence below, output a JSON object with exactly three keys: "Category1", "Category2", and "FunctionalRole". 
"Category1" is a short abstract category summarizing the sentence's primary intent or subject.
"Category2" is a short abstract category providing additional context.
"FunctionalRole" is the sentence's functional role, chosen exactly from: Statement, Observation, Question, Answer, Clarification, Instruction, Command, Suggestion, Request, Greeting, Small Talk.
Output only the JSON object without any extra text or formatting.
If you cannot produce exactly this, output exactly: {{"Category1": "Unknown", "Category2": "Unknown", "FunctionalRole": "Unknown"}}.
Sentence: {text}
Answer:"""

    command = [
        "/app/llama-cli",
        "-m", "/workspaces/ai_data_preprocessor/models/llama-2-7b-chat.Q4_K_M.gguf",
        "-t", "0.0",  # deterministic generation
        "-p", prompt_text,
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running LLaMA model: {result.stderr}")
        return '{"Category1": "Unknown", "Category2": "Unknown", "FunctionalRole": "Unknown"}'

    output = result.stdout.strip()
    if output.startswith(prompt_text):
        output = output[len(prompt_text):].strip()
    output = output.replace("[end of text]", "").strip()

    # Combine all non-empty lines to capture the complete answer.
    answer = " ".join(line.strip() for line in output.splitlines() if line.strip())
    print(f"LLaMA raw output: {answer}")

    try:
        data = json.loads(answer)
        # Verify that all expected keys are present
        if all(key in data for key in ["Category1", "Category2", "FunctionalRole"]):
            return json.dumps(data)
    except Exception as e:
        print(f"JSON parse error: {e}")

    print(f"Warning: Unexpected JSON response format from LLaMA: {answer}")
    return '{"Category1": "Unknown", "Category2": "Unknown", "FunctionalRole": "Unknown"}'

def process_interview(interview_id: str, interviews_dir: Path):
    """
    Process a single interview by classifying text chunks using a local LLaMA model.
    """
    print(f"Starting processing for interview ID: {interview_id}")
    
    audio_folder, audio_file = interviews_dir / "audio", interviews_dir / "audio" / f"{interview_id}.wav"
    transcript_file, aligned_csv = interviews_dir / "transcripts" / f"{interview_id}.txt", interviews_dir / "aligned" / f"{interview_id}_aligned.csv"
    ensure_directories_exist(aligned_csv)
    convert_mp4a_to_wav_if_needed(audio_folder)
    
    if not validate_audio(audio_file):
        print(f"Skipping invalid audio file: {audio_file}")
        return

    if not transcript_file.exists():
        print(f"Transcribing audio for: {interview_id}")
        transcribe_audio(audio_file, transcript_file)

    try:
        with open(transcript_file, "r") as f:
            text = f.read()
        print(f"Transcript loaded for: {interview_id}")

        chunks = split_text_into_chunks_with_punctuation(text)
        print(f"Generated {len(chunks)} chunks for classification.")

        context_chunks = create_context_windows(chunks)
    except Exception as e:
        print(f"Error reading transcript file {transcript_file}: {e}")
        return

    try:
        embeddings = generate_embeddings(context_chunks)
        labels = group_related_chunks(embeddings)

        tagged_chunks = []
        for i, chunk in enumerate(chunks):
            print(f"Classifying chunk {i + 1}/{len(chunks)}: {chunk[:50]}...")
            
            response = classify_with_llama(chunk)
            # Instead of splitting on commas, parse the JSON output.
            try:
                data = json.loads(response)
                category1 = data.get("Category1", "Unknown")
                category2 = data.get("Category2", "Unknown")
                functional_class = data.get("FunctionalRole", "Unknown")
            except Exception as e:
                print(f"Warning: Unexpected response format from LLaMA: {response}")
                category1, category2, functional_class = "Unknown", "Unknown", "Unknown"

            # Print the real-time LLaMA classification output
            print(f"LLaMA Output: {category1}, {category2}, {functional_class}")

            tagged_chunks.append((chunk.strip(), category1, category2, functional_class))

        print(f"Classification complete for {interview_id}.")

        df_chunks = pd.DataFrame(tagged_chunks, columns=["Text", "Category1", "Category2", "FunctionalCategory"])
    except Exception as e:
        print(f"Error during classification: {e}")
        return

    try:
        df_chunks.to_csv(aligned_csv, index=False)
        print(f"Aligned CSV saved to: {aligned_csv}")
    except Exception as e:
        print(f"Error saving aligned CSV: {e}")

def main():
    """
    Main entry point for processing all interviews.
    """
    base_dir, interviews_dir = Path(__file__).resolve().parent.parent, Path(__file__).resolve().parent.parent / "data" / "interviews"
    transcripts_dir, all_transcripts = interviews_dir / "transcripts", sorted((interviews_dir / "transcripts").glob("*.txt"))
    for transcript_file in all_transcripts:
        process_interview(transcript_file.stem, interviews_dir)

if __name__ == "__main__":
    main()
    