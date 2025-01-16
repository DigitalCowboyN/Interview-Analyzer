import os
import pandas as pd
import webvtt
from pathlib import Path
from transformers import pipeline

def vtt_to_csv(vtt_path):
    try:
        captions = webvtt.read(vtt_path)
        rows = []
        for caption in captions:
            start = caption.start
            end = caption.end
            text = caption.text.replace('\n', ' ')  # Replace newlines in captions with space
            rows.append([start, end, text])
        df = pd.DataFrame(rows, columns=['Start', 'End', 'Text'])
        csv_path = vtt_path.with_suffix('.csv')
        df.to_csv(csv_path, index=False)
        print(f"Successfully converted {vtt_path} to {csv_path}.")
        return csv_path
    except Exception as e:
        print(f"Failed to convert {vtt_path}: {str(e)}")
        return None

def process_csv(csv_path):
    try:
        data = pd.read_csv(csv_path)
        nlp = pipeline("sentiment-analysis")
        def process_text(text):
            result = nlp(text)
            return result
        data['Sentiment'] = data['Text'].apply(lambda x: process_text(x) if isinstance(x, str) else x)
        processed_csv_path = csv_path.with_name(csv_path.stem + '_processed.csv')
        data.to_csv(processed_csv_path, index=False)
        print(f"Processing complete. Results saved to {processed_csv_path}.")
    except Exception as e:
        print(f"Failed to process {csv_path}: {str(e)}")

def process_directory(directory_path):
    pathlist = Path(directory_path).glob('**/*.vtt')  # Using recursive glob pattern
    for path in pathlist:
        csv_path = vtt_to_csv(path)
        if csv_path:
            process_csv(Path(csv_path))

# Specify the directory containing VTT files
directory_path = 'Transcripts/Interviews/'
process_directory(directory_path)
