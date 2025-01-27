# ai_data_preprocessor/src/classify_relevance.py

import pandas as pd
from transformers import pipeline

def load_zero_shot_classifier(model_name="facebook/bart-large-mnli"):
    """
    Create or load a zero-shot classification pipeline.
    """
    classifier = pipeline("zero-shot-classification", model=model_name)
    return classifier

def classify_line(text: str, classifier, threshold=0.8):
    """
    Classify text into one of four categories:
    - relevant
    - might be relevant
    - irrelevant
    - might be irrelevant

    Uses a zero-shot classifier with candidate labels [relevant, irrelevant].
    'threshold' is the confidence cutoff to decide
    between a definite label vs. "might be" label.
    """

    # candidate labels
    result = classifier(
        text,
        candidate_labels=["relevant", "irrelevant"]
    )
    # result structure: { 'labels': ['relevant', 'irrelevant'], 'scores': [0.7, 0.3], ... }
    top_label = result["labels"][0]          # 'relevant' or 'irrelevant'
    confidence = result["scores"][0]        # e.g. 0.7

    if top_label == "relevant":
        if confidence >= threshold:
            return "relevant"
        else:
            return "might be relevant"
    else:  # top_label == "irrelevant"
        if confidence >= threshold:
            return "irrelevant"
        else:
            return "might be irrelevant"

def classify_csv(csv_path, classifier, threshold=0.8, output_path=None):
    """
    Read a CSV (e.g. an aligned transcript),
    classify each line's text into 4 categories,
    add an 'AutoClass' column, and optionally save to a new CSV.

    :param csv_path: Path to the input CSV file (aligned transcripts).
    :param classifier: A zero-shot classification pipeline.
    :param threshold: Float threshold for "confidence" to decide definite vs. 'might be'.
    :param output_path: Optional path to save the updated CSV. If None, we don't save.
    :return: A DataFrame with the new 'AutoClass' column.
    """
    df = pd.read_csv(csv_path)

    # Ensure there's a 'Text' column
    if "Text" not in df.columns:
        print(f"Error: CSV {csv_path} has no 'Text' column.")
        return df

    # Classify each row
    auto_classes = []
    for _, row in df.iterrows():
        text_line = str(row["Text"])
        auto_label = classify_line(text_line, classifier, threshold=threshold)
        auto_classes.append(auto_label)

    df["AutoClass"] = auto_classes

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Saved classified CSV to: {output_path}")

    return df
