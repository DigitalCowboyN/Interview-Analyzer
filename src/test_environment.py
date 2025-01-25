# test_environment.py

import os

def test_webvtt():
    print("Testing webvtt...")

    try:
        import webvtt
        # If you have a small .vtt file to test, you can do something like:
        # vtt = webvtt.read("path/to/example.vtt")
        # print("Number of captions:", len(vtt))
        print("webvtt imported successfully!")
    except Exception as e:
        print("webvtt test FAILED:", e)


def test_pandas():
    print("Testing pandas...")

    try:
        import pandas as pd
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        print("DataFrame created:\n", df)
        print("pandas imported successfully!")
    except Exception as e:
        print("pandas test FAILED:", e)


def test_pyannote():
    print("Testing pyannote.audio...")

    try:
        from pyannote.audio import Pipeline
        from pyannote.core import Segment
        # Minimal check: create a pipeline object or a Segment
        dummy_segment = Segment(start=0, end=10)
        print("Created a dummy segment:", dummy_segment)
        print("pyannote.audio imported successfully!")
    except Exception as e:
        print("pyannote.audio test FAILED:", e)

if __name__ == "__main__":
    test_webvtt()
    test_pandas()
    test_pyannote()
