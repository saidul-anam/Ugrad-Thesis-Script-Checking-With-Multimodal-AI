"""
Ground Truth Text Cleaning and Normalization Script for OCR Evaluation (Gemma & EasyOCR).
Strips punctuation, lowercases, removes bracketed OCR tags, and cleans whitespace
to produce standardized reference text for fair Character Error Rate (CER)
and Word Error Rate (WER) computation.
"""

import os
import sys
import csv
import re
import unicodedata
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.eval.ocr_metrics import clean_text_for_ocr_eval, compute_cer, compute_wer


def clean_extraction_csv(input_csv_path: str, output_csv_path: str) -> pd.DataFrame:
    """
    Load extraction.csv, clean the ground-truth extracted text, and write extraction_cleaned.csv.
    """
    df = pd.read_csv(input_csv_path)

    print(f"Loaded {len(df)} rows from {input_csv_path}")

    # Clean the extracted text and question prompts
    df["extracted_text_clean"] = df["extracted_text"].apply(
        lambda x: clean_text_for_ocr_eval(str(x), remove_punct=True, to_lower=True) if pd.notna(x) else ""
    )
    
    df["question_clean"] = df["question"].apply(
        lambda x: clean_text_for_ocr_eval(str(x), remove_punct=False, to_lower=False) if pd.notna(x) else ""
    )

    df["clean_word_count"] = df["extracted_text_clean"].apply(
        lambda x: len(str(x).split()) if pd.notna(x) else 0
    )

    # Save to output CSV
    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
    df.to_csv(output_csv_path, index=False, encoding="utf-8")
    print(f"Successfully saved cleaned ground truth to {output_csv_path}")

    return df


if __name__ == "__main__":
    cand_paths = [
        os.path.abspath("e:/thesis/extraction.csv"),
        os.path.abspath("../extraction.csv"),
        os.path.abspath("extraction.csv"),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "extraction.csv")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "extraction.csv"))
    ]
    
    input_path = None
    for p in cand_paths:
        if os.path.exists(p):
            input_path = p
            break
            
    if not input_path:
        input_path = os.path.abspath("e:/thesis/extraction.csv")

    output_path = os.path.join(os.path.dirname(input_path), "extraction_cleaned.csv")

    print(f"Cleaning Ground Truth from: {input_path}")
    print(f"Output destination: {output_path}")

    df_cleaned = clean_extraction_csv(input_path, output_path)

    # Preview first 3 records
    for i in range(min(3, len(df_cleaned))):
        row = df_cleaned.iloc[i]
        print(f"\n--- Task {row['task_id']} ({row['question_type']}) ---")
        print("RAW GT  :", str(row['extracted_text'])[:90].replace('\n', ' '))
        print("CLEAN GT:", str(row['extracted_text_clean'])[:90])
        print(f"Clean Words: {row['clean_word_count']}")
