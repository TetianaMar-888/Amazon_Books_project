"""
Downloads a stratified sample of Amazon Book Reviews (2023) from Hugging Face
and merges it with product metadata, reproducing the dataset used in this project.

Usage:
    python src/download_data.py
"""

import json
import pandas as pd
from huggingface_hub import hf_hub_download

N_REVIEWS = 50000
SEED = 42
OUTPUT_PATH = "data/sample/reviews_books_sample.parquet"


def download_reviews(n_samples=N_REVIEWS):
    """Download the first n_samples reviews from the raw Books category file."""
    print("Downloading reviews file (this may take a while, ~20GB source file)...")
    path = hf_hub_download(
        repo_id="McAuley-Lab/Amazon-Reviews-2023",
        filename="raw/review_categories/Books.jsonl",
        repo_type="dataset",
    )

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n_samples:
                break
            records.append(json.loads(line))

    return pd.DataFrame(records)


def download_metadata(wanted_asins):
    """Download only the metadata records matching the review sample's parent_asin."""
    print("Downloading metadata file and filtering to relevant products...")
    meta_path = hf_hub_download(
        repo_id="McAuley-Lab/Amazon-Reviews-2023",
        filename="raw/meta_categories/meta_Books.jsonl",
        repo_type="dataset",
    )

    records = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("parent_asin") in wanted_asins:
                records.append(rec)

    return pd.DataFrame(records)


def main():
    reviews_df = download_reviews()
    print(f"Downloaded {len(reviews_df)} reviews.")

    wanted_asins = set(reviews_df["parent_asin"])
    meta_df = download_metadata(wanted_asins)
    print(f"Downloaded {len(meta_df)} matching metadata records.")

    df = reviews_df.merge(meta_df, on="parent_asin", how="left", suffixes=("_review", "_book"))

    import os
    os.makedirs("data/sample", exist_ok=True)
    df.to_parquet(OUTPUT_PATH)
    print(f"Saved merged sample ({df.shape}) to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
