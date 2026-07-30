"""Text and tabular data preprocessing utilities for the Amazon Books sentiment project."""

import re
import html
import ast
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer


def clean_text(text: str) -> str:
    """Remove HTML artifacts (tags, entities) from raw review text."""
    if pd.isna(text):
        return ""
    text = html.unescape(text)
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_list_str(val: str) -> str:
    """Convert a stringified Python list (e.g. metadata fields) into plain text."""
    if not val or val == "" or val == "Unknown":
        return ""
    try:
        parsed = ast.literal_eval(val)
        if isinstance(parsed, list):
            return " ".join(str(x) for x in parsed)
        return str(parsed)
    except (ValueError, SyntaxError):
        return str(val)


def clean_price(val) -> float:
    """Extract a numeric price from messy string values (e.g. 'from 30.05')."""
    if pd.isna(val):
        return np.nan
    match = re.search(r"[\d]+\.?\d*", str(val))
    return float(match.group()) if match else np.nan


def to_sentiment(rating: float) -> str:
    """Map a 1-5 star rating to a 3-class sentiment label."""
    if rating <= 2:
        return "negative"
    elif rating == 3:
        return "neutral"
    else:
        return "positive"


def build_full_text(df: pd.DataFrame) -> pd.Series:
    """Concatenate title, review text, and description into a single text field."""
    return (
        df["title_review_clean"].fillna("") + ". " +
        df["text_clean"].fillna("") + ". " +
        df["description_clean"].fillna("")
    )


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full cleaning pipeline to a raw merged reviews+metadata dataframe."""
    df = df.copy()
    df = df.drop(columns=["bought_together"], errors="ignore")

    df["has_image_review"] = df["images_review"].notna().astype(int) if "images_review" in df else 0
    df["has_video"] = df["videos"].notna().astype(int) if "videos" in df else 0
    df = df.drop(columns=["images_review", "videos", "images_book"], errors="ignore")

    df = df.dropna(subset=["text"])

    for col in ["author", "subtitle", "store", "main_category", "categories", "features"]:
        if col in df:
            df[col] = df[col].fillna("Unknown")

    df["has_description"] = df["description"].notna().astype(int)
    df["description_clean"] = df["description"].apply(parse_list_str).apply(clean_text)
    df["features_clean"] = df["features"].apply(parse_list_str)

    df["text_clean"] = df["text"].apply(clean_text)
    df["title_review_clean"] = df["title_review"].apply(clean_text)
    df["full_text"] = build_full_text(df)
    df["text_length"] = df["text_clean"].str.len()

    df["price_clean"] = df["price"].apply(clean_price)
    df["price_clean"] = df.groupby("main_category")["price_clean"].transform(
        lambda x: x.fillna(x.median())
    )
    df["price_clean"] = df["price_clean"].fillna(df["price_clean"].median())

    df["sentiment"] = df["rating"].apply(to_sentiment)

    return df


def fit_feature_pipeline(X_train: pd.DataFrame, max_features: int = 10000):
    """Fit TF-IDF, OneHotEncoder, and StandardScaler on training data.

    Returns fitted transformers and the transformed training matrix.
    """
    tfidf = TfidfVectorizer(
        max_features=max_features, ngram_range=(1, 2),
        stop_words="english", min_df=5
    )
    X_train_tfidf = tfidf.fit_transform(X_train["full_text"])

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    train_cat = ohe.fit_transform(X_train[["main_category"]])

    numeric_cols = ["price_clean", "helpful_vote", "average_rating", "rating_number"]
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train[numeric_cols])

    return {
        "tfidf": tfidf, "ohe": ohe, "scaler": scaler,
        "numeric_cols": numeric_cols,
        "X_train_tfidf": X_train_tfidf,
    }


def transform_features(X: pd.DataFrame, pipeline: dict):
    """Apply a fitted feature pipeline to new data (val/test)."""
    from scipy.sparse import hstack, csr_matrix

    X_tfidf = pipeline["tfidf"].transform(X["full_text"])
    X_cat = pipeline["ohe"].transform(X[["main_category"]])
    X_num = pipeline["scaler"].transform(X[pipeline["numeric_cols"]])
    X_verified = X["verified_purchase"].astype(int).values.reshape(-1, 1)

    X_tab = np.hstack([X_num, X_verified, X_cat])
    return hstack([X_tfidf, csr_matrix(X_tab)])
