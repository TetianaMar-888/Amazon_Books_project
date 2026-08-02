# Data

This project uses a sample of the **McAuley-Lab/Amazon-Reviews-2023** dataset 
(Books category) from Hugging Face:
https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023

## How to reproduce the dataset

```bash
python src/download_data.py
```

This downloads a stratified sample of 50,000 reviews (seed=42) and their 
matching product metadata, then saves the merged raw dataset to 
`data/sample/reviews_books_sample.parquet`.

## Data Pipeline

1. `python src/download_data.py` — downloads raw reviews + metadata sample (~50k rows)
2. `notebooks/01_eda.ipynb` — applies cleaning (via `src/preprocessing.py`), 
   producing the cleaned dataset (`books_cleaned.parquet`) used by all 
   subsequent notebooks
