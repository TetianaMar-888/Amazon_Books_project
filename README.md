# Amazon_Books_project
Sentiment classification on Amazon Book Reviews using TF-IDF+XGBoost, fine-tuned DistilBERT, and LLM API — with EDA, imbalanced learning, and business insights.

This project demonstrates:

- **Feature engineering** for NLP tasks on real-world, messy data
- A comparison of **classical ML, fine-tuned transformer, and LLM API** approaches to classification
- Handling **imbalanced classes**
- **Business-oriented interpretation** of results

These are core skills expected in modern Data Scientist roles across fintech, product, and data-driven companies.

## Data Source
Full dataset: McAuley-Lab/Amazon-Reviews-2023 (Books category), Hugging Face
https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023

Run `src/download_data.py` to reproduce the sample used in this project
(50,000 reviews, seed=42).

## Exploratory Data Analysis (EDA)

### Target Variable Distribution
The dataset is highly imbalanced: **84.6% of reviews are positive** (rating 4-5), 
**8.9% neutral** (rating 3), and only **6.4% negative** (rating 1-2). This reflects 
a common pattern in e-commerce reviews — satisfied customers are more likely to 
leave feedback. This imbalance directly informed our choice to use **macro F1** 
as the primary metric and to apply class-weighting / resampling techniques 
during modeling.

### Text Length by Sentiment
Interestingly, **neutral reviews tend to be longer** (median ~600 characters) 
than both negative (~290) and positive (~310) reviews. A likely explanation: 
users giving a middling rating often explain both pros and cons in detail, 
while strongly positive or negative reactions tend to be shorter and more emotional.

### Numeric Feature Distributions
All count-based features (`price`, `helpful_vote`, `rating_number`, `text_length`) 
are strongly right-skewed, typical of e-commerce data. `average_rating` is the 
exception, roughly normally distributed around 4.3–4.7 — most books in the 
dataset are already well-rated, which limits its usefulness as a standalone 
predictive feature.

### Correlation Analysis
Correlations among numeric features are weak overall (|r| ≤ 0.16), meaning no 
single tabular feature strongly predicts sentiment. The two mildest patterns: 
`text_length` correlates positively with `helpful_vote` (r=0.16, longer reviews 
are seen as more helpful) and negatively with `average_rating` (r=-0.12, slightly 
longer reviews tend to accompany lower-rated books). This confirms that **text 
content is the primary signal** for this classification task, not tabular metadata.

### Top Words by Sentiment Class
After removing HTML artifacts (`<br>` tags, HTML entities) from the raw review 
text, word-frequency analysis revealed:
- Negation words (**"don't", "didn't"**) appear almost exclusively in negative reviews
- **"great"** is a strong positive-only marker
- Common words (*book, read, story, author*) dominate across all classes, 
  suggesting bigrams or a custom tokenizer preserving negations (e.g. "didn't like") 
  would improve signal quality over single-word (unigram) features

### Verified Purchase vs. Sentiment
Verified purchases show slightly more polarized sentiment (6.9% negative, 85.5% 
positive) compared to non-verified ones (5.9% negative, 83.5% positive, but 10.6% 
neutral). This suggests customers who actually paid for the product tend to have 
stronger opinions — either very satisfied or genuinely disappointed — while 
non-verified reviewers are more likely to stay neutral.

### Data Quality
One extreme price outlier ($4,975.50) was investigated and found to be a 
legitimate rare/collectible book listing rather than a parsing error, and was 
therefore retained in the dataset.
