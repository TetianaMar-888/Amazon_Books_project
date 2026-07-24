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

## Baseline Model

A `most-frequent-class` baseline achieves 84.6% accuracy — but this is misleading:
it never predicts `negative` or `neutral` at all (0% recall for both), simply because
`positive` dominates the dataset. **Macro F1 of 0.31** exposes this weakness clearly, 
confirming our earlier EDA decision to use macro F1 as the primary evaluation metric 
rather than accuracy. Any model that beats this baseline must demonstrate real 
predictive power on the minority classes, not just overall accuracy.

## Model 1: Logistic Regression (TF-IDF + tabular features)

Logistic Regression with `class_weight="balanced"` improves macro F1 from 0.31 
(baseline) to **0.49** — a substantial gain in the model's ability to detect 
minority classes (negative recall: 0% → 51%, neutral recall: 0% → 40%).

This comes at the cost of overall accuracy (84.6% → 72.2%), since the model now 
misclassifies some `positive` reviews as `negative`/`neutral` in exchange for much 
better minority-class detection. This is the expected and desirable trade-off when 
optimizing for macro F1 rather than accuracy — for a real business use case 
(e.g., flagging negative reviews for customer support), catching more true 
negatives matters more than raw accuracy on the already-easy majority class.

Precision for `negative` (0.27) and `neutral` (0.20) remains low, meaning the model 
still generates many false positives for these classes — an area for improvement 
with more sophisticated models (fine-tuned transformers, ensemble methods).

## Model 2: XGBoost (TF-IDF + tabular features)

XGBoost was trained on the same feature set as Logistic Regression (TF-IDF text 
vectors + tabular features), using `sample_weight` to address class imbalance.

**Initial run** (n_estimators=300, max_depth=6, learning_rate=0.1): 
macro F1 = **0.524**, accuracy = 77.2%.

**Hyperparameter tuning:** A `RandomizedSearchCV` (4 candidates × 2-fold CV, 
scoring=`f1_macro`) was run over `n_estimators`, `max_depth`, and `learning_rate`. 
Due to compute constraints in Colab (a full search on the 35k training set was 
computationally prohibitive), tuning was performed on a **10k stratified subsample**. 
The best configuration found (n_estimators=200, max_depth=5, learning_rate=0.2) 
achieved a CV macro F1 of 0.573 on the subsample.

**Limitation:** When this best configuration was retrained on the full 35k 
training set and evaluated on the full validation set, it performed slightly 
worse (macro F1 = 0.514) than the original untuned configuration (macro F1 = 0.524). 
This suggests that hyperparameters optimal on a subsample do not always transfer 
perfectly to the full-scale model — a known trade-off of this tuning strategy 
under compute constraints.

**Final choice:** Given this result, the original configuration 
(n_estimators=300, max_depth=6, learning_rate=0.1, macro F1 = 0.524) was retained 
as the representative XGBoost result for comparison with other models, since it 
achieved the best validation performance in practice.

| Model | Params | Val Accuracy | Val Macro F1 | Train Time |
|---|---|---|---|---|
| XGBoost (initial) | n_estimators=300, max_depth=6, lr=0.1 | 77.2% | **0.524** | ~16 min |
| XGBoost (tuned on 10k subsample) | n_estimators=200, max_depth=5, lr=0.2 | 76.0% | 0.514 | ~10 min |
