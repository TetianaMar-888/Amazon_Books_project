"""Model training utilities for baseline, XGBoost, and DistilBERT."""

import time
import torch
import torch.nn as nn
import xgboost as xgb
from torch.utils.data import Dataset as TorchDataset
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments

LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def train_baseline(X_train, y_train):
    model = DummyClassifier(strategy="most_frequent")
    model.fit(X_train, y_train)
    return model


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    start = time.time()
    model.fit(X_train, y_train)
    return model, time.time() - start


def train_xgboost(X_train, y_train, params=None):
    """Train an XGBoost classifier with class-balanced sample weights."""
    default_params = dict(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        objective="multi:softmax", num_class=3, eval_metric="mlogloss",
        tree_method="hist", random_state=42, n_jobs=-1
    )
    if params:
        default_params.update(params)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train_enc)

    model = xgb.XGBClassifier(**default_params)
    start = time.time()
    model.fit(X_train, y_train_enc, sample_weight=sample_weights)
    return model, le, time.time() - start


class ReviewsDataset(TorchDataset):
    """PyTorch Dataset for DistilBERT fine-tuning / inference."""

    def __init__(self, texts, tokenizer, labels=None, max_length=256):
        self.encodings = tokenizer(
            list(texts), truncation=True, padding="max_length",
            max_length=max_length, return_tensors="pt"
        )
        self.labels = [LABEL_MAP[l] for l in labels] if labels is not None else None

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])
        return item


class WeightedTrainer(Trainer):
    """HuggingFace Trainer subclass supporting a custom class-weighted loss."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def train_distilbert(train_ds, val_ds, class_weights, output_dir, epochs=3):
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=3
    )
    training_args = TrainingArguments(
        output_dir=output_dir, num_train_epochs=epochs,
        per_device_train_batch_size=16, per_device_eval_batch_size=32,
        eval_strategy="epoch", save_strategy="epoch", logging_steps=50,
        load_best_model_at_end=True, metric_for_best_model="macro_f1",
        report_to="none"
    )
    trainer = WeightedTrainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        class_weights=class_weights
    )
    start = time.time()
    trainer.train()
    return trainer, time.time() - start


def classify_sentiment_llm(text, client, model="claude-sonnet-4-6", max_retries=3):
    """Zero-shot sentiment classification via the Claude API."""
    import re

    prompt = f"""Classify the sentiment of this book review as exactly one word: negative, neutral, or positive.

Review: "{text[:1000]}"

Respond with only one word: negative, neutral, or positive."""

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model, max_tokens=10,
                messages=[{"role": "user", "content": prompt}]
            )
            answer = re.sub(r"[^a-z]", "", response.content[0].text.strip().lower())
            return answer if answer in LABEL_MAP else "positive"
        except Exception:
            if attempt == max_retries - 1:
                return "positive"
            time.sleep(2 ** attempt)