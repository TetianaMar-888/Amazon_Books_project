import os
import streamlit as st
import torch
import gdown
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

st.set_page_config(page_title="Book Review Sentiment Classifier", page_icon="📚")

LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}
LABEL_EMOJI = {"negative": "😞", "neutral": "😐", "positive": "😊"}


@st.cache_resource
def load_model():
    model_path = "models/distilbert_model"
    safetensors_path = f"{model_path}/model.safetensors"

    if not os.path.exists(safetensors_path):
        os.makedirs(model_path, exist_ok=True)
        gdown.download(
            id="1EYgMNTw2Q4Trp6cuduyuK-QwdB2_90nx",
            output=safetensors_path,
            quiet=False
        )

    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    model.eval()
    return model, tokenizer


model, tokenizer = load_model()

st.title("📚 Book Review Sentiment Classifier")
st.markdown(
    "Fine-tuned DistilBERT model trained on Amazon Book Reviews. "
    "Enter a review below to predict its sentiment."
)

review_text = st.text_area(
    "Enter a book review:", height=150,
    placeholder="e.g. This book completely changed how I think about..."
)

if st.button("Classify Sentiment", type="primary"):
    if not review_text.strip():
        st.warning("Please enter a review first.")
    else:
        with st.spinner("Analyzing..."):
            inputs = tokenizer(
                review_text, truncation=True, padding="max_length",
                max_length=256, return_tensors="pt"
            )
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)[0]
                pred_idx = torch.argmax(probs).item()

        sentiment = LABEL_MAP[pred_idx]
        confidence = probs[pred_idx].item()

        st.markdown(f"## {LABEL_EMOJI[sentiment]} **{sentiment.upper()}**")
        st.progress(confidence)
        st.caption(f"Confidence: {confidence*100:.1f}%")

        st.markdown("### Class probabilities")
        for idx, label in LABEL_MAP.items():
            st.write(f"{LABEL_EMOJI[label]} {label}: {probs[idx].item()*100:.1f}%")

st.markdown("---")
st.caption("Model: fine-tuned DistilBERT | [GitHub repo](https://github.com/yourusername/Amazon_Books_project)")
