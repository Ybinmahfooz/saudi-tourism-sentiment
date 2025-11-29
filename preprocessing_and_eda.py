# preprocessing_and_eda.py
# 🧹 Preprocessing + EDA (copied from main.py logic)

import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os


def load_dataset(path="data/dataset_augmented.xlsx"):
    print("📂 Loading dataset...")
    df = pd.read_excel(path)

    # normalize label column (same as main.py)
    if "manual labelling" in df.columns:
        df = df.rename(columns={"manual labelling": "label"})
    elif "zero-shot labelling" in df.columns:
        df = df.rename(columns={"zero-shot labelling": "label"})

    df = df[["text", "label"]].dropna().reset_index(drop=True)
    print(f"✅ Loaded {len(df)} samples.")
    print(df["label"].value_counts())
    return df


def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^أ-يa-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def apply_cleaning(df):
    df = df.copy()
    df["clean_text"] = df["text"].apply(clean_text)
    print("✅ Data cleaned successfully!")
    return df


def plot_label_distribution(df, out_path="results/sentiment_distribution.png"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(6, 4))
    sns.countplot(x="label", data=df, palette="viridis")
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment Label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"📊 Saved label distribution to {out_path}")


def plot_wordcloud(df, out_path="results/wordcloud.png", font_path="Amiri-Regular.ttf"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    all_words = " ".join(df["clean_text"])
    wc = WordCloud(
        width=900,
        height=450,
        background_color="white",
        font_path=font_path
    ).generate(all_words)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Arabic Tweet Word Cloud")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"🌥️ Saved word cloud to {out_path}")


if __name__ == "__main__":
    # Optional standalone test
    df = load_dataset()
    df = apply_cleaning(df)
    plot_label_distribution(df)
    plot_wordcloud(df)
