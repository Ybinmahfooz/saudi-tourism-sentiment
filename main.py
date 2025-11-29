# ==========================================================
#  Graduation Project - Sentiment Analysis on Saudi Tourism
#  Author: Yazeed Bin-Mahfooz
# ==========================================================

# ==========================================================
# 1️⃣ IMPORT REQUIRED LIBRARIES
# ==========================================================
import os
import re
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from gensim.models import Word2Vec
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

# disable telemetry + W&B tracking
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["WANDB_DISABLED"] = "true"

# ==========================================================
# 2️⃣ LOAD DATASET
# ==========================================================
print("📂 Loading dataset...")

df = pd.read_excel("data/dataset_augmented.xlsx")

# normalize label column
if "manual labelling" in df.columns:
    df = df.rename(columns={"manual labelling": "label"})
elif "zero-shot labelling" in df.columns:
    df = df.rename(columns={"zero-shot labelling": "label"})

df = df[["text", "label"]].dropna().reset_index(drop=True)

print(f"✅ Loaded {len(df)} samples.")
print(df["label"].value_counts())

# ==========================================================
# 3️⃣ CLEAN TEXT DATA
# ==========================================================
def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^أ-يa-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = df["text"].apply(clean_text)
print("✅ Data cleaned successfully!")

# ==========================================================
# 4️⃣ EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================================
plt.figure(figsize=(6, 4))
sns.countplot(x="label", data=df, palette="viridis")
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment Label")
plt.ylabel("Count")
plt.savefig("results/sentiment_distribution.png", dpi=300)
plt.close()

arabic_font = "Amiri-Regular.ttf"
all_words = " ".join(df["clean_text"])
wordcloud = WordCloud(width=900, height=450, background_color="white",
                      font_path=arabic_font).generate(all_words)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Arabic Tweet Word Cloud")
plt.savefig("results/wordcloud.png", dpi=300)
plt.close()

# ==========================================================
# 5️⃣ WORD2VEC ENCODING + CLASSICAL ML MODELS
# ==========================================================
sentences = [t.split() for t in df["clean_text"]]
w2v = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

def vectorize_text(txt):
    words = txt.split()
    vecs = [w2v.wv[w] for w in words if w in w2v.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(100)

df["vectors"] = df["clean_text"].apply(vectorize_text)
X = np.vstack(df["vectors"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

ml_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(kernel="linear"),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}

results = {}
for name, model in ml_models.items():
    print(f"\n🔹 Training {name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"{name} Accuracy: {acc:.2f}")
    print(classification_report(y_test, preds))

# ==========================================================
# 6️⃣ BERT MODELS 
# ==========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

deep_models = {
    "AraBERT": "models/AraBERT_final",          
    "mBERT": "bert-base-multilingual-cased",    
    "XLM-R": "xlm-roberta-base"                 
}


train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["clean_text"].tolist(), df["label"].tolist(), test_size=0.2, random_state=42
)

le = LabelEncoder()
train_labels = le.fit_transform(train_labels)
test_labels = le.transform(test_labels)

def make_dataset(tokenizer, texts, labels):
    enc = tokenizer(texts, truncation=True, padding=True, max_length=128)
    class DS(torch.utils.data.Dataset):
        def __init__(self, enc, lbl): self.enc, self.lbl = enc, lbl
        def __getitem__(self, i):
            item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
            item["labels"] = torch.tensor(int(self.lbl[i]))
            return item
        def __len__(self): return len(self.lbl)
    return DS(enc, labels)

# ==========================================================
# 7️⃣ TRAIN EACH MODEL SEQUENTIALLY
# ==========================================================
for model_name, checkpoint in deep_models.items():
    print(f"\n🤖 Training {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=len(le.classes_)
    )

    train_ds = make_dataset(tokenizer, train_texts, train_labels)
    test_ds = make_dataset(tokenizer, test_texts, test_labels)

    # training setup
    training_args = TrainingArguments(
        output_dir=f"results/{model_name}",
        num_train_epochs=4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.05,
        logging_steps=100,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds
    )

    trainer.train()
    preds = trainer.predict(test_ds)
    acc = np.mean(np.argmax(preds.predictions, axis=1) == np.array(test_labels))
    results[model_name] = acc
    print(f"✅ {model_name} Accuracy: {acc:.2f}")

        # ==========================================================
    # 💾 SAVE TRAINED MODEL CHECKPOINTS
    # ==========================================================
    if model_name == "mBERT":
        model.save_pretrained("models/mBERT_final")
        tokenizer.save_pretrained("models/mBERT_final")
        print("💾 mBERT model saved to models/mBERT_final/")

    if model_name == "XLM-R":
        model.save_pretrained("models/XLMR_final")
        tokenizer.save_pretrained("models/XLMR_final")
        print("💾 XLM-R model saved to models/XLMR_final/")


     # fine-tune only AraBERT
     if model_name == "AraBERT":
         print("\n🔁 Fine-tuning AraBERT with optimized settings...")
         ft_args = TrainingArguments(
             output_dir="results/AraBERT_finetuned",
             num_train_epochs=7,
             per_device_train_batch_size=8,
             per_device_eval_batch_size=8,
             learning_rate=1.5e-5,
             weight_decay=0.01,
             warmup_ratio=0.1,
             evaluation_strategy="epoch",
             save_strategy="no",
             logging_dir="logs/AraBERT_finetune",
             logging_steps=50
         )
         ft_trainer = Trainer(model=model, args=ft_args,
                              train_dataset=train_ds, eval_dataset=test_ds)
         ft_trainer.train()

         model.save_pretrained("models/AraBERT_final")
         tokenizer.save_pretrained("models/AraBERT_final")
         print("💾 Fine-tuned AraBERT saved to models/AraBERT_final/")

# ==========================================================
# 8️⃣ SAVE RESULTS AND VISUALIZE
# ==========================================================
plt.figure(figsize=(10, 5))
sns.barplot(x=list(results.keys()), y=list(results.values()), palette="coolwarm")
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results/final_results.png", dpi=300)
plt.close()

summary = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])
summary.to_csv("results/final_accuracy.csv", index=False)

print("\n📊 Final Results:")
print(summary)

# ==========================================================
# 9️⃣ SAVE TRAINING LOG
# ==========================================================
with open("logs/training_log.txt", "w", encoding="utf-8") as f:
    for k, v in results.items():
        f.write(f"{k}: {v:.4f}\n")

print("\n✅ Training completed successfully. Logs saved in /logs/.")
