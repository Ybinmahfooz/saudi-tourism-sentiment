# deep_models.py
# 🤖 Deep models: AraBERT, mBERT, XLM-R

import os
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["WANDB_DISABLED"] = "true"


def make_dataset(tokenizer, texts, labels):
    enc = tokenizer(texts, truncation=True, padding=True, max_length=128)

    class DS(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
            item["labels"] = torch.tensor(int(self.labels[idx]))
            return item

        def __len__(self):
            return len(self.labels)

    return DS(enc, labels)


def train_deep_models(df, initial_results=None):
    if initial_results is None:
        initial_results = {}

    deep_models = {
        "AraBERT": "aubmindlab/bert-base-arabertv2",
        "mBERT": "bert-base-multilingual-cased",
        "XLM-R": "xlm-roberta-base"
    }

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df["clean_text"].tolist(),
        df["label"].tolist(),
        test_size=0.2,
        random_state=42
    )

    le = LabelEncoder()
    train_labels = le.fit_transform(train_labels)
    test_labels = le.transform(test_labels)

    results = dict(initial_results)

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    for model_name, checkpoint in deep_models.items():
        print(f"\n🤖 Training {model_name}...")

        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint, num_labels=len(le.classes_)
        )

        train_ds = make_dataset(tokenizer, train_texts, train_labels)
        test_ds = make_dataset(tokenizer, test_texts, test_labels)

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
        acc = np.mean(
            np.argmax(preds.predictions, axis=1) == np.array(test_labels)
        )
        results[model_name] = acc
        print(f"✅ {model_name} Accuracy: {acc:.2f}")

        # extra saving like in main.py
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
            ft_trainer = Trainer(
                model=model,
                args=ft_args,
                train_dataset=train_ds,
                eval_dataset=test_ds
            )
            ft_trainer.train()

            model.save_pretrained("models/AraBERT_final")
            tokenizer.save_pretrained("models/AraBERT_final")
            print("💾 Fine-tuned AraBERT saved to models/AraBERT_final/")
        elif model_name == "mBERT":
            model.save_pretrained("models/mBERT_final")
            tokenizer.save_pretrained("models/mBERT_final")
            print("💾 mBERT model saved to models/mBERT_final/")
        elif model_name == "XLM-R":
            model.save_pretrained("models/XLMR_final")
            tokenizer.save_pretrained("models/XLMR_final")
            print("💾 XLM-R model saved to models/XLMR_final/")

    return results


if __name__ == "__main__":
    # Example standalone usage
    from preprocessing_and_eda import load_dataset, apply_cleaning

    df = load_dataset()
    df = apply_cleaning(df)
    results = train_deep_models(df)
    print("\n📊 Deep model results:", results)
