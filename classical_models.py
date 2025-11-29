# classical_models.py
# 📦 Classical ML models (Word2Vec + LR/SVM/DT/RF)

import os
import numpy as np
import joblib
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def build_word2vec(df, vector_size=100):
    sentences = [t.split() for t in df["clean_text"]]
    w2v = Word2Vec(
        sentences,
        vector_size=vector_size,
        window=5,
        min_count=1,
        workers=4
    )
    return w2v


def vectorize_text(text, w2v, vector_size=100):
    words = text.split()
    vecs = [w2v.wv[w] for w in words if w in w2v.wv]
    if not vecs:
        return np.zeros(vector_size)
    return np.mean(vecs, axis=0)


def prepare_classical_data(df, w2v, test_size=0.2):
    df = df.copy()
    df["vectors"] = df["clean_text"].apply(lambda t: vectorize_text(t, w2v))
    X = np.vstack(df["vectors"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    return X_train, X_test, y_train, y_test


def train_classical_models(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(kernel="linear", probability=True),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=100)
    }

    results = {}
    for name, model in models.items():
        print(f"\n🔹 Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[name] = acc
        print(f"{name} Accuracy: {acc:.2f}")
        print(classification_report(y_test, preds))

    return models, results


def save_classical_models(models, out_dir="models"):
    os.makedirs(out_dir, exist_ok=True)
    name_map = {
        "Logistic Regression": "Logistic_Regression.pkl",
        "SVM": "SVM.pkl",
        "Decision Tree": "Decision_Tree.pkl",
        "Random Forest": "Random_Forest.pkl"
    }
    for name, model in models.items():
        path = os.path.join(out_dir, name_map[name])
        joblib.dump(model, path)
        print(f"💾 Saved {name} to {path}")


if __name__ == "__main__":
    # Example standalone run:
    from preprocessing_and_eda import load_dataset, apply_cleaning

    df = load_dataset()
    df = apply_cleaning(df)

    w2v = build_word2vec(df)
    X_train, X_test, y_train, y_test = prepare_classical_data(df, w2v)
    models, results = train_classical_models(X_train, X_test, y_train, y_test)
    save_classical_models(models)
    print("\n📊 Classical Results:", results)
