# ======================================================
# 🧠 Sentiment Analysis Demo - AraBERT, mBERT, XLM-R
# Author: Yazeed Bin Mahfooz
# ======================================================

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# -------------------------------
# Load models
# -------------------------------
models = {
    "AraBERT": "models/AraBERT_final",      # fine-tuned
    "mBERT": "models/mBERT_final",          # fine-tuned
    "XLM-R": "models/XLMR_final"            # fine-tuned
}

print("\n✅ All fine-tuned models loaded successfully.\n")



label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

print("=" * 70)
print("🧠  Arabic Sentiment Analysis Demo (AraBERT, mBERT, XLM-R)")
print("=" * 70)
print("💬 Type an Arabic sentence to analyze its sentiment")
print("💡 Type 'exit' to quit")
print("=" * 70)
print("\n🚀 Loading models...\n")

loaded_models = {}
for name, path in models.items():
    try:
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        model.eval()
        loaded_models[name] = (tokenizer, model)
        print(f"✅ Loaded {name} from {path}")
    except Exception as e:
        print(f"⚠️ Could not load {name}: {e}")

print("\n✅ All available models loaded.\n")

# -------------------------------
# Interactive input loop
# -------------------------------
while True:
    text = input("\n📝 Enter an Arabic sentence: ")
    if text.lower() == "exit":
        print("👋 Exiting demo.")
        break

    print("-" * 65)
    print(f"🔍 Sentence: {text}")

    for name, (tokenizer, model) in loaded_models.items():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            label_id = torch.argmax(probs).item()
            confidence = probs[0][label_id].item()

        label = label_map.get(label_id, "Unknown")
        print(f"{name:10}: {label} ({confidence*100:.2f}%)")
    print("-" * 65)
