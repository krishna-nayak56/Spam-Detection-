"""
Train an SMS spam classifier using a public SMS Spam Collection dataset.
If the dataset cannot be downloaded, a small built-in teaching dataset is used.
The fallback is only for demonstration and will not be as reliable as the full dataset.
"""
from pathlib import Path
import io
import zipfile
import urllib.request

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "spam_model.joblib"
DATA_PATH = ROOT / "SMSSpamCollection"

# UCI SMS Spam Collection dataset (label<TAB>message)
DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

def load_data():
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH, sep="\\t", header=None, names=["label", "message"], encoding="utf-8")
        return df

    try:
        print("Downloading public SMS Spam Collection dataset...")
        with urllib.request.urlopen(DATA_URL, timeout=15) as response:
            raw_zip = response.read()
        with zipfile.ZipFile(io.BytesIO(raw_zip)) as zf:
            filename = next(name for name in zf.namelist() if name.endswith("SMSSpamCollection"))
            content = zf.read(filename)
        DATA_PATH.write_bytes(content)
        df = pd.read_csv(DATA_PATH, sep="\\t", header=None, names=["label", "message"], encoding="utf-8")
        return df
    except Exception as exc:
        print(f"Dataset download unavailable ({exc}). Using small built-in demo dataset.")
        examples = [
            ("ham", "Hey, are we meeting at college tomorrow?"),
            ("ham", "Please call me when you reach home."),
            ("ham", "I will be late by ten minutes."),
            ("ham", "Can you send me the notes from class?"),
            ("ham", "Your appointment is confirmed for Monday."),
            ("ham", "Thanks for helping me with the assignment."),
            ("ham", "Are you free this evening?"),
            ("ham", "The train will arrive at platform two."),
            ("ham", "Happy birthday! Have a wonderful day."),
            ("ham", "I am on my way, see you soon."),
            ("ham", "Please bring your notebook tomorrow."),
            ("ham", "Your order has been shipped and will arrive soon."),
            ("spam", "Congratulations! You have won a free prize. Call now to claim."),
            ("spam", "URGENT! You have been selected to win cash. Reply now."),
            ("spam", "Claim your free entry in our prize draw today."),
            ("spam", "You won a reward! Click the link to collect your prize."),
            ("spam", "Limited time offer! Buy now and save big."),
            ("spam", "Winner! Call this number immediately to claim your cash."),
            ("spam", "Get a free ringtone by texting WIN to 80000."),
            ("spam", "Congratulations, claim your exclusive bonus now."),
            ("spam", "You have been chosen for a special cash offer."),
            ("spam", "Urgent: your account is selected for a free reward."),
            ("spam", "Earn money fast from home. Sign up today."),
            ("spam", "Act now! Exclusive offer expires tonight."),
            ("ham", "Did you finish the project report?"),
            ("ham", "Mom said dinner is ready."),
            ("ham", "Let's study together after lunch."),
            ("ham", "I have shared the presentation file."),
            ("spam", "You are a lucky winner. Call now for your prize."),
            ("spam", "Free offer just for you. Reply YES to subscribe."),
        ]
        return pd.DataFrame(examples, columns=["label", "message"])

def main():
    df = load_data()
    df = df.dropna(subset=["label", "message"])
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df["message"] = df["message"].astype(str)
    df = df[df["label"].isin(["ham", "spam"])].drop_duplicates("message")

    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))),
        ("classifier", MultinomialNB(alpha=0.5)),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\\n=== SMS SPAM DETECTION MODEL REPORT ===")
    print(f"Dataset messages: {len(df)}")
    print(f"Training messages: {len(X_train)} | Testing messages: {len(X_test)}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print("\\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix (labels: ham, spam):")
    print(confusion_matrix(y_test, predictions, labels=["ham", "spam"]))

    joblib.dump(model, MODEL_PATH)
    print(f"\\nSaved trained model to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
