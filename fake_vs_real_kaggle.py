import os
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load two CSV files (Kaggle input folder when run there, otherwise ./data)
DATA_DIR = "/kaggle/input/fake-and-real-news-dataset"
if not os.path.isdir(DATA_DIR):
    DATA_DIR = "data"
fake_df = pd.read_csv(os.path.join(DATA_DIR, "Fake.csv"))
real_df = pd.read_csv(os.path.join(DATA_DIR, "True.csv"))

fake_df["label"] = 1  # 1 for Fake
real_df["label"] = 0  # 0 for Real

# Combine
df = pd.concat([fake_df, real_df], ignore_index=True)
print("Combined Shape:", df.shape)
print(df.head())

# check distribution of labels
plt.figure(figsize=(4,4))
df["label"].value_counts().plot(kind="bar", color=["red","blue"])
plt.title("Label Distribution (Fake=1, Real=0)")
plt.xlabel("Label")
plt.ylabel("Count")
plt.show()

# Check a numeric column if any
df["word_count"] = df["text"].apply(lambda x: len(str(x).split()))

plt.figure(figsize=(6,4))
df["word_count"].plot.hist(bins=20)
plt.title("Word Count Distribution")
plt.xlabel("Word Count")
plt.ylabel("Frequency")
plt.show()

# Create a simple cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[{}]".format(re.escape(string.punctuation)), "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Apply cleaning
df["cleaned_text"] = df["text"].apply(clean_text)

# Features and label
X = df["cleaned_text"]
y = df["label"]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Vectorize text
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,2),
    stop_words="english"
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=300, solver="saga", random_state=42)
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Real","Fake"]))

submission = pd.DataFrame({
    "Index": X_test.index,
    "PredictedLabel": y_pred
})
submission.to_csv("fake_real_submission.csv", index=False)
print("Created fake_real_submission.csv.")