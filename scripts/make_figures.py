"""Rebuild the README figures and numbers.

Run from the repo root with Fake.csv and True.csv in data/:  python scripts/make_figures.py
Retrains the same TF-IDF + logistic regression pipeline as fake_vs_real_kaggle.py, fits two comparison
models, repeats the run with the Reuters wire tags removed, and writes PNGs to docs/assets/.
Takes about 4 minutes, most of it in the TF-IDF bigram vectorizer.
"""
import os
import re
import string

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "docs", "assets")
os.makedirs(OUT, exist_ok=True)

BG, PANEL = "#0d0c0b", "#131211"
INK, INK2, INK3 = "#ede7de", "#aaa299", "#7a736c"
GRID = "#24221f"
EMBER, TEMPER, PURPLE, GREEN = "#d95926", "#3987e5", "#9085e9", "#199e70"
SERIES = [EMBER, TEMPER, PURPLE, GREEN, "#e8b44c", "#c0509c"]
HEAT = LinearSegmentedColormap.from_list("heat", ["#141110", "#2a1410", "#4b170c", "#7a1f0a", "#ad320c",
                                                  "#d9520f", "#f07a1e", "#f9a63b", "#fccf6b", "#fff0c2"])
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titleweight": "semibold",
    "axes.titlesize": 12.5, "axes.titlelocation": "left", "axes.titlepad": 12, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "xtick.color": INK3, "ytick.color": INK3, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 6, "ytick.major.pad": 6,
    "axes.axisbelow": True, "font.family": ["Segoe UI", "DejaVu Sans"], "lines.linewidth": 2,
    "lines.solid_capstyle": "round", "legend.frameon": False, "legend.fontsize": 9.5, "legend.labelcolor": INK2,
    "figure.dpi": 110, "savefig.dpi": 200,
})


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"wrote {path} ({os.path.getsize(path) / 1024:.0f} KB)")


def caption(fig, text):
    fig.text(0.012, -0.015, text, color=INK3, fontsize=8.5, ha="left", va="top")


# same cleaning as fake_vs_real_kaggle.py
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[{}]".format(re.escape(string.punctuation)), "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_wire(text):
    """Drop the 'CITY (Reuters) - ' dateline and any other mention of Reuters."""
    text = re.sub(r"^.{0,120}?\(Reuters\)\s*-\s*", "", str(text))
    return re.sub(r"(?i)reuters", "", text)


def fit_lr(train_text, test_text, y_train):
    vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words="english")
    a, b = vec.fit_transform(train_text), vec.transform(test_text)
    model = LogisticRegression(max_iter=300, solver="saga", random_state=42).fit(a, y_train)
    return vec, model, a, b


def main():
    fake = pd.read_csv(os.path.join(DATA, "Fake.csv"))
    real = pd.read_csv(os.path.join(DATA, "True.csv"))
    fake["label"], real["label"] = 1, 0
    df = pd.concat([fake, real], ignore_index=True)
    df["cleaned_text"] = df["text"].apply(clean_text)
    df["stripped_text"] = df["text"].apply(strip_wire).apply(clean_text)

    idx_train, idx_test = train_test_split(df.index, test_size=0.2, random_state=42, stratify=df["label"])
    y_train, y_test = df.loc[idx_train, "label"], df.loc[idx_test, "label"]
    n_test = len(idx_test)

    vec, lr, a, b = fit_lr(df.loc[idx_train, "cleaned_text"], df.loc[idx_test, "cleaned_text"], y_train)
    rows = {"LogisticRegression\n(repo model)": lr.predict(b)}
    rows["MultinomialNB"] = MultinomialNB().fit(a, y_train).predict(b)
    rows["LinearSVC"] = LinearSVC().fit(a, y_train).predict(b)
    vec2, lr2, _, b2 = fit_lr(df.loc[idx_train, "stripped_text"], df.loc[idx_test, "stripped_text"], y_train)
    rows["LogisticRegression\nretrained, tags removed"] = lr2.predict(b2)
    rows["repo model,\ntags removed at test time"] = lr.predict(vec.transform(df.loc[idx_test, "stripped_text"]))

    wire = real["text"].str.contains(r"\(Reuters\)").mean()
    print(f"articles {len(df)}  fake {len(fake)}  real {len(real)}  train {len(idx_train)}  test {n_test}")
    print(f"real articles with a '(Reuters)' dateline: {wire:.1%}   "
          f"fake: {fake['text'].str.contains(r'[(]Reuters[)]').sum()} articles")
    for marker in ["Featured image", "21st Century Wire"]:
        print(f"'{marker}' in fake: {fake['text'].str.contains(marker, case=False).sum()}  "
              f"in real: {real['text'].str.contains(marker, case=False).sum()}")
    print(f"fake articles with empty text: {(fake['text'].str.strip() == '').sum()}")
    print(f"rows whose text duplicates an earlier row: {df['text'].duplicated().sum()}")
    errors = {}
    for name, pred in rows.items():
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        errors[name] = n_test - np.trace(cm)
        print(f"{name.replace(chr(10), ' '):48s} acc {accuracy_score(y_test, pred):.4f}  errors {errors[name]:4d}  "
              f"real->fake {cm[0, 1]:4d}  fake->real {cm[1, 0]:4d}")

    names = vec.get_feature_names_out()
    coef = lr.coef_[0]
    order = np.argsort(coef)
    print("top real:", ", ".join(names[order[:12]]))
    print("top fake:", ", ".join(names[order[-12:]][::-1]))
    names2 = vec2.get_feature_names_out()
    order2 = np.argsort(lr2.coef_[0])
    print("top real without tags:", ", ".join(names2[order2[:12]]))

    # hero: confusion matrix + strongest weights
    cm = confusion_matrix(y_test, rows["LogisticRegression\n(repo model)"], labels=[0, 1])
    fig = plt.figure(figsize=(12.5, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.45], wspace=0.34)
    ax = fig.add_subplot(gs[0])
    ax.imshow(cm, cmap=HEAT, vmin=0, vmax=cm.max(), aspect="auto")
    ax.grid(False)
    for i in range(2):
        for j in range(2):
            v = cm[i, j]
            ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=15, fontweight="semibold",
                    color=BG if v > cm.max() * 0.55 else INK)
    ax.set_xticks([0, 1], ["real", "fake"])
    ax.set_yticks([0, 1], ["real", "fake"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    for s in ax.spines.values():
        s.set_visible(False)
    acc = accuracy_score(y_test, rows["LogisticRegression\n(repo model)"])
    ax.set_title(f"8,980 held-out articles · accuracy {acc:.3f}")

    ax = fig.add_subplot(gs[1])
    n = 12
    idx = np.concatenate([order[:n], order[-n:]])
    vals = coef[idx]
    y = np.arange(len(idx))
    ax.barh(y, vals, color=[TEMPER if v < 0 else EMBER for v in vals], height=0.72)
    ax.set_yticks(y, names[idx])
    ax.axvline(0, color=INK3, lw=0.8)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("logistic regression weight")
    ax.set_title("Strongest weights: the wire-service tag dominates")
    ax.legend(handles=[Patch(color=EMBER, label="pushes toward fake"),
                       Patch(color=TEMPER, label="pushes toward real")], loc="upper left")
    caption(fig, "Stratified 80/20 split (random_state=42). TF-IDF on unigrams and bigrams, 10,000 features, "
                 "English stop words removed.")
    save(fig, "hero.png")

    # data: class balance + length by class
    df["words"] = df["text"].astype(str).str.split().str.len()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1, 2.2], "wspace": 0.25})
    ax = axes[0]
    counts = [len(real), len(fake)]
    bars = ax.bar(["real", "fake"], counts, color=[TEMPER, EMBER], width=0.6)
    for bar, v in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 300, f"{v:,}", ha="center", color=INK2, fontsize=9.5)
    ax.set_ylim(0, max(counts) * 1.15)
    ax.set_ylabel("articles")
    ax.set_title("Class balance")
    ax = axes[1]
    bins = np.arange(0, 1501, 30)
    for lab, name, col in [(0, "real", TEMPER), (1, "fake", EMBER)]:
        w = df.loc[df.label == lab, "words"]
        ax.hist(w.clip(upper=1500), bins=bins, density=True, color=col, alpha=0.55,
                label=f"{name} (median {int(w.median())} words)")
    ax.set_xlabel("words per article (clipped at 1,500)")
    ax.set_ylabel("share of class")
    ax.set_title("Article length by class")
    ax.legend(loc="upper right")
    save(fig, "data.png")

    # models: test errors, zero baseline
    fig, ax = plt.subplots(figsize=(11, 4.8))
    labels = list(rows)
    vals = [errors[k] for k in labels]
    colors = [EMBER, TEMPER, GREEN, PURPLE, PURPLE]
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=colors, width=0.62)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 8, f"{v}  ({v / n_test:.1%})", ha="center",
                color=INK2, fontsize=9.5)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, max(vals) * 1.15)
    ax.set_ylabel(f"misclassified test articles (of {n_test:,})")
    ax.set_title("Test errors by model and by input")
    caption(fig, "Purple bars remove the 'CITY (Reuters) - ' dateline and every other mention of Reuters, "
                 "either before training or only at test time.")
    save(fig, "models.png")


if __name__ == "__main__":
    main()
