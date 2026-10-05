"""Model training, evaluation, persistence, sentiment + aspect prediction."""
import re, numpy as np, pandas as pd, joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None
from .nlp import clean, LABELS
from .data import ASPECTS

def save(vec, model, name, path="models/best.joblib"):
    joblib.dump({"vec": vec, "model": model, "name": name}, path)

def load(path="models/best.joblib"):
    d = joblib.load(path); return d["vec"], d["model"], d["name"]

def train(n, _df):
    X = _df.review.map(clean); y = _df.sentiment.astype(str)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    A, B = vec.fit_transform(Xtr), vec.transform(Xte)
    models = {"Naive Bayes": MultinomialNB(), "Logistic Regression": LogisticRegression(max_iter=1000),
              "Random Forest": RandomForestClassifier(150, random_state=42), "SVM": LinearSVC()}
    if XGBClassifier: models["XGBoost"] = XGBClassifier(n_estimators=150, eval_metric="mlogloss")
    res, cms, fitted = [], {}, {}
    for name, m in models.items():
        enc = {l: i for i, l in enumerate(LABELS)}
        m.fit(A, ytr.map(enc) if name == "XGBoost" else ytr)
        p = m.predict(B)
        if name == "XGBoost": p = np.array(LABELS)[p]
        pr, rc, f1, _ = precision_recall_fscore_support(yte, p, average="weighted", zero_division=0)
        res.append([name, accuracy_score(yte, p), pr, rc, f1])
        cms[name] = confusion_matrix(yte, p, labels=LABELS); fitted[name] = m
    r = pd.DataFrame(res, columns=["Model", "Accuracy", "Precision", "Recall", "F1"]).sort_values("F1", ascending=False)
    return vec, fitted, r.reset_index(drop=True), cms

def predict(vec, models, best, texts):
    p = models[best].predict(vec.transform([clean(t) for t in texts]))
    return list(np.array(LABELS)[p]) if best == "XGBoost" else list(p)

def aspect_sentiments(text, vec, models, best):
    out = {}
    for c in re.split(r"[.!?;,]| but ", text):
        for a, kws in ASPECTS.items():
            if any(k in c.lower() for k in kws) and len(c.strip()) > 3:
                out[a] = predict(vec, models, best, [c])[0]
    return out
