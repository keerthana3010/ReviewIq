"""Text preprocessing, suspicious-review rules and extractive summarisation."""
import re, numpy as np, pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

LABELS = ["Negative", "Neutral", "Positive"]
KEEP = {"not", "no", "never", "nor", "but", "very", "too", "cannot"}  # negations/intensifiers carry sentiment
STOP = set(ENGLISH_STOP_WORDS) - KEEP

def _stem(w):
    for s in ("ingly", "ing", "edly", "ed", "es", "s"):
        if w.endswith(s) and len(w) - len(s) >= 3: return w[: -len(s)]
    return w

def clean(t):
    """lowercase -> strip symbols -> drop stopwords (keep negations) -> light stemming."""
    t = re.sub(r"[^a-z\s]", " ", str(t).lower())
    return " ".join(_stem(w) for w in t.split() if w not in STOP)

def suspicious(df):
    s = pd.Series(0, index=df.index); why = pd.Series("", index=df.index)
    def flag(mask, pts, msg):
        nonlocal s, why; s = s + mask * pts; why = why + np.where(mask, msg + "; ", "")
    r = df.review.astype(str)
    flag(r.duplicated(keep=False), 2, "duplicate text")
    flag(r.str.len() < 15, 1, "very short")
    flag(r.str.count("!") >= 3, 2, "excess exclamation")
    flag(r.map(lambda t: sum(c.isupper() for c in t) / max(len(t), 1)) > 0.5, 2, "mostly caps")
    flag(r.str.contains(r"buy now|click|discount code|http|www", case=False), 3, "promotional")
    flag(df.groupby("user").review.transform("count") >= 5, 1, "prolific reviewer")
    out = df.assign(risk=s, reasons=why)
    return out[out.risk >= 3].sort_values("risk", ascending=False)

def summarize(texts, k=2):
    sents = [x.strip() for t in texts for x in re.split(r"(?<=[.!?])\s+", str(t)) if len(x.strip()) > 15]
    if not sents: return "Not enough reviews."
    freq = pd.Series(" ".join(map(clean, sents)).split()).value_counts()
    sc = {x: sum(freq.get(w, 0) for w in clean(x).split()) / (len(clean(x).split()) or 1) for x in set(sents)}
    return " ".join(sorted(sc, key=sc.get, reverse=True)[:k])
