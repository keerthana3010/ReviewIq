"""Offline training CLI:  python train.py --csv data/reviews.csv --review-col text --rating-col stars"""
import argparse, os, pandas as pd
from src.data import load_csv, make_demo
from src.ml import train, save
from src.nlp import LABELS

ap = argparse.ArgumentParser()
ap.add_argument("--csv"); ap.add_argument("--review-col", default="review"); ap.add_argument("--rating-col", default="rating")
a = ap.parse_args()
df = load_csv(a.csv, a.review_col, a.rating_col) if a.csv else make_demo()
df["sentiment"] = pd.cut(df.rating, [0, 2, 3, 5], labels=LABELS)
vec, models, res, cms = train(len(df), df)
os.makedirs("reports", exist_ok=True); os.makedirs("models", exist_ok=True)
res.to_csv("reports/metrics.csv", index=False)
for n, cm in cms.items(): pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(f"reports/cm_{n.replace(' ', '_')}.csv")
save(vec, models[res.Model[0]], res.Model[0])
print(res.round(3).to_string(index=False)); print("Best:", res.Model[0], "-> models/best.joblib")
