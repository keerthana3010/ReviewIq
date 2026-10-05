# ReviewIQ
Integrated e-commerce review analysis: **Sentiment + Aspect Analysis + Suspicious-Review Detection + Summarisation + Customer Insights + Real-Time Dashboard**.

## Structure
```
app.py            Streamlit UI (login, products, dashboard, live analyzer, model lab)
train.py          Offline training CLI -> models/best.joblib, reports/metrics.csv
src/nlp.py        Preprocessing, suspicious-review rules, extractive summary
src/ml.py         TF-IDF + 5 models, evaluation, aspect sentiment, persistence
src/data.py       Demo data generator, real-CSV loader
tests/            pytest unit tests
```
## Run
```
pip install -r requirements.txt
python train.py --csv data/reviews.csv --review-col text --rating-col stars   # optional, real data
streamlit run app.py        # demo admin: admin / Admin@123  (change it!)
pytest
```
## Method
Reviews -> cleaning (stopwords kept for negations, light stemming) -> TF-IDF (1-2 grams) -> NB / LR / RF / SVM / XGBoost -> best by weighted F1.
Labels come from ratings (1-2 Negative, 3 Neutral, 4-5 Positive). Aspects use keyword matching per clause, with the best model scoring each clause.

## Limitations
Results depend on dataset quality and diversity. TF-IDF models struggle with sarcasm and mixed opinions. Suspicious-review detection is rule-based and indicates risk, not proof. Bundled demo data is synthetic, so its scores are optimistic.
