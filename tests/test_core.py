import pandas as pd
from src.nlp import clean, suspicious, summarize

def test_clean_keeps_negation():
    assert "not" in clean("This is NOT good!!!").split()

def test_suspicious_flags_promo():
    df = pd.DataFrame({"review": ["BUY NOW!!! BEST EVER!!!", "Solid build, works well for me."], "user": ["a", "b"], "rating": [5, 4]})
    assert len(suspicious(df)) == 1

def test_summarize_nonempty():
    assert summarize(["The battery is great. The screen is bright and sharp."])
