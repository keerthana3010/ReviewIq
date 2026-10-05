"""Demo data generation and dataset loading."""
import random, datetime as dt, pandas as pd

PRODUCTS = ["Aero Wireless Headphones", "PixelPro Smartphone", "SmartFit Watch",
            "BrewMaster Coffee Maker", "TrailRunner Shoes", "UltraBook Laptop"]

def load_csv(path, review_col="review", rating_col="rating"):
    """Load a real dataset (Amazon/Flipkart/Kaggle) and map it to ReviewIQ's schema."""
    u = pd.read_csv(path).rename(columns={review_col: "review", rating_col: "rating"})
    u = u.dropna(subset=["review", "rating"]).drop_duplicates("review")
    u["rating"] = u.rating.astype(float).round().clip(1, 5).astype(int)
    u["product"] = u["product"] if "product" in u else random.Random(1).choices(PRODUCTS, k=len(u))
    u["user"] = u["user"] if "user" in u else "anon"
    u["date"] = u["date"] if "date" in u else dt.date.today().isoformat()
    return u[["product", "user", "rating", "review", "date"]].reset_index(drop=True)

ASPECTS = {
    "Quality": ["quality", "build", "material", "finish"],
    "Price": ["price", "cost", "value", "expensive", "cheap", "money"],
    "Delivery": ["delivery", "shipping", "arrived", "courier", "late"],
    "Packaging": ["packaging", "package", "box", "packed"],
    "Durability": ["durable", "durability", "broke", "lasted", "sturdy", "last"],
    "Performance": ["performance", "battery", "speed", "works", "sound", "fast", "slow"],
}

PH = {
    "Quality": (["The quality is excellent and the build feels premium.", "Great material and finish."], ["The quality is poor and the build feels cheap.", "Terrible finish and flimsy material."]),
    "Price": (["Great value for the price.", "Worth every penny, very affordable."], ["Way too expensive for what it offers.", "Not worth the money."]),
    "Delivery": (["Delivery was fast and on time.", "Shipping was quick."], ["Delivery was very late.", "The courier delayed shipping for days."]),
    "Packaging": (["Packaging was neat and secure.", "The box was well packed."], ["Packaging was damaged and sloppy.", "The box arrived crushed."]),
    "Durability": (["Very durable and sturdy.", "It has lasted months without issues."], ["It broke within a week.", "Not durable at all."]),
    "Performance": (["Performance is fast and smooth.", "Battery life is superb."], ["Performance is slow and laggy.", "Battery drains quickly."]),
}

def make_demo(n=900):
    rnd = random.Random(7); rows = []
    for _ in range(n):
        rating = rnd.choices([1, 2, 3, 4, 5], [10, 12, 18, 30, 30])[0]
        asp = rnd.sample(list(PH), rnd.choice([1, 2, 3]))
        parts = []
        for i, a in enumerate(asp):
            pos, neg = PH[a]
            pick = pos if rating >= 4 else neg if rating <= 2 else (pos if i % 2 == 0 else neg)
            parts.append(rnd.choice(pick))
        text = " ".join(parts)
        if rating == 3: text += " It is okay overall."
        rows.append((rnd.choice(list(PRODUCTS)), f"cust{rnd.randint(1, 300)}", rating, text,
                     (dt.date(2026, 1, 1) + dt.timedelta(days=rnd.randint(0, 270))).isoformat()))
    # a few fake-looking reviews for the suspicious detector
    for k in range(12):
        rows.append((rnd.choice(list(PRODUCTS)), "promo_user", 5, "BEST PRODUCT EVER!!! BUY NOW!!! Amazing!!!", "2026-09-01"))
    return pd.DataFrame(rows, columns=["product", "user", "rating", "review", "date"])
