"""ReviewIQ - integrated e-commerce review analysis system (Streamlit)."""
import re, random, sqlite3, hashlib, os, datetime as dt
import numpy as np, pandas as pd, streamlit as st
import plotly.express as px
from PIL import Image, ImageDraw
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
 
from src.nlp import clean, suspicious, summarize
from src.ml import train, predict, aspect_sentiments
from src.data import make_demo, ASPECTS
train = st.cache_resource(train)
make_demo = st.cache_data(make_demo)
st.set_page_config(page_title="ReviewIQ", page_icon="📊", layout="wide")
px.defaults.template = "plotly_white"   # light charts
if not hasattr(st, "_orig_plotly_chart"):   # wrap only once (Streamlit re-runs this script)
    st._orig_plotly_chart = st.plotly_chart
st.plotly_chart = lambda fig, **k: st._orig_plotly_chart(fig, **{**k, "theme": None})
_cfg = os.path.join(os.path.dirname(__file__), ".streamlit", "config.toml")
if not os.path.exists(_cfg):   # light theme for tables/inputs (applies after restart)
    os.makedirs(os.path.dirname(_cfg), exist_ok=True)
    open(_cfg, "w").write('[theme]\nbase = "light"\nprimaryColor = "#6366f1"\nbackgroundColor = "#f8fafc"\nsecondaryBackgroundColor = "#ffffff"\ntextColor = "#1e293b"\n')
LABELS = ["Negative", "Neutral", "Positive"]
COLORS = {"Positive": "#22c55e", "Neutral": "#f59e0b", "Negative": "#ef4444"}
DB = os.path.join(os.path.dirname(__file__), "reviewiq.db")
 
st.markdown("""<style>
.stApp{background:linear-gradient(135deg,#f8fafc 0%,#eef2ff 100%);color:#1e293b}
[data-testid=stSidebar]{background:#ffffff;border-right:1px solid #e2e8f0}
h1,h2,h3{color:#4f46e5}
.card{background:#ffffff;border-radius:14px;padding:16px;border:1px solid #e2e8f0;box-shadow:0 2px 8px rgba(15,23,42,.06);margin-bottom:10px;color:#1e293b}
.kpi{font-size:2rem;font-weight:700;color:#4f46e5}
div.stButton>button{background:linear-gradient(90deg,#6366f1,#8b5cf6);color:#fff;border:0;border-radius:8px}
[data-testid=stSidebar] *,[data-testid=stWidgetLabel] *,.stMarkdown,.stMarkdown p,label,p,li,span{color:#1e293b}
div.stButton>button *{color:#fff!important}
.stAlert *{color:inherit}
</style>""", unsafe_allow_html=True)
 
# ---------------- Database & auth ----------------
def db():
    c = sqlite3.connect(DB)
    c.execute("CREATE TABLE IF NOT EXISTS users(username TEXT PRIMARY KEY,salt TEXT,pw TEXT,role TEXT)")
    c.execute("CREATE TABLE IF NOT EXISTS reviews(id INTEGER PRIMARY KEY,product TEXT,user TEXT,rating INT,review TEXT,date TEXT)")
    return c
 
def hpw(p, salt): return hashlib.pbkdf2_hmac("sha256", p.encode(), salt.encode(), 100000).hex()
 
def add_user(u, p, role="user"):
    salt = os.urandom(8).hex()
    try:
        with db() as c: c.execute("INSERT INTO users VALUES(?,?,?,?)", (u, salt, hpw(p, salt), role))
        return True
    except sqlite3.IntegrityError:
        return False
 
def check_user(u, p):
    r = db().execute("SELECT salt,pw,role FROM users WHERE username=?", (u,)).fetchone()
    return r[2] if r and hpw(p, r[0]) == r[1] else None
 
if not db().execute("SELECT 1 FROM users WHERE username='admin'").fetchone():
    add_user("admin", "Admin@123", "admin")
 
def login_page():
    st.markdown("<h1 style='text-align:center;color:#4f46e5'>📊 ReviewIQ</h1><p style='text-align:center'>Turn customer reviews into actionable insights</p>", unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        t1, t2 = st.tabs(["Login", "Register"])
        with t1:
            u = st.text_input("Username", key="lu"); p = st.text_input("Password", type="password", key="lp")
            if st.button("Login", use_container_width=True):
                role = check_user(u, p)
                if role: st.session_state.update(user=u, role=role); st.rerun()
                else: st.error("Invalid credentials")
            st.caption("Demo admin: admin / Admin@123")
        with t2:
            u = st.text_input("New username", key="ru"); p = st.text_input("New password (min 6)", type="password", key="rp")
            if st.button("Create account", use_container_width=True):
                if len(u) < 3 or len(p) < 6: st.warning("Username ≥3 and password ≥6 characters")
                elif add_user(u, p): st.success("Account created - please log in")
                else: st.error("Username taken")
 
if "user" not in st.session_state:
    login_page(); st.stop()
 
# ---------------- Products & demo data ----------------
PRODUCTS = {
    "Aero Wireless Headphones": (2999, "#6366f1", "#06b6d4"),
    "PixelPro Smartphone": (18999, "#ec4899", "#8b5cf6"),
    "SmartFit Watch": (4499, "#10b981", "#3b82f6"),
    "BrewMaster Coffee Maker": (3499, "#f59e0b", "#ef4444"),
    "TrailRunner Shoes": (2499, "#14b8a6", "#84cc16"),
    "UltraBook Laptop": (54999, "#0ea5e9", "#6366f1"),
}
 
@st.cache_data
def product_image(name):
    for ext in ("jpg", "jpeg", "png", "webp"):
        p = os.path.join(os.path.dirname(__file__), "images", f"{name}.{ext}")
        if os.path.exists(p):
            return Image.open(p).convert("RGB").resize((400, 260))
    c1, c2 = PRODUCTS[name][1:]
    w, h = 400, 260
    a, b = [tuple(int(c[i:i + 2], 16) for i in (1, 3, 5)) for c in (c1, c2)]
    img = Image.new("RGB", (w, h)); d = ImageDraw.Draw(img)
    for y in range(h):
        t = y / h; d.line([(0, y), (w, y)], fill=tuple(int(a[i] * (1 - t) + b[i] * t) for i in range(3)))
    d.rounded_rectangle([130, 50, 270, 170], 24, outline="white", width=6)
    d.ellipse([185, 90, 215, 130], fill="white")
    d.text((20, 215), name, fill="white")
    return img
 
 
 
def load_data():
    base = st.session_state.get("uploaded", make_demo())
    new = pd.read_sql("SELECT product,user,rating,review,date FROM reviews", db())
    df = pd.concat([base, new], ignore_index=True)
    df["sentiment"] = pd.cut(df.rating, [0, 2, 3, 5], labels=LABELS)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df
 
 
# ---------------- ML ----------------
 
 
 
 
 
# ---------------- App ----------------
df = load_data()
vec, models, results, cms = train(len(df), df)
best = results.Model[0]
 
st.sidebar.title("📊 ReviewIQ")
st.sidebar.caption(f"👤 {st.session_state.user} ({st.session_state.role})")
pages = ["Dashboard", "Products", "Live Analyzer", "Suspicious Reviews", "Model Lab"]
if st.session_state.role == "admin": pages.append("Data Upload")
page = st.sidebar.radio("Navigate", pages)
if st.sidebar.button("Logout"): st.session_state.clear(); st.rerun()
st.sidebar.success(f"Best model: {best} (F1 {results.F1[0]:.3f})")
 
if page == "Dashboard":
    st.title("Customer Insights Dashboard")
    f1, f2, f3 = st.columns(3)
    prods = f1.multiselect("Product", list(PRODUCTS), default=list(PRODUCTS))
    sents = f2.multiselect("Sentiment", LABELS, default=LABELS)
    rr = f3.slider("Rating", 1, 5, (1, 5))
    d = df[df["product"].isin(prods) & df.sentiment.isin(sents) & df.rating.between(*rr)]
    k = st.columns(4)
    for col, (lab, v) in zip(k, [("Reviews", len(d)), ("Avg rating", f"{d.rating.mean():.2f}"),
                                  ("% Positive", f"{(d.sentiment == 'Positive').mean() * 100:.0f}%"),
                                  ("Suspicious", len(suspicious(d)))]):
        col.markdown(f"<div class='card'>{lab}<div class='kpi'>{v}</div></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.pie(d, names="sentiment", color="sentiment", color_discrete_map=COLORS, hole=.5, title="Sentiment split"), use_container_width=True)
    c2.plotly_chart(px.histogram(d, x="rating", color="sentiment", color_discrete_map=COLORS, title="Rating distribution"), use_container_width=True)
    sample = d.sample(min(len(d), 300), random_state=1) if len(d) else d
    rows = [(a, s) for t in sample.review for a, s in aspect_sentiments(t, vec, models, best).items()]
    if rows:
        ad = pd.DataFrame(rows, columns=["Aspect", "Sentiment"]).value_counts().reset_index(name="n")
        st.plotly_chart(px.bar(ad, x="Aspect", y="n", color="Sentiment", color_discrete_map=COLORS, barmode="group", title="Aspect-level sentiment"), use_container_width=True)
        neg = ad[ad.Sentiment == "Negative"].sort_values("n", ascending=False)
        pos = ad[ad.Sentiment == "Positive"].sort_values("n", ascending=False)
        st.subheader("💡 Insights")
        if len(neg): st.warning(f"Top complaint area: **{neg.Aspect.iloc[0]}** ({neg.n.iloc[0]} negative mentions) - prioritise improvement.")
        if len(pos): st.success(f"Biggest strength: **{pos.Aspect.iloc[0]}** ({pos.n.iloc[0]} positive mentions) - highlight in marketing.")
    tr = d.dropna(subset=["date"]).set_index("date").resample("MS").rating.mean().reset_index()
    st.plotly_chart(px.line(tr, x="date", y="rating", markers=True, title="Average rating trend"), use_container_width=True)
 
elif page == "Products":
    st.title("Product Catalog")
    cols = st.columns(3)
    for i, (n, (price, *_)) in enumerate(PRODUCTS.items()):
        with cols[i % 3]:
            sub = df[df["product"] == n]
            st.image(product_image(n), use_container_width=True)
            st.markdown(f"**{n}**  \n₹{price:,} · ⭐ {sub.rating.mean():.1f} ({len(sub)})")
            if st.button("View reviews", key=n): st.session_state.sel = n
    n = st.session_state.get("sel")
    if n:
        st.divider(); st.header(n)
        sub = df[df["product"] == n]
        st.info("**Summary (positive):** " + summarize(sub[sub.sentiment == "Positive"].review))
        st.error("**Summary (negative):** " + summarize(sub[sub.sentiment == "Negative"].review))
        st.dataframe(sub[["user", "rating", "review", "date"]].tail(15), use_container_width=True)
        with st.form("addrev"):
            rt = st.slider("Your rating", 1, 5, 4); tx = st.text_area("Your review")
            if st.form_submit_button("Submit review") and tx.strip():
                with db() as c:
                    c.execute("INSERT INTO reviews(product,user,rating,review,date) VALUES(?,?,?,?,?)",
                              (n, st.session_state.user, rt, tx, dt.date.today().isoformat()))
                st.success("Review saved!"); st.rerun()
 
elif page == "Live Analyzer":
    st.title("Real-time Review Analyzer")
    txt = st.text_area("Paste a review", "The quality is great but delivery was very late and the box was damaged!")
    if st.button("Analyze") and txt.strip():
        s = predict(vec, models, best, [txt])[0]
        st.markdown(f"### Sentiment: <span style='color:{COLORS[s]}'>{s}</span>", unsafe_allow_html=True)
        asp = aspect_sentiments(txt, vec, models, best)
        if asp: st.table(pd.DataFrame(asp.items(), columns=["Aspect", "Sentiment"]))
        sus = suspicious(pd.DataFrame({"review": [txt], "user": ["x"], "rating": [3]}).assign(risk=0))
        st.write("🚩 Looks suspicious" if len(sus) else "✅ No suspicious patterns found")
        st.caption("Limitation: TF-IDF models can miss sarcasm and mixed opinions.")
 
elif page == "Suspicious Reviews":
    st.title("Potentially Suspicious Reviews")
    sus = suspicious(df)
    st.write(f"{len(sus)} flagged (rule-based heuristics - indicators, not proof).")
    st.dataframe(sus[["product", "user", "rating", "review", "risk", "reasons"]], use_container_width=True)
    st.download_button("Download CSV", sus.to_csv(index=False), "suspicious_reviews.csv")
 
elif page == "Model Lab":
    st.title("Model Comparison")
    st.dataframe(results.style.format({c: "{:.3f}" for c in results.columns[1:]}).highlight_max(subset=["F1"], color="#4338ca"), use_container_width=True)
    st.plotly_chart(px.bar(results.melt("Model"), x="Model", y="value", color="variable", barmode="group"), use_container_width=True)
    m = st.selectbox("Confusion matrix", list(cms), index=list(cms).index(best))
    st.plotly_chart(px.imshow(cms[m], x=LABELS, y=LABELS, text_auto=True, color_continuous_scale="Purples", labels=dict(x="Predicted", y="Actual")), use_container_width=True)
    st.caption("Demo data is synthetic, so scores are optimistic; use a real dataset (Data Upload) for honest results.")
 
elif page == "Data Upload":
    st.title("Upload Real Dataset")
    st.write("CSV columns required: `review`, `rating` (optional: `product`, `user`, `date`).")
    f = st.file_uploader("CSV", type="csv")
    if f:
        u = pd.read_csv(f)
        u["product"] = u.get("product", pd.Series(random.choices(list(PRODUCTS), k=len(u))))
        u["user"] = u.get("user", "anon"); u["date"] = u.get("date", dt.date.today().isoformat())
        st.session_state.uploaded = u[["product", "user", "rating", "review", "date"]]
        st.success(f"Loaded {len(u)} reviews - models will retrain.")
 