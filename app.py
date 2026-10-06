"""ReviewIQ - integrated e-commerce review analysis system (Streamlit)."""

import re
import random
import sqlite3
import hashlib
import os
import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from PIL import Image, ImageDraw
# ---------------------------------------------------------
# Project modules
# ---------------------------------------------------------

from src.nlp import clean, suspicious, summarize
from src.ml import train as train_models, predict, aspect_sentiments
from src.data import make_demo, ASPECTS


# ---------------------------------------------------------
# Streamlit configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="ReviewIQ",
    page_icon="📊",
    layout="wide"
)

px.defaults.template = "plotly_white"


# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------

LABELS = ["Negative", "Neutral", "Positive"]

COLORS = {
    "Positive": "#22c55e",
    "Neutral": "#f59e0b",
    "Negative": "#ef4444"
}

DB = os.path.join(
    os.path.dirname(__file__),
    "reviewiq.db"
)


# ---------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------

st.markdown(
    """
    <style>

    .stApp {
        background: linear-gradient(
            135deg,
            #f8fafc 0%,
            #eef2ff 100%
        );
        color: #1e293b;
    }

    [data-testid=stSidebar] {
        background: #ffffff;
        border-right: 1px solid #e2e8f0;
    }

    h1, h2, h3 {
        color: #4f46e5;
    }

    .card {
        background: #ffffff;
        border-radius: 14px;
        padding: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(15, 23, 42, .06);
        margin-bottom: 10px;
        color: #1e293b;
    }

    .kpi {
        font-size: 2rem;
        font-weight: 700;
        color: #4f46e5;
    }

    div.stButton > button {
        background: linear-gradient(
            90deg,
            #6366f1,
            #8b5cf6
        );
        color: #fff;
        border: 0;
        border-radius: 8px;
    }

    [data-testid=stSidebar] *,
    [data-testid=stWidgetLabel] *,
    .stMarkdown,
    .stMarkdown p,
    label,
    p,
    li,
    span {
        color: #1e293b;
    }

    div.stButton > button * {
        color: #fff !important;
    }

    .stAlert * {
        color: inherit;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# Database & Authentication
# ---------------------------------------------------------

def db():
    c = sqlite3.connect(DB)

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS users(
            username TEXT PRIMARY KEY,
            salt TEXT,
            pw TEXT,
            role TEXT
        )
        """
    )

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS reviews(
            id INTEGER PRIMARY KEY,
            product TEXT,
            user TEXT,
            rating INT,
            review TEXT,
            date TEXT
        )
        """
    )

    c.commit()

    return c


def hpw(p, salt):
    return hashlib.pbkdf2_hmac(
        "sha256",
        p.encode(),
        salt.encode(),
        100000
    ).hex()


def add_user(u, p, role="user"):

    salt = os.urandom(8).hex()

    try:
        with db() as c:
            c.execute(
                "INSERT INTO users VALUES(?,?,?,?)",
                (
                    u,
                    salt,
                    hpw(p, salt),
                    role
                )
            )

        return True

    except sqlite3.IntegrityError:
        return False


def check_user(u, p):

    r = db().execute(
        "SELECT salt,pw,role FROM users WHERE username=?",
        (u,)
    ).fetchone()

    return (
        r[2]
        if r and hpw(p, r[0]) == r[1]
        else None
    )


# Create default admin account
if not db().execute(
    "SELECT 1 FROM users WHERE username='admin'"
).fetchone():

    add_user(
        "admin",
        "Admin@123",
        "admin"
    )


# ---------------------------------------------------------
# Login Page
# ---------------------------------------------------------

def login_page():

    st.markdown(
        """
        <h1 style='text-align:center;color:#4f46e5'>
        📊 ReviewIQ
        </h1>

        <p style='text-align:center'>
        Turn customer reviews into actionable insights
        </p>
        """,
        unsafe_allow_html=True
    )

    _, mid, _ = st.columns([1, 1.2, 1])

    with mid:

        t1, t2 = st.tabs(
            ["Login", "Register"]
        )

        # -------------------------
        # Login
        # -------------------------

        with t1:

            u = st.text_input(
                "Username",
                key="lu"
            )

            p = st.text_input(
                "Password",
                type="password",
                key="lp"
            )

            if st.button(
                "Login",
                use_container_width=True
            ):

                role = check_user(u, p)

                if role:

                    st.session_state.update(
                        user=u,
                        role=role
                    )

                    st.rerun()

                else:

                    st.error(
                        "Invalid credentials"
                    )

            st.caption(
                "Demo admin: admin / Admin@123"
            )

        # -------------------------
        # Register
        # -------------------------

        with t2:

            u = st.text_input(
                "New username",
                key="ru"
            )

            p = st.text_input(
                "New password (min 6)",
                type="password",
                key="rp"
            )

            if st.button(
                "Create account",
                use_container_width=True
            ):

                if len(u) < 3 or len(p) < 6:

                    st.warning(
                        "Username ≥3 and password ≥6 characters"
                    )

                elif add_user(u, p):

                    st.success(
                        "Account created - please log in"
                    )

                else:

                    st.error(
                        "Username taken"
                    )


# ---------------------------------------------------------
# Require Login
# ---------------------------------------------------------

if "user" not in st.session_state:

    login_page()

    st.stop()


# ---------------------------------------------------------
# Products
# ---------------------------------------------------------

PRODUCTS = {

    "Aero Wireless Headphones":
        (2999, "#6366f1", "#06b6d4"),

    "PixelPro Smartphone":
        (18999, "#ec4899", "#8b5cf6"),

    "SmartFit Watch":
        (4499, "#10b981", "#3b82f6"),

    "BrewMaster Coffee Maker":
        (3499, "#f59e0b", "#ef4444"),

    "TrailRunner Shoes":
        (2499, "#14b8a6", "#84cc16"),

    "UltraBook Laptop":
        (54999, "#0ea5e9", "#6366f1"),
}


# ---------------------------------------------------------
# Product Images
# ---------------------------------------------------------

@st.cache_data
def product_image(name):

    for ext in (
        "jpg",
        "jpeg",
        "png",
        "webp"
    ):

        p = os.path.join(
            os.path.dirname(__file__),
            "images",
            f"{name}.{ext}"
        )

        if os.path.exists(p):

            return Image.open(
                p
            ).convert("RGB").resize(
                (400, 260)
            )

    c1, c2 = PRODUCTS[name][1:]

    w, h = 400, 260

    a, b = [
        tuple(
            int(c[i:i + 2], 16)
            for i in (1, 3, 5)
        )
        for c in (c1, c2)
    ]

    img = Image.new(
        "RGB",
        (w, h)
    )

    d = ImageDraw.Draw(img)

    for y in range(h):

        t = y / h

        d.line(
            [(0, y), (w, y)],
            fill=tuple(
                int(
                    a[i] * (1 - t)
                    + b[i] * t
                )
                for i in range(3)
            )
        )

    d.rounded_rectangle(
        [130, 50, 270, 170],
        24,
        outline="white",
        width=6
    )

    d.ellipse(
        [185, 90, 215, 130],
        fill="white"
    )

    d.text(
        (20, 215),
        name,
        fill="white"
    )

    return img


# ---------------------------------------------------------
# Data Loading
# ---------------------------------------------------------

def load_data():

    base = st.session_state.get(
        "uploaded",
        make_demo()
    )

    new = pd.read_sql(
        """
        SELECT product,user,rating,review,date
        FROM reviews
        """,
        db()
    )

    df = pd.concat(
        [base, new],
        ignore_index=True
    )

    df["sentiment"] = pd.cut(
        df.rating,
        [0, 2, 3, 5],
        labels=LABELS
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    return df


# ---------------------------------------------------------
# Load Data
# ---------------------------------------------------------

df = load_data()


# ---------------------------------------------------------
# Train Models
# ---------------------------------------------------------

vec, models, results, cms = train_models(
    len(df),
    df
)

best = results.Model.iloc[0]


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

st.sidebar.title(
    "📊 ReviewIQ"
)

st.sidebar.caption(
    f"👤 {st.session_state.user} "
    f"({st.session_state.role})"
)

pages = [
    "Dashboard",
    "Products",
    "Live Analyzer",
    "Suspicious Reviews",
    "Model Lab"
]

if st.session_state.role == "admin":

    pages.append(
        "Data Upload"
    )

page = st.sidebar.radio(
    "Navigate",
    pages
)

if st.sidebar.button(
    "Logout"
):

    st.session_state.clear()

    st.rerun()


st.sidebar.success(
    f"Best model: {best} "
    f"(F1 {results.F1.iloc[0]:.3f})"
)


# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    st.title(
        "Customer Insights Dashboard"
    )

    f1, f2, f3 = st.columns(3)

    prods = f1.multiselect(
        "Product",
        list(PRODUCTS),
        default=list(PRODUCTS)
    )

    sents = f2.multiselect(
        "Sentiment",
        LABELS,
        default=LABELS
    )

    rr = f3.slider(
        "Rating",
        1,
        5,
        (1, 5)
    )

    d = df[
        df["product"].isin(prods)
        & df.sentiment.isin(sents)
        & df.rating.between(*rr)
    ]

    k = st.columns(4)

    metrics = [
        ("Reviews", len(d)),
        (
            "Avg rating",
            f"{d.rating.mean():.2f}"
            if len(d)
            else "0.00"
        ),
        (
            "% Positive",
            f"{(d.sentiment == 'Positive').mean() * 100:.0f}%"
            if len(d)
            else "0%"
        ),
        (
            "Suspicious",
            len(suspicious(d))
        )
    ]

    for col, (lab, v) in zip(
        k,
        metrics
    ):

        col.markdown(
            f"""
            <div class='card'>
                {lab}
                <div class='kpi'>
                    {v}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    c1, c2 = st.columns(2)

    # Sentiment pie chart

    if len(d):

        c1.plotly_chart(
            px.pie(
                d,
                names="sentiment",
                color="sentiment",
                color_discrete_map=COLORS,
                hole=.5,
                title="Sentiment split"
            ),
            use_container_width=True
        )

        c2.plotly_chart(
            px.histogram(
                d,
                x="rating",
                color="sentiment",
                color_discrete_map=COLORS,
                title="Rating distribution"
            ),
            use_container_width=True
        )

    # Aspect analysis

    sample = (
        d.sample(
            min(len(d), 300),
            random_state=1
        )
        if len(d)
        else d
    )

    rows = [
        (a, s)
        for t in sample.review
        for a, s in aspect_sentiments(
            t,
            vec,
            models,
            best
        ).items()
    ]

    if rows:

        ad = (
            pd.DataFrame(
                rows,
                columns=[
                    "Aspect",
                    "Sentiment"
                ]
            )
            .value_counts()
            .reset_index(
                name="n"
            )
        )

        st.plotly_chart(
            px.bar(
                ad,
                x="Aspect",
                y="n",
                color="Sentiment",
                color_discrete_map=COLORS,
                barmode="group",
                title="Aspect-level sentiment"
            ),
            use_container_width=True
        )

        neg = (
            ad[
                ad.Sentiment == "Negative"
            ]
            .sort_values(
                "n",
                ascending=False
            )
        )

        pos = (
            ad[
                ad.Sentiment == "Positive"
            ]
            .sort_values(
                "n",
                ascending=False
            )
        )

        st.subheader(
            "💡 Insights"
        )

        if len(neg):

            st.warning(
                f"Top complaint area: "
                f"**{neg.Aspect.iloc[0]}** "
                f"({neg.n.iloc[0]} negative mentions) "
                f"- prioritise improvement."
            )

        if len(pos):

            st.success(
                f"Biggest strength: "
                f"**{pos.Aspect.iloc[0]}** "
                f"({pos.n.iloc[0]} positive mentions) "
                f"- highlight in marketing."
            )

    # Rating trend

    tr = (
        d.dropna(
            subset=["date"]
        )
        .set_index("date")
        .resample("MS")
        .rating.mean()
        .reset_index()
    )

    if len(tr):

        st.plotly_chart(
            px.line(
                tr,
                x="date",
                y="rating",
                markers=True,
                title="Average rating trend"
            ),
            use_container_width=True
        )


# =========================================================
# PRODUCTS
# =========================================================

elif page == "Products":

    st.title(
        "Product Catalog"
    )

    cols = st.columns(3)

    for i, (n, (price, *_)) in enumerate(
        PRODUCTS.items()
    ):

        with cols[i % 3]:

            sub = df[
                df["product"] == n
            ]

            st.image(
                product_image(n),
                use_container_width=True
            )

            avg_rating = (
                sub.rating.mean()
                if len(sub)
                else 0
            )

            st.markdown(
                f"""
                **{n}**

                ₹{price:,} · ⭐
                {avg_rating:.1f}
                ({len(sub)})
                """
            )

            if st.button(
                "View reviews",
                key=n
            ):

                st.session_state.sel = n

    n = st.session_state.get(
        "sel"
    )

    if n:

        st.divider()

        st.header(n)

        sub = df[
            df["product"] == n
        ]

        positive_reviews = sub[
            sub.sentiment == "Positive"
        ].review

        negative_reviews = sub[
            sub.sentiment == "Negative"
        ].review

        st.info(
            "**Summary (positive):** "
            + summarize(positive_reviews)
        )

        st.error(
            "**Summary (negative):** "
            + summarize(negative_reviews)
        )

        st.dataframe(
            sub[
                [
                    "user",
                    "rating",
                    "review",
                    "date"
                ]
            ].tail(15),
            use_container_width=True
        )

        with st.form("addrev"):

            rt = st.slider(
                "Your rating",
                1,
                5,
                4
            )

            tx = st.text_area(
                "Your review"
            )

            submitted = st.form_submit_button(
                "Submit review"
            )

            if submitted and tx.strip():

                with db() as c:

                    c.execute(
                        """
                        INSERT INTO reviews
                        (product,user,rating,review,date)
                        VALUES(?,?,?,?,?)
                        """,
                        (
                            n,
                            st.session_state.user,
                            rt,
                            tx,
                            dt.date.today().isoformat()
                        )
                    )

                st.success(
                    "Review saved!"
                )

                st.rerun()


# =========================================================
# LIVE ANALYZER
# =========================================================

elif page == "Live Analyzer":

    st.title(
        "Real-time Review Analyzer"
    )

    txt = st.text_area(
        "Paste a review",
        "The quality is great but delivery was very late and the box was damaged!"
    )

    if st.button(
        "Analyze"
    ) and txt.strip():

        s = predict(
            vec,
            models,
            best,
            [txt]
        )[0]

        st.markdown(
            f"""
            ### Sentiment:
            <span style='color:{COLORS[s]}'>
            {s}
            </span>
            """,
            unsafe_allow_html=True
        )

        asp = aspect_sentiments(
            txt,
            vec,
            models,
            best
        )

        if asp:

            st.table(
                pd.DataFrame(
                    asp.items(),
                    columns=[
                        "Aspect",
                        "Sentiment"
                    ]
                )
            )

        sus = suspicious(
            pd.DataFrame(
                {
                    "review": [txt],
                    "user": ["x"],
                    "rating": [3]
                }
            ).assign(
                risk=0
            )
        )

        st.write(
            "🚩 Looks suspicious"
            if len(sus)
            else
            "✅ No suspicious patterns found"
        )

        st.caption(
            "Limitation: TF-IDF models can miss "
            "sarcasm and mixed opinions."
        )


# =========================================================
# SUSPICIOUS REVIEWS
# =========================================================

elif page == "Suspicious Reviews":

    st.title(
        "Potentially Suspicious Reviews"
    )

    sus = suspicious(df)

    st.write(
        f"{len(sus)} flagged "
        "(rule-based heuristics - indicators, not proof)."
    )

    st.dataframe(
        sus[
            [
                "product",
                "user",
                "rating",
                "review",
                "risk",
                "reasons"
            ]
        ],
        use_container_width=True
    )

    st.download_button(
        "Download CSV",
        sus.to_csv(index=False),
        "suspicious_reviews.csv"
    )


# =========================================================
# MODEL LAB
# =========================================================

elif page == "Model Lab":

    st.title(
        "Model Comparison"
    )

    st.dataframe(
        results.style.format(
            {
                c: "{:.3f}"
                for c in results.columns[1:]
            }
        ).highlight_max(
            subset=["F1"],
            color="#4338ca"
        ),
        use_container_width=True
    )

    st.plotly_chart(
        px.bar(
            results.melt("Model"),
            x="Model",
            y="value",
            color="variable",
            barmode="group"
        ),
        use_container_width=True
    )

    m = st.selectbox(
        "Confusion matrix",
        list(cms),
        index=list(cms).index(best)
    )

    st.plotly_chart(
        px.imshow(
            cms[m],
            x=LABELS,
            y=LABELS,
            text_auto=True,
            color_continuous_scale="Purples",
            labels={
                "x": "Predicted",
                "y": "Actual"
            }
        ),
        use_container_width=True
    )

    st.caption(
        "Demo data is synthetic, so scores are optimistic; "
        "use a real dataset (Data Upload) for honest results."
    )


# =========================================================
# DATA UPLOAD
# =========================================================

elif page == "Data Upload":

    st.title(
        "Upload Real Dataset"
    )

    st.write(
        "CSV columns required: "
        "`review`, `rating` "
        "(optional: `product`, `user`, `date`)."
    )

    f = st.file_uploader(
        "CSV",
        type="csv"
    )

    if f:

        u = pd.read_csv(f)

        if "review" not in u.columns:
            st.error(
                "CSV must contain a 'review' column."
            )
            st.stop()

        if "rating" not in u.columns:
            st.error(
                "CSV must contain a 'rating' column."
            )
            st.stop()

        if "product" not in u.columns:

            u["product"] = random.choices(
                list(PRODUCTS),
                k=len(u)
            )

        if "user" not in u.columns:

            u["user"] = "anon"

        if "date" not in u.columns:

            u["date"] = dt.date.today().isoformat()

        st.session_state.uploaded = u[
            [
                "product",
                "user",
                "rating",
                "review",
                "date"
            ]
        ]

        st.success(
            f"Loaded {len(u)} reviews - models will retrain."
        )