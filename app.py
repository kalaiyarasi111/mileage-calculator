"""
MPG Studio - Predicting Automobile Fuel Efficiency (MPG) with Machine Learning
Run with:  streamlit run app.py
"""

import io
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="MPG Studio",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

NUMERIC = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model year"]
CATEGORICAL = ["origin"]
REQUIRED = ["mpg"] + NUMERIC + CATEGORICAL
ORIGIN_LABELS = {1: "USA", 2: "Europe", 3: "Japan"}

PALETTE = {
    "night": "#0E0B24",
    "magenta": "#FF3D81",
    "amber": "#FFB020",
    "teal": "#19E3C2",
    "violet": "#7B5CFF",
}
MODEL_COLORS = ["#FF3D81", "#FFB020", "#19E3C2", "#7B5CFF", "#4DA3FF"]

# ----------------------------------------------------------------------------
# Styling: animated night-sky gradient, glass panels, neon accents
# ----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Unbounded:wght@500;700;900&family=Sora:wght@300;400;600&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Sora', sans-serif; }

.stApp {
    background: linear-gradient(125deg, #0E0B24, #1B0F3F, #2A0F4D, #0B2A3F, #0E0B24);
    background-size: 400% 400%;
    animation: drift 24s ease infinite;
    color: #F4F1FF;
}
@keyframes drift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] {
    background: rgba(14, 11, 36, 0.78);
    backdrop-filter: blur(14px);
    border-right: 1px solid rgba(123, 92, 255, 0.35);
}

/* Hero */
.hero {
    position: relative;
    overflow: hidden;
    padding: 34px 38px 30px 38px;
    border-radius: 28px;
    background: linear-gradient(120deg, rgba(255,61,129,.22), rgba(123,92,255,.22) 55%, rgba(25,227,194,.18));
    border: 1px solid rgba(255,255,255,.14);
    margin-bottom: 18px;
}
.hero h1 {
    font-family: 'Unbounded', sans-serif;
    font-weight: 900;
    font-size: clamp(2rem, 4.6vw, 3.6rem);
    line-height: 1.05;
    margin: 0 0 8px 0;
    background: linear-gradient(90deg, #FF3D81, #FFB020, #19E3C2, #7B5CFF, #FF3D81);
    background-size: 300% 100%;
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    animation: shine 7s linear infinite;
}
@keyframes shine { to { background-position: 300% 0; } }
.hero p { font-weight: 300; font-size: 1.05rem; margin: 0; max-width: 640px; color: #D9D3F5; }

/* The road: a car drives across the hero once the page loads */
.road {
    position: absolute; left: 0; right: 0; bottom: 0; height: 4px;
    background: repeating-linear-gradient(90deg, rgba(255,255,255,.55) 0 26px, transparent 26px 52px);
    background-size: 52px 4px;
    animation: dashes 1.1s linear infinite;
}
@keyframes dashes { to { background-position: -52px 0; } }
.car {
    position: absolute; bottom: 6px; font-size: 2.6rem;
    animation: cruise 9s cubic-bezier(.4,.1,.3,1) infinite;
    filter: drop-shadow(0 0 12px rgba(255,61,129,.7));
}
@keyframes cruise {
    0% { left: -8%; transform: scaleX(-1); }
    48% { left: 92%; transform: scaleX(-1); }
    50% { left: 92%; transform: scaleX(1); }
    98% { left: -8%; transform: scaleX(1); }
    100% { left: -8%; transform: scaleX(-1); }
}

/* Glass panels */
.glass {
    padding: 22px 24px;
    border-radius: 22px;
    background: rgba(255,255,255,.06);
    border: 1px solid rgba(255,255,255,.12);
    backdrop-filter: blur(10px);
}
.panel-title {
    font-family: 'Unbounded', sans-serif;
    font-weight: 700; font-size: 1.05rem; margin-bottom: 10px;
}

/* Stat tiles */
.tile {
    padding: 16px 18px; border-radius: 18px;
    background: rgba(255,255,255,.07);
    border: 1px solid rgba(255,255,255,.12);
    transition: transform .25s ease, border-color .25s ease;
}
.tile:hover { transform: translateY(-4px) rotate(-.6deg); border-color: var(--c); }
.tile .v { font-family: 'Unbounded', sans-serif; font-size: 1.7rem; font-weight: 700; color: var(--c); }
.tile .l { font-size: .85rem; color: #BFB7E6; }

/* Tabs */
button[data-baseweb="tab"] {
    font-family: 'Unbounded', sans-serif; font-weight: 500; font-size: .85rem;
    border-radius: 14px 14px 0 0; padding: 10px 18px;
}
button[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(120deg, rgba(255,61,129,.35), rgba(123,92,255,.35));
}

/* Buttons */
.stButton > button {
    font-family: 'Unbounded', sans-serif; font-weight: 700;
    border: none; border-radius: 16px; padding: .8rem 1.4rem; color: #14102E;
    background: linear-gradient(120deg, #FFB020, #FF3D81, #7B5CFF);
    background-size: 220% 100%;
    transition: background-position .5s ease, transform .2s ease, box-shadow .2s ease;
}
.stButton > button:hover {
    background-position: 100% 0; transform: scale(1.03);
    box-shadow: 0 0 28px rgba(255,61,129,.55); color: #14102E;
}
.stButton > button:active { transform: scale(.97); }

/* Verdict badge that pops in after a prediction */
.verdict {
    display: inline-block; padding: 8px 18px; border-radius: 999px;
    font-family: 'Unbounded', sans-serif; font-weight: 700; font-size: .95rem;
    animation: pop .6s cubic-bezier(.2,1.6,.4,1) both;
}
@keyframes pop { from { transform: scale(.2); opacity: 0; } to { transform: scale(1); opacity: 1; } }

/* Floating sparks */
.spark { position: fixed; z-index: 0; pointer-events: none; font-size: 1.4rem; opacity: .25;
         animation: floaty 14s ease-in-out infinite; }
@keyframes floaty { 0%,100% { transform: translateY(0) rotate(0); } 50% { transform: translateY(-40px) rotate(14deg); } }

@media (prefers-reduced-motion: reduce) {
    .stApp, .hero h1, .road, .car, .spark, .verdict { animation: none !important; }
}
</style>
<div class="spark" style="top:18%;left:3%;">⛽</div>
<div class="spark" style="top:62%;right:3%;animation-delay:-4s;">🛞</div>
<div class="spark" style="top:84%;left:46%;animation-delay:-8s;">⚙️</div>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Data handling
# ----------------------------------------------------------------------------
def make_demo_data(n: int = 398, seed: int = 7) -> pd.DataFrame:
    """Synthetic stand-in used only when no dataset is available."""
    rng = np.random.default_rng(seed)
    cyl = rng.choice([4, 6, 8, 3, 5], size=n, p=[0.52, 0.22, 0.22, 0.02, 0.02])
    disp = cyl * rng.normal(48, 8, n).clip(30, 70)
    hp = (disp * 0.45 + rng.normal(25, 12, n)).clip(46, 230)
    weight = (disp * 5.5 + 1500 + rng.normal(0, 220, n)).clip(1600, 5100)
    acc = (22 - hp / 14 + rng.normal(0, 1.4, n)).clip(8, 25)
    year = rng.integers(70, 83, n)
    origin = rng.choice([1, 2, 3], size=n, p=[0.62, 0.17, 0.21])
    mpg = (47 - weight / 190 - hp / 40 + (year - 70) * 0.5 + (origin > 1) * 2.2
           + rng.normal(0, 1.8, n)).clip(9, 47)
    df = pd.DataFrame({
        "mpg": mpg.round(1), "cylinders": cyl, "displacement": disp.round(0),
        "horsepower": hp.round(0), "weight": weight.round(0),
        "acceleration": acc.round(1), "model year": year, "origin": origin,
        "car name": [f"demo car {i}" for i in range(n)],
    })
    miss = rng.choice(n, size=6, replace=False)
    df.loc[miss, "horsepower"] = np.nan
    return df


def clean_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names/types. Missing values are left for the pipeline."""
    df = raw.copy()
    df.columns = [str(c).strip().lower().replace("_", " ") for c in df.columns]
    missing_cols = [c for c in REQUIRED if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required column(s): {', '.join(missing_cols)}")

    df = df.replace("?", np.nan)
    for col in ["mpg"] + NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["mpg"]).reset_index(drop=True)

    # origin -> readable text labels (works for 1/2/3 codes or text)
    num_origin = pd.to_numeric(df["origin"], errors="coerce")
    if num_origin.notna().mean() > 0.9:
        df["origin"] = num_origin.round().map(ORIGIN_LABELS)
    else:
        df["origin"] = df["origin"].astype(str).str.strip().str.title().replace(
            {"Usa": "USA", "Nan": np.nan}
        )

    # model year given as 70-82 or 1970-1982: keep as in dataset, remember style
    if "car name" not in df.columns:
        df["car name"] = "unknown"
    return df


@st.cache_data(show_spinner=False)
def read_csv_bytes(content: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(content))


def full_year(y):
    """70 -> 1970 for display only."""
    return int(y) + 1900 if y < 100 else int(y)


# ----------------------------------------------------------------------------
# ML: preprocessing pipeline + model arena
# ----------------------------------------------------------------------------
def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC),
        ("cat", categorical_pipe, CATEGORICAL),
    ])


def model_zoo():
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=250, learning_rate=0.06,
                                                      max_depth=3, random_state=42),
        "Support Vector (SVR)": SVR(C=15, epsilon=0.3),
    }


@st.cache_resource(show_spinner=False)
def train_arena(df: pd.DataFrame):
    X = df[NUMERIC + CATEGORICAL]
    y = df["mpg"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    rows, fitted = [], {}
    for name, est in model_zoo().items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", est)])
        cv = cross_val_score(pipe, X_tr, y_tr, cv=5, scoring="r2")
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        rows.append({
            "Model": name,
            "R² (test)": r2_score(y_te, pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_te, pred))),
            "MAE": mean_absolute_error(y_te, pred),
            "CV R² (5-fold)": cv.mean(),
        })
        fitted[name] = pipe

    results = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    best_name = results.loc[0, "Model"]

    # Importance for the champion (measured on held-out data)
    imp = permutation_importance(fitted[best_name], X_te, y_te, n_repeats=8,
                                 random_state=42, scoring="r2")
    importance = (pd.DataFrame({"Feature": X.columns, "Importance": imp.importances_mean})
                  .sort_values("Importance"))

    # Deploy: refit every pipeline on ALL rows so predictions use the full data
    final = {}
    for name, est in model_zoo().items():
        p = Pipeline([("prep", build_preprocessor()), ("model", est)])
        p.fit(X, y)
        final[name] = p

    holdout = pd.DataFrame({"Actual": y_te.values, "Predicted": fitted[best_name].predict(X_te)})
    return results, best_name, final, importance, holdout


# ----------------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------------
def style_fig(fig, height=420):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        font=dict(family="Sora", color="#EDE9FF"),
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,.08)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,.08)", zeroline=False)
    return fig


def gauge(value: float, avg: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        number={"suffix": " MPG", "font": {"size": 44, "family": "Unbounded"}},
        delta={"reference": avg, "increasing": {"color": PALETTE["teal"]},
               "decreasing": {"color": PALETTE["magenta"]}},
        gauge={
            "axis": {"range": [0, 50], "tickcolor": "#BFB7E6"},
            "bar": {"color": "#FFFFFF", "thickness": 0.22},
            "bgcolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [0, 15], "color": PALETTE["magenta"]},
                {"range": [15, 22], "color": PALETTE["amber"]},
                {"range": [22, 30], "color": PALETTE["violet"]},
                {"range": [30, 50], "color": PALETTE["teal"]},
            ],
            "threshold": {"line": {"color": "white", "width": 4}, "thickness": 0.9, "value": avg},
        },
    ))
    fig.update_layout(height=330, paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Sora", color="#EDE9FF"),
                      margin=dict(l=20, r=20, t=30, b=0))
    return fig


def verdict_for(mpg: float):
    if mpg < 15:
        return "Gas guzzler", PALETTE["magenta"]
    if mpg < 22:
        return "Everyday cruiser", PALETTE["amber"]
    if mpg < 30:
        return "Smart sipper", PALETTE["violet"]
    return "Eco champion", PALETTE["teal"]


def count_up(value: float, color: str):
    """Number that rolls up from 0 to the prediction."""
    components.html(f"""
    <div style="text-align:center;font-family:'Unbounded',sans-serif;">
      <div id="n" style="font-size:78px;font-weight:900;color:{color};
           text-shadow:0 0 30px {color}88;">0.0</div>
      <div style="color:#BFB7E6;font-family:Sora,sans-serif;font-size:15px;">
        miles per gallon</div>
    </div>
    <script>
      const target = {value:.2f}, el = document.getElementById('n');
      const t0 = performance.now(), dur = 1400;
      function tick(t) {{
        const p = Math.min((t - t0) / dur, 1), e = 1 - Math.pow(1 - p, 3);
        el.textContent = (target * e).toFixed(1);
        if (p < 1) requestAnimationFrame(tick);
      }}
      requestAnimationFrame(tick);
    </script>
    """, height=150)


def tile(value, label, color):
    return (f'<div class="tile" style="--c:{color}"><div class="v">{value}</div>'
            f'<div class="l">{label}</div></div>')


# ----------------------------------------------------------------------------
# Sidebar: dataset upload
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📂 Dataset")
    uploaded = st.file_uploader("Upload auto-mpg.csv", type=["csv"],
                                help="Needs: mpg, cylinders, displacement, horsepower, "
                                     "weight, acceleration, model year, origin")
    st.caption("Missing values (like the `?` in horsepower) are handled automatically.")
    st.divider()
    st.markdown("### 🧭 How it works")
    st.markdown(
        "1. Upload your CSV\n"
        "2. Five models race on your data\n"
        "3. The winner is deployed\n"
        "4. Enter car specs, get an MPG"
    )

source_note = ""
raw_df = None
if uploaded is not None:
    try:
        raw_df = read_csv_bytes(uploaded.getvalue())
        source_note = f"Using your upload: **{uploaded.name}**"
    except Exception as exc:  # noqa: BLE001
        st.sidebar.error(f"Could not read that file: {exc}")

if raw_df is None:
    try:
        raw_df = pd.read_csv("auto-mpg.csv")
        source_note = "Using **auto-mpg.csv** found next to the app."
    except Exception:  # noqa: BLE001
        raw_df = make_demo_data()
        source_note = ("No dataset yet, so a **synthetic demo dataset** is loaded. "
                       "Upload auto-mpg.csv in the sidebar to use the real data.")

try:
    df = clean_dataframe(raw_df)
except ValueError as err:
    st.error(f"{err}. Check the column names and upload again.")
    st.stop()

with st.spinner("The models are racing..."):
    results, best_name, models, importance, holdout = train_arena(df)

# ----------------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <h1>MPG Studio</h1>
      <p>Describe a car, and a model trained on real vehicle specs tells you how far
         it will go on a gallon.</p>
      <div class="car">🏎️</div>
      <div class="road"></div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.info(source_note, icon="📌")

tab_predict, tab_explore, tab_arena, tab_pipe, tab_data = st.tabs(
    ["🎯 Predict", "📊 Explore", "🏁 Model arena", "🧪 Pipeline", "🗂️ Data"]
)

# ----------------------------------------------------------------------------
# Tab: Predict
# ----------------------------------------------------------------------------
with tab_predict:
    left, right = st.columns([1, 1.15], gap="large")

    def rng_of(col, as_int=False):
        lo, hi = float(df[col].min()), float(df[col].max())
        med = float(df[col].median())
        return (int(lo), int(hi), int(med)) if as_int else (lo, hi, med)

    with left:
        st.markdown('<div class="panel-title">🔧 Build your car</div>', unsafe_allow_html=True)
        cyl_opts = sorted(int(c) for c in df["cylinders"].dropna().unique())
        cylinders = st.select_slider("Cylinders", options=cyl_opts,
                                     value=4 if 4 in cyl_opts else cyl_opts[0])
        lo, hi, med = rng_of("displacement")
        displacement = st.slider("Displacement (cu. in.)", lo, hi, med)
        lo, hi, med = rng_of("horsepower")
        horsepower = st.slider("Horsepower", lo, hi, med)
        lo, hi, med = rng_of("weight")
        weight = st.slider("Weight (lbs)", lo, hi, med)
        lo, hi, med = rng_of("acceleration")
        acceleration = st.slider("Acceleration (0 to 60 mph, seconds)", lo, hi, med)

        y_lo, y_hi = int(df["model year"].min()), int(df["model year"].max())
        year = st.slider("Model year", y_lo, y_hi, int(df["model year"].median()),
                         format="%d")
        st.caption(f"Model year {full_year(year)}")
        origins = sorted(df["origin"].dropna().unique())
        origin = st.radio("Country of origin", origins, horizontal=True)

        model_choice = st.selectbox(
            "Model", list(models.keys()),
            index=list(models.keys()).index(best_name),
            format_func=lambda m: f"{m}  (deployed champion)" if m == best_name else m,
        )
        go_btn = st.button("🚀 Predict MPG", use_container_width=True)

    with right:
        st.markdown('<div class="panel-title">📟 Dashboard</div>', unsafe_allow_html=True)
        if go_btn or "last" in st.session_state:
            if go_btn:
                row = pd.DataFrame([{
                    "cylinders": cylinders, "displacement": displacement,
                    "horsepower": horsepower, "weight": weight,
                    "acceleration": acceleration, "model year": year, "origin": origin,
                }])
                preds = {n: float(m.predict(row)[0]) for n, m in models.items()}
                st.session_state["last"] = (preds, model_choice, row)
                st.session_state["fresh"] = True

            preds, chosen, row = st.session_state["last"]
            value = preds[chosen]
            label, color = verdict_for(value)
            avg = float(df["mpg"].mean())

            count_up(value, color)
            st.markdown(
                f'<div style="text-align:center"><span class="verdict" '
                f'style="background:{color}33;border:1px solid {color};color:{color}">'
                f'{label}</span></div>', unsafe_allow_html=True)
            st.plotly_chart(gauge(value, avg), use_container_width=True)

            if st.session_state.pop("fresh", False):
                if value >= 30:
                    st.balloons()
                elif value < 15:
                    st.snow()

            c1, c2, c3 = st.columns(3)
            c1.markdown(tile(f"{value * 0.4251:.1f}", "km per litre", PALETTE["teal"]),
                        unsafe_allow_html=True)
            c2.markdown(tile(f"{235.215 / value:.1f}", "litres per 100 km", PALETTE["amber"]),
                        unsafe_allow_html=True)
            c3.markdown(tile(f"{value - avg:+.1f}", "vs dataset average", PALETTE["magenta"]),
                        unsafe_allow_html=True)

            with st.expander("⛽ Trip cost calculator"):
                t1, t2 = st.columns(2)
                miles = t1.number_input("Trip distance (miles)", 1, 5000, 250)
                price = t2.number_input("Fuel price per gallon", 0.5, 20.0, 3.5, 0.1)
                st.success(f"About **{miles / value:.1f} gallons**, costing "
                           f"**{miles / value * price:,.2f}** for this trip.")

            st.markdown('<div class="panel-title" style="margin-top:14px">'
                        '🤝 What every model says</div>', unsafe_allow_html=True)
            cmp_df = pd.DataFrame({"Model": list(preds.keys()),
                                   "Predicted MPG": list(preds.values())})
            fig = px.bar(cmp_df, x="Predicted MPG", y="Model", orientation="h",
                         color="Model", color_discrete_sequence=MODEL_COLORS,
                         text=cmp_df["Predicted MPG"].round(1))
            fig.update_layout(showlegend=False)
            st.plotly_chart(style_fig(fig, 300), use_container_width=True)
        else:
            st.markdown(
                '<div class="glass" style="text-align:center;padding:60px 20px">'
                '<div style="font-size:3rem">🏁</div>'
                '<div class="panel-title">Set your specs, then press Predict MPG</div>'
                '<div style="color:#BFB7E6">The dashboard lights up here.</div></div>',
                unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Tab: Explore
# ----------------------------------------------------------------------------
with tab_explore:
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(tile(f"{len(df):,}", "cars in dataset", PALETTE["teal"]), unsafe_allow_html=True)
    k2.markdown(tile(f"{df['mpg'].mean():.1f}", "average MPG", PALETTE["amber"]), unsafe_allow_html=True)
    k3.markdown(tile(f"{df['mpg'].max():.1f}", "best MPG", PALETTE["violet"]), unsafe_allow_html=True)
    k4.markdown(tile(f"{int(df[NUMERIC].isna().sum().sum())}", "missing values fixed",
                     PALETTE["magenta"]), unsafe_allow_html=True)
    st.write("")

    st.markdown('<div class="panel-title">🎞️ Press play: how cars changed year by year</div>',
                unsafe_allow_html=True)
    anim_df = df.dropna(subset=["weight", "horsepower"]).copy()
    anim_df["year"] = anim_df["model year"].apply(full_year)
    anim_df = anim_df.sort_values("year")
    fig = px.scatter(
        anim_df, x="weight", y="mpg", size="horsepower", color="origin",
        animation_frame="year", hover_name="car name",
        color_discrete_map={"USA": PALETTE["magenta"], "Europe": PALETTE["teal"],
                            "Japan": PALETTE["amber"]},
        range_x=[anim_df["weight"].min() - 100, anim_df["weight"].max() + 100],
        range_y=[anim_df["mpg"].min() - 3, anim_df["mpg"].max() + 3],
        size_max=26,
    )
    st.plotly_chart(style_fig(fig, 480), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="mpg", color="origin", nbins=30, barmode="overlay",
                           opacity=0.75, title="MPG distribution by origin",
                           color_discrete_map={"USA": PALETTE["magenta"],
                                               "Europe": PALETTE["teal"],
                                               "Japan": PALETTE["amber"]})
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with c2:
        corr = df[["mpg"] + NUMERIC].corr()
        fig = px.imshow(corr, text_auto=".2f", aspect="auto", title="What moves with MPG?",
                        color_continuous_scale=["#FF3D81", "#1B0F3F", "#19E3C2"],
                        zmin=-1, zmax=1)
        st.plotly_chart(style_fig(fig), use_container_width=True)

    st.markdown('<div class="panel-title">🔍 Pick any two features</div>', unsafe_allow_html=True)
    f1, f2 = st.columns(2)
    xcol = f1.selectbox("X axis", ["mpg"] + NUMERIC, index=4)
    ycol = f2.selectbox("Y axis", ["mpg"] + NUMERIC, index=0)
    fig = px.scatter(df, x=xcol, y=ycol, color="cylinders", hover_name="car name",
                     color_continuous_scale=["#19E3C2", "#7B5CFF", "#FF3D81"],
                     trendline="ols" if len(df) > 5 else None)
    st.plotly_chart(style_fig(fig, 440), use_container_width=True)

# ----------------------------------------------------------------------------
# Tab: Model arena
# ----------------------------------------------------------------------------
with tab_arena:
    best_row = results.iloc[0]
    st.markdown(
        f'<div class="glass"><div class="panel-title">🏆 Deployed model: {best_name}</div>'
        f'Lowest error on the held-out 20% of cars: RMSE <b>{best_row["RMSE"]:.2f}</b> MPG, '
        f'R² <b>{best_row["R² (test)"]:.3f}</b>. '
        f'Five algorithms were trained through the same preprocessing pipeline and '
        f'ranked by test RMSE.</div>', unsafe_allow_html=True)
    st.write("")

    metric = st.radio("Compare by", ["R² (test)", "RMSE", "MAE", "CV R² (5-fold)"],
                      horizontal=True)
    sort_asc = metric in ("RMSE", "MAE")
    plot_df = results.sort_values(metric, ascending=not sort_asc)
    fig = px.bar(plot_df, x=metric, y="Model", orientation="h", color="Model",
                 color_discrete_sequence=MODEL_COLORS, text=plot_df[metric].round(3))
    fig.update_layout(showlegend=False)
    st.plotly_chart(style_fig(fig, 360), use_container_width=True)

    st.dataframe(
        results.style.format({"R² (test)": "{:.3f}", "RMSE": "{:.2f}", "MAE": "{:.2f}",
                              "CV R² (5-fold)": "{:.3f}"})
        .background_gradient(subset=["R² (test)", "CV R² (5-fold)"], cmap="PuBuGn")
        .background_gradient(subset=["RMSE", "MAE"], cmap="RdPu_r"),
        use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(holdout, x="Actual", y="Predicted",
                         title=f"{best_name}: predicted vs actual (unseen cars)",
                         color_discrete_sequence=[PALETTE["teal"]])
        lo, hi = holdout.min().min(), holdout.max().max()
        fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi,
                      line=dict(color=PALETTE["magenta"], dash="dash"))
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with c2:
        fig = px.bar(importance, x="Importance", y="Feature", orientation="h",
                     title="What the champion relies on most",
                     color="Importance",
                     color_continuous_scale=["#7B5CFF", "#FF3D81", "#FFB020"])
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_fig(fig), use_container_width=True)

# ----------------------------------------------------------------------------
# Tab: Pipeline
# ----------------------------------------------------------------------------
with tab_pipe:
    st.markdown('<div class="panel-title">🧪 End-to-end preprocessing pipeline</div>',
                unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    s1.markdown(
        '<div class="glass"><div class="panel-title">1. Fill the gaps</div>'
        'Numeric features use the <b>median</b>; origin uses the <b>most frequent</b> '
        'value. This runs inside the model, so it also protects live predictions.</div>',
        unsafe_allow_html=True)
    s2.markdown(
        '<div class="glass"><div class="panel-title">2. Scale numbers</div>'
        'Cylinders, displacement, horsepower, weight, acceleration and model year go '
        'through <b>StandardScaler</b> so no feature dominates.</div>',
        unsafe_allow_html=True)
    s3.markdown(
        '<div class="glass"><div class="panel-title">3. Encode categories</div>'
        '<b>One-hot encoding</b> turns country of origin into separate yes/no columns '
        'and ignores unseen values safely.</div>', unsafe_allow_html=True)
    st.write("")

    miss = df[["mpg"] + NUMERIC + CATEGORICAL].isna().sum().reset_index()
    miss.columns = ["Feature", "Missing values"]
    fig = px.bar(miss, x="Feature", y="Missing values", color="Missing values",
                 title="Missing values found in your dataset (before cleaning)",
                 color_continuous_scale=["#19E3C2", "#FFB020", "#FF3D81"])
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(style_fig(fig, 340), use_container_width=True)

    st.code(
        "Pipeline([\n"
        "  ('prep', ColumnTransformer([\n"
        "      ('num', Pipeline([SimpleImputer('median'), StandardScaler()]), numeric_cols),\n"
        "      ('cat', Pipeline([SimpleImputer('most_frequent'), OneHotEncoder()]), ['origin']),\n"
        "  ])),\n"
        "  ('model', best_regressor),\n"
        "])", language="python")

# ----------------------------------------------------------------------------
# Tab: Data
# ----------------------------------------------------------------------------
with tab_data:
    st.markdown('<div class="panel-title">🗂️ Your dataset</div>', unsafe_allow_html=True)
    search = st.text_input("Search by car name", placeholder="e.g. toyota")
    view = df if not search else df[df["car name"].astype(str).str.contains(
        search, case=False, na=False)]
    st.dataframe(view, use_container_width=True, height=420)
    st.download_button("⬇️ Download cleaned data (CSV)",
                       view.to_csv(index=False).encode("utf-8"),
                       file_name="auto-mpg-cleaned.csv", mime="text/csv")
    st.markdown("**Summary statistics**")
    st.dataframe(df[["mpg"] + NUMERIC].describe().T, use_container_width=True)
