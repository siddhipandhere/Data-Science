"""
Bank Term-Deposit Predictor — Streamlit app
Data Science Mini Project · Siddhi Pandhere, Pillai College of Engineering

Ensemble of AdaBoost + Gradient Boosting + XGBoost (soft voting) trained on
the UCI Bank Marketing dataset (bank-additional-full.csv).
"""
import io
import os
import zipfile

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

st.set_page_config(page_title="Term Deposit Predictor", page_icon="🏦", layout="wide")

YES = "#1F7A5C"
NO = "#8A989A"
BRASS = "#B8741A"
TEAL = "#0F4C4A"

DATA_FILE = "bank-additional-full.csv"
URLS = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip",
    "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip",
]

# Best hyper-parameters found by GridSearchCV in the notebook
BEST = {
    "ada": dict(n_estimators=200, learning_rate=1.0),
    "gb": dict(n_estimators=100, learning_rate=0.1, max_depth=4),
    "xgb": dict(n_estimators=200, learning_rate=0.05, max_depth=3),
}


# ----------------------------------------------------------------------------
# Data + model (cached so they run once per server start)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading the UCI Bank Marketing dataset…")
def load_data() -> pd.DataFrame:
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE, sep=";")
    for url in URLS:
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
            r.raise_for_status()
            z = zipfile.ZipFile(io.BytesIO(r.content))
            if "bank-additional.zip" in z.namelist():  # bundled archive nests a zip
                z = zipfile.ZipFile(io.BytesIO(z.read("bank-additional.zip")))
            name = next(n for n in z.namelist() if n.endswith("bank-additional-full.csv"))
            return pd.read_csv(z.open(name), sep=";")
        except Exception:
            continue
    st.error("Could not load the dataset. Put `bank-additional-full.csv` next to app.py.")
    st.stop()


@st.cache_resource(show_spinner="Training AdaBoost, Gradient Boosting and XGBoost (first load takes about a minute)…")
def train(df: pd.DataFrame):
    data = df.drop(columns=["duration"]).copy()
    data["y"] = data["y"].map({"yes": 1, "no": 0})
    X = pd.get_dummies(data.drop(columns=["y"]))
    y = data["y"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    spw = (y_train == 0).sum() / (y_train == 1).sum()

    ada = AdaBoostClassifier(random_state=42, **BEST["ada"]).fit(X_train, y_train)
    gb = GradientBoostingClassifier(random_state=42, **BEST["gb"]).fit(X_train, y_train)
    xgb = XGBClassifier(random_state=42, eval_metric="logloss",
                        scale_pos_weight=spw, **BEST["xgb"]).fit(X_train, y_train)
    ens = VotingClassifier(
        estimators=[("adaboost", ada), ("gboost", gb), ("xgboost", xgb)], voting="soft"
    ).fit(X_train, y_train)

    models = {"AdaBoost": ada, "Gradient Boosting": gb, "XGBoost": xgb, "Ensemble": ens}
    rows, probas = [], {}
    for name, m in models.items():
        pred = m.predict(X_test)
        proba = m.predict_proba(X_test)[:, 1]
        probas[name] = proba
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "ROC-AUC": roc_auc_score(y_test, proba),
            "Precision (yes)": precision_score(y_test, pred, zero_division=0),
            "Recall (yes)": recall_score(y_test, pred),
            "F1 (yes)": f1_score(y_test, pred),
        })
    metrics = pd.DataFrame(rows).set_index("Model")
    importances = pd.Series(xgb.feature_importances_, index=X.columns).sort_values(ascending=False)
    return dict(models=models, metrics=metrics, probas=probas, y_test=y_test,
                columns=X.columns, importances=importances, n_train=len(X_train),
                n_test=len(X_test))


def encode(row: dict, columns) -> pd.DataFrame:
    return pd.get_dummies(pd.DataFrame([row])).reindex(columns=columns, fill_value=0)


df = load_data()
art = train(df)
metrics = art["metrics"]

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.title("🏦 Term Deposit Predictor")
page = st.sidebar.radio(
    "Go to", ["Home", "Explore the data", "Model performance", "Predict a customer"]
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Data Science Mini Project  \nSiddhi Pandhere  \nPillai College of Engineering, New Panvel"
)
st.sidebar.caption("Data: [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)")

# ----------------------------------------------------------------------------
# Home
# ----------------------------------------------------------------------------
if page == "Home":
    st.title("Which bank customers will say yes to a term deposit?")
    st.write(
        "A Portuguese bank phoned **41,188** customers to sell a fixed-term deposit, and only "
        "about 1 in 9 subscribed. This app combines **AdaBoost**, **Gradient Boosting** and "
        "**XGBoost** into a soft-voting ensemble that predicts subscription *before* the call is made."
    )
    yes_rate = (df["y"] == "yes").mean()
    e = metrics.loc["Ensemble"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Calls analysed", f"{len(df):,}")
    c2.metric("Subscribed", f"{yes_rate:.1%}")
    c3.metric("Ensemble recall (yes)", f"{e['Recall (yes)']:.2f}",
              f"{e['Recall (yes)'] - metrics.loc['AdaBoost', 'Recall (yes)']:+.2f} vs AdaBoost")
    c4.metric("Ensemble ROC-AUC", f"{e['ROC-AUC']:.3f}")

    st.subheader("How the model is built")
    steps = [
        ("Load", "41,188 rows × 21 columns from the UCI archive."),
        ("Remove leakage", "Drop `duration`: call length is only known after the call ends."),
        ("Encode", "One-hot encode categorical columns → 62 features. Trees need no scaling."),
        ("Split", f"Stratified 80/20: {art['n_train']:,} train rows, {art['n_test']:,} test rows."),
        ("Tune three boosters", "GridSearchCV (3-fold, ROC-AUC). XGBoost also gets `scale_pos_weight` for imbalance."),
        ("Soft vote", "Average the three models' predicted probabilities."),
    ]
    cols = st.columns(3)
    for i, (t, d) in enumerate(steps):
        with cols[i % 3]:
            st.markdown(f"**Step {i + 1} · {t}**  \n{d}")

    st.info("Use the sidebar to explore the data, compare the models, or score your own customer.")

# ----------------------------------------------------------------------------
# Explore
# ----------------------------------------------------------------------------
elif page == "Explore the data":
    st.title("Explore the data")
    counts = df["y"].value_counts()
    c1, c2 = st.columns([1, 2])
    with c1:
        fig = px.pie(values=counts.values, names=counts.index, hole=0.55,
                     color=counts.index, color_discrete_map={"yes": YES, "no": NO})
        fig.update_layout(title="Target: subscribed?", showlegend=True, height=340,
                          margin=dict(t=50, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown(
            f"Only **{counts['yes']:,}** of **{len(df):,}** calls ended in a subscription "
            f"(**{counts['yes'] / len(df):.1%}**). A model that always says *no* would be "
            f"{counts['no'] / len(df):.1%} accurate and useless, which is why we judge models "
            "by **recall** and **ROC-AUC** on the *yes* class."
        )

    st.subheader("Subscription rate by feature")
    cat_cols = ["job", "marital", "education", "default", "housing", "loan",
                "contact", "month", "day_of_week", "poutcome"]
    col = st.selectbox("Choose a categorical feature", cat_cols, index=cat_cols.index("poutcome"))
    rate = (df.assign(yes=(df["y"] == "yes"))
              .groupby(col)["yes"].agg(["mean", "size"]).reset_index()
              .sort_values("mean", ascending=False))
    fig = px.bar(rate, x=col, y="mean", text=rate["mean"].map("{:.1%}".format),
                 hover_data={"size": True}, color_discrete_sequence=[TEAL])
    fig.add_hline(y=(df["y"] == "yes").mean(), line_dash="dash", line_color=BRASS,
                  annotation_text="overall rate", annotation_position="top right")
    fig.update_layout(yaxis_tickformat=".0%", yaxis_title="subscription rate", height=420)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Numeric features by outcome")
    num_cols = ["age", "campaign", "euribor3m", "nr.employed", "emp.var.rate", "cons.conf.idx", "cons.price.idx"]
    ncol = st.selectbox("Choose a numeric feature", num_cols, index=num_cols.index("euribor3m"))
    fig = px.box(df, x="y", y=ncol, color="y", color_discrete_map={"yes": YES, "no": NO})
    fig.update_layout(showlegend=False, height=420, xaxis_title="subscribed")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Preview raw data"):
        st.dataframe(df.head(100), use_container_width=True)

# ----------------------------------------------------------------------------
# Performance
# ----------------------------------------------------------------------------
elif page == "Model performance":
    st.title("Model performance")
    st.caption(f"Held-out test set of {art['n_test']:,} customers.")
    st.dataframe(
        metrics.style.format("{:.4f}").highlight_max(axis=0, color="#D6EDE3"),
        use_container_width=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Accuracy hides missed subscribers")
        m = metrics.reset_index().melt(id_vars="Model", value_vars=["Accuracy", "Recall (yes)", "F1 (yes)"])
        fig = px.bar(m, x="Model", y="value", color="variable", barmode="group",
                     color_discrete_sequence=[NO, YES, BRASS])
        fig.update_layout(yaxis_range=[0, 1], legend_title="", height=400, yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("ROC curves")
        fig = go.Figure()
        colors = {"AdaBoost": NO, "Gradient Boosting": TEAL, "XGBoost": BRASS, "Ensemble": YES}
        for name, proba in art["probas"].items():
            fpr, tpr, _ = roc_curve(art["y_test"], proba)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{name} ({metrics.loc[name, 'ROC-AUC']:.3f})",
                                     line=dict(color=colors[name], width=3 if name == "Ensemble" else 1.5)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="gray"), showlegend=False))
        fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate", height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Confusion matrix")
    c1, c2 = st.columns([1, 1])
    with c1:
        choice = st.selectbox("Model", list(art["models"]), index=3)
        threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05,
                              help="Lower it to catch more subscribers at the cost of more false alarms.")
    pred = (art["probas"][choice] >= threshold).astype(int)
    cm = confusion_matrix(art["y_test"], pred)
    with c2:
        fig = px.imshow(cm, text_auto=True, x=["pred no", "pred yes"], y=["actual no", "actual yes"],
                        color_continuous_scale=["#EEF2F1", TEAL])
        fig.update_layout(height=320, coloraxis_showscale=False, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
    tn, fp, fn, tp = cm.ravel()
    flagged = tp + fp
    with c1:
        st.markdown(
            f"At threshold **{threshold:.2f}**, {choice} finds **{tp}** of **{tp + fn}** subscribers "
            f"(recall {tp / (tp + fn):.2f}) and flags **{flagged}** customers, of whom "
            f"{(tp / flagged if flagged else 0):.0%} subscribe."
        )

    st.subheader("What drives subscription? (XGBoost feature importance)")
    top = art["importances"].head(15).sort_values()
    fig = px.bar(x=top.values, y=top.index, orientation="h", color_discrete_sequence=[TEAL])
    fig.update_layout(height=520, xaxis_title="importance", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Economic indicators (`nr.employed`, `emp.var.rate`, `euribor3m`) dominate, followed by "
               "previous-campaign outcome and contact month.")

# ----------------------------------------------------------------------------
# Predict
# ----------------------------------------------------------------------------
else:
    st.title("Predict a customer")
    st.write("Fill in a customer's details. Defaults match the example customer from the notebook.")

    def opts(col):
        return sorted(df[col].unique().tolist())

    with st.form("customer"):
        st.markdown("**Customer profile**")
        a, b, c, d = st.columns(4)
        age = a.number_input("Age", 17, 98, 41)
        job = b.selectbox("Job", opts("job"), index=opts("job").index("technician"))
        marital = c.selectbox("Marital status", opts("marital"), index=opts("marital").index("married"))
        education = d.selectbox("Education", opts("education"), index=opts("education").index("university.degree"))
        a, b, c = st.columns(3)
        default = a.selectbox("Credit in default?", opts("default"), index=opts("default").index("no"))
        housing = b.selectbox("Housing loan?", opts("housing"), index=opts("housing").index("yes"))
        loan = c.selectbox("Personal loan?", opts("loan"), index=opts("loan").index("no"))

        st.markdown("**This campaign**")
        a, b, c, d = st.columns(4)
        contact = a.selectbox("Contact type", opts("contact"), index=opts("contact").index("cellular"))
        months = ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
        month = b.selectbox("Month", months, index=months.index("may"))
        day = c.selectbox("Day of week", ["mon", "tue", "wed", "thu", "fri"])
        campaign = d.number_input("Contacts this campaign", 1, 60, 2)

        st.markdown("**Previous campaigns**")
        a, b, c = st.columns(3)
        never = a.checkbox("Never contacted before", True)
        pdays = b.number_input("Days since last contact", 0, 30, 5, disabled=never)
        previous = c.number_input("Previous contacts", 0, 10, 0)
        poutcome = st.selectbox("Previous outcome", ["nonexistent", "failure", "success"])

        st.markdown("**Economic conditions at call time**")
        a, b, c, d, e = st.columns(5)
        emp = a.number_input("emp.var.rate", -3.5, 1.5, 1.1, 0.1)
        cpi = b.number_input("cons.price.idx", 92.0, 95.0, 93.99, 0.01)
        cci = c.number_input("cons.conf.idx", -51.0, -26.0, -36.4, 0.1)
        eur = d.number_input("euribor3m", 0.6, 5.1, 4.86, 0.01)
        nre = e.number_input("nr.employed", 4960.0, 5230.0, 5191.0, 1.0)

        submitted = st.form_submit_button("Predict", type="primary")

    row = {
        "age": age, "job": job, "marital": marital, "education": education,
        "default": default, "housing": housing, "loan": loan, "contact": contact,
        "month": month, "day_of_week": day, "campaign": campaign,
        "pdays": 999 if never else pdays, "previous": previous, "poutcome": poutcome,
        "emp.var.rate": emp, "cons.price.idx": cpi, "cons.conf.idx": cci,
        "euribor3m": eur, "nr.employed": nre,
    }
    X_new = encode(row, art["columns"])
    probs = {n: float(m.predict_proba(X_new)[0, 1]) for n, m in art["models"].items()}
    p = probs["Ensemble"]

    c1, c2 = st.columns([1, 1])
    with c1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=p * 100, number={"suffix": "%", "valueformat": ".1f"},
            title={"text": "Ensemble probability of subscribing"},
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": YES if p >= 0.5 else BRASS},
                   "threshold": {"line": {"color": "black", "width": 3}, "value": 50}},
        ))
        fig.update_layout(height=300, margin=dict(t=60, b=10))
        st.plotly_chart(fig, use_container_width=True)
        if p >= 0.5:
            st.success(f"**Will subscribe** · {p:.1%} probability. Worth calling.")
        else:
            st.warning(f"**Will not subscribe** · {p:.1%} probability "
                       f"({p / (df['y'] == 'yes').mean():.1f}× the 11.3% base rate).")
    with c2:
        st.markdown("**What each model says**")
        pb = pd.DataFrame({"Model": list(probs), "Probability": list(probs.values())})
        fig = px.bar(pb, x="Probability", y="Model", orientation="h",
                     text=pb["Probability"].map("{:.1%}".format),
                     color="Model", color_discrete_map={"AdaBoost": NO, "Gradient Boosting": TEAL,
                                                         "XGBoost": BRASS, "Ensemble": YES})
        fig.add_vline(x=0.5, line_dash="dash")
        fig.update_layout(xaxis_range=[0, 1], xaxis_tickformat=".0%", showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("AdaBoost and Gradient Boosting were trained without class weighting, so they "
                   "give lower probabilities than XGBoost. The ensemble averages all three.")
