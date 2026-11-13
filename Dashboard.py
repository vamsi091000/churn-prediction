#!/usr/bin/env python3
"""
Customer Churn Dashboard (Streamlit)

Built on the pipeline Churnthreshold.py (Steps 1-5c). The source CSV is read
directly from the project's public GitHub repository.

Run locally (after the pipeline artifacts have been generated in outputs/):
    pip install -r requirements.txt
    streamlit run Dashboard.py

Needs in the same folder:
    Churnthreshold.py                       (leakage list, split settings, helpers are reused)
    outputs/                                (models, scores and charts written by the pipeline)

Pages (pipeline step in brackets)
    1. Overview            – KPIs and churn rate by any variable
    2. Data & EDA          – leakage columns removed, train/test split, EDA charts   (Steps 1-3a)
    3. Feature selection   – Information Gain, Chi-square, Cramer's V, keep/drop     (Step 3b)
    4. Model performance   – tuning results, test metrics, ROC curves, confusion     (Steps 4, 5a)
    5. Threshold explorer  – move the threshold, every metric, cost/benefit calc     (Step 5c)
    6. Model insights      – decision tree, permutation importance, odds, calibration (Steps 5b, 5c)
    7. Predict churn       – score one customer, or upload a CSV and download predictions
"""
import importlib
import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Customer Churn Dashboard", page_icon="📉", layout="wide")

# Work from the folder this file is in, wherever `streamlit run` is started from,
# so the pipeline module, the CSV and outputs/ are always found.
APP_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(APP_DIR)
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

# Name of the pipeline file WITHOUT ".py". Change this if you saved it under another name.
PIPELINE_MODULE = "Churnthreshold"
DATA_URL = (
    "https://raw.githubusercontent.com/vamsi091000/churn-prediction/"
    "main/E-commerce_Customer_Churn_2026.csv"
)
try:
    cp = importlib.import_module(PIPELINE_MODULE)
except ModuleNotFoundError as err:
    if err.name != PIPELINE_MODULE:        # the pipeline exists but one of ITS imports is missing
        st.error(f"Python package `{err.name}` is not installed. Run:  `pip install -r requirements.txt`")
        st.stop()
    found = sorted(f for f in os.listdir(APP_DIR) if f.endswith(".py"))
    st.error(
        f"Could not find **{PIPELINE_MODULE}.py** in `{APP_DIR}`.\n\n"
        f"Python files in that folder: {', '.join(f'`{f}`' for f in found) or 'none'}.\n\n"
        f"Fix: copy `{PIPELINE_MODULE}.py` into this folder, or, if you saved the pipeline under another "
        f"name, set `PIPELINE_MODULE` near the top of this dashboard file to that name (without `.py`).")
    st.stop()

sns.set_theme(style="whitegrid")

OUT = cp.OUTPUT_DIR
MODEL_NAMES = ["Logistic Regression", "Decision Tree", "SVM", "Random Forest"]
METRIC_LABELS = {"sensitivity": "Sensitivity (recall)", "specificity": "Specificity",
                 "precision": "Precision", "accuracy": "Accuracy", "f1": "F1", "fpr": "False positive rate"}
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.01), 2)


def metrics_at(y_true, prob, t):
    """Confusion-matrix counts and metrics when customers with probability >= t are flagged as churners."""
    y_true = np.asarray(y_true)
    pred = (np.asarray(prob) >= t).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    return {"threshold": t, "TP": tp, "FN": fn, "TN": tn, "FP": fp,
            "sensitivity": sens, "specificity": spec, "precision": prec,
            "accuracy": (tp + tn) / len(y_true),
            "f1": 2 * prec * sens / (prec + sens) if prec + sens else 0.0,
            "fpr": 1 - spec, "flagged_%": pred.mean() * 100}


# =============================================================================
# Data and models (cached so they load once per session)
# =============================================================================
@st.cache_data(show_spinner="Loading dataset from GitHub…")
def load_raw_data(url):
    return pd.read_csv(url)


def require(path):
    if not os.path.exists(path):
        st.error(f"Missing file: `{path}`. Run `python {PIPELINE_MODULE}.py` first, "
                 "then start the dashboard from the same folder.")
        st.stop()
    return path


@st.cache_data(show_spinner="Loading data…")
def load_data():
    raw = load_raw_data(DATA_URL)
    data = raw.drop(columns=[c for c in cp.LEAK_COLS if c in raw.columns])
    X, y = data.drop(columns=[cp.TARGET]), data[cp.TARGET]
    # exactly the same split as the pipeline, so "test" here is the same untouched test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cp.TEST_SIZE, stratify=y, random_state=cp.RANDOM_STATE)
    X_train, X_test = cp.fill_missing_categories(X_train), cp.fill_missing_categories(X_test)
    return cp.fill_missing_categories(X), y, X_train, X_test, y_train, y_test


@st.cache_data(show_spinner="Checking leakage columns…")
def leakage_table():
    """For every dropped column: its group, and how perfectly it separates churners (from the raw data)."""
    raw = load_raw_data(DATA_URL)
    y = raw[cp.TARGET]
    groups = {
        "post_churn": ("Filled in only after churn", "Not known before the customer leaves"),
        "model_outputs": ("Earlier model's scores", "Output of a previous churn model, not a customer fact"),
        "post_churn_actions": ("Actions taken after churn", "Describes what was done after churn was detected"),
        "identifiers": ("Identifiers / constants", "No predictive meaning"),
    }
    rows = []
    for group, cols in cp.LEAKAGE.items():
        label, reason = groups.get(group, (group, ""))
        for c in cols:
            if c not in raw.columns:
                continue
            s = raw[c]
            n_unique = s.nunique(dropna=False)
            if n_unique == 1:
                evidence = f"constant (always {s.iloc[0]})"
            elif c.endswith("_id") or (not pd.api.types.is_numeric_dtype(s) and n_unique > 0.5 * len(s)):
                evidence = f"ID ({n_unique:,} distinct values)"
            elif pd.api.types.is_numeric_dtype(s) and n_unique > 20:
                evidence = f"mean {s[y == 0].mean():,.2f} (retained) vs {s[y == 1].mean():,.2f} (churned)"
            else:
                ct = pd.crosstab(s.astype(str), y)
                pure = ct[(ct == 0).any(axis=1)].values.sum() / len(s)   # rows whose value occurs in one class only
                if pure > 0.5:
                    evidence = f"value alone reveals the outcome for {pure:.0%} of customers"
                else:
                    rate = (ct[1] / ct.sum(axis=1))[ct.sum(axis=1) >= 50]
                    evidence = (f"churn rate {rate.min():.0%}–{rate.max():.0%} across its "
                                f"{len(rate)} values" if n_unique <= 50 else f"{n_unique:,} distinct values")
            rows.append({"group": label, "column": c, "why dropped": reason, "evidence": evidence})
    return pd.DataFrame(rows), raw.shape


@st.cache_data
def load_scores():
    scores = pd.read_csv(require(f"{OUT}/feature_selection_scores.csv"))
    selected = scores.loc[scores.decision == "KEEP", "variable"].tolist()
    return scores, selected


@st.cache_resource(show_spinner="Loading models…")
def load_models():
    return {n: joblib.load(require(cp.model_file(n))) for n in MODEL_NAMES}


@st.cache_data(show_spinner="Scoring the test set…")
def test_probabilities(_models, X_test, y_train, selected):
    """Churn probabilities on the test set. SVM scores -> probabilities with Platt scaling
    fitted on the training out-of-fold scores (same method as Step 5c of the pipeline)."""
    probs = {}
    for name, m in _models.items():
        if name == "SVM":
            oof = pd.read_csv(require(f"{OUT}/oof_scores_train.csv"), index_col=0)
            platt = LogisticRegression().fit(oof[["SVM"]].values, y_train.loc[oof.index])
            probs[name] = platt.predict_proba(m.decision_function(X_test[selected]).reshape(-1, 1))[:, 1]
        else:
            probs[name] = m.predict_proba(X_test[selected])[:, 1]
    return probs


@st.cache_resource
def svm_platt(y_train):
    oof = pd.read_csv(require(f"{OUT}/oof_scores_train.csv"), index_col=0)
    return LogisticRegression().fit(oof[["SVM"]].values, y_train.loc[oof.index])


def predict_proba_all(models, X, y_train):
    out = {}
    for name, m in models.items():
        if name == "SVM":
            out[name] = svm_platt(y_train).predict_proba(m.decision_function(X).reshape(-1, 1))[:, 1]
        else:
            out[name] = m.predict_proba(X)[:, 1]
    return out


def read_csv_if_exists(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def show_image(path, caption=None):
    if os.path.exists(path):
        st.image(path, caption=caption)
    else:
        st.info(f"`{path}` not found – re-run the pipeline to create it.")


def confusion_fig(m, title):
    cm = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Retained", "Churned"], yticklabels=["Retained", "Churned"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    return fig


def pct(v):
    return f"{v:.1%}"


# =============================================================================
# Pages
# =============================================================================
def page_overview(X_all, y_all, X_test, y_test, probs, selected):
    st.title("📉 Customer Churn Dashboard")
    st.caption("E-commerce customer churn – EDA, feature selection and four machine-learning models "
               "(Logistic Regression, Decision Tree, SVM, Random Forest).")

    aucs = {n: roc_auc_score(y_test, p) for n, p in probs.items()}
    best = max(aucs, key=aucs.get)
    c = st.columns(5)
    c[0].metric("Customers", f"{len(y_all):,}")
    c[1].metric("Churn rate", pct(y_all.mean()))
    c[2].metric("Test set (untouched)", f"{len(X_test):,}")
    c[3].metric("Variables selected", f"{len(selected)} of {X_all.shape[1]}")
    c[4].metric("Best test ROC-AUC", f"{aucs[best]:.3f}", help=best)

    st.subheader("Churn rate by variable")
    left, right = st.columns([1, 3])
    with left:
        options = selected + [c for c in X_all.columns if c not in selected]
        var = st.selectbox("Variable", options,
                           format_func=lambda v: f"{v}  ✓" if v in selected else v)
        st.caption("✓ = selected for the models")
    col = X_all[var]
    if cp.is_categorical(col) or col.nunique() <= 12:
        groups = col.astype(str)
    else:
        groups = pd.qcut(col, q=10, duplicates="drop").astype(str)
    table = (pd.DataFrame({"group": groups, "churn": y_all.values})
             .groupby("group", sort=False).churn.agg(["mean", "size"])
             .rename(columns={"mean": "churn_rate", "size": "customers"}))
    if not cp.is_categorical(col) and col.nunique() <= 12:
        table = table.sort_index(key=lambda s: s.astype(float))
    elif not cp.is_categorical(col):
        table = table.sort_index(key=lambda s: s.str.extract(r"([-\d.]+)")[0].astype(float))
    else:
        table = table.sort_values("churn_rate")
    with right:
        st.bar_chart(table["churn_rate"], height=340)
    with left:
        st.dataframe(table.style.format({"churn_rate": "{:.1%}", "customers": "{:,}"}))
    st.caption(f"Overall churn rate: {pct(y_all.mean())}. Numeric variables with many values are shown in deciles.")


def page_data(X_train, X_test, y_all, y_train, y_test):
    st.title("Data & EDA")
    leak, raw_shape = leakage_table()

    st.subheader("Step 1 – Leakage columns removed")
    st.markdown(f"The raw file has **{raw_shape[0]:,} rows × {raw_shape[1]} columns**. "
                f"**{len(leak)} columns** were dropped because their values are only known after a customer "
                "has churned, come from an earlier churn model, or carry no predictive meaning. Keeping them "
                "would give near-perfect but meaningless accuracy.")
    counts = leak.group.value_counts()
    cols = st.columns(len(counts))
    for col, (g, n) in zip(cols, counts.items()):
        col.metric(g, n)
    st.dataframe(leak, hide_index=True)
    st.caption("Evidence is computed from the raw data. 'Value alone reveals the outcome' means the column "
               "by itself tells you whether the customer churned.")

    st.subheader("Step 2 – Train / test split")
    split = pd.DataFrame({"customers": [len(y_all), len(y_train), len(y_test)],
                          "churn_rate": [y_all.mean(), y_train.mean(), y_test.mean()]},
                         index=["All", "Train", "Test"])
    a, b = st.columns([1, 2])
    with a:
        st.dataframe(split.style.format({"customers": "{:,}", "churn_rate": "{:.2%}"}))
    with b:
        st.markdown(f"* **{1 - cp.TEST_SIZE:.0%} / {cp.TEST_SIZE:.0%}** split, stratified on churn, "
                    f"`random_state = {cp.RANDOM_STATE}`.\n"
                    "* The split happens **before** EDA and feature selection, so the test set never "
                    "influences which variables are chosen or how models are tuned.\n"
                    f"* {X_train.shape[1]} candidate variables remain after leakage removal.")

    st.subheader("Step 3a – EDA (training set only)")
    tabs = st.tabs(["Numeric distributions", "Churn rate by category", "Correlation", "Summary table"])
    with tabs[0]:
        st.caption("Distribution of each numeric variable for churned vs retained customers. "
                   "Curves that overlap completely carry little information about churn.")
        show_image(f"{OUT}/eda_numeric_distributions.png")
    with tabs[1]:
        st.caption("Red dashed line = overall churn rate.")
        show_image(f"{OUT}/eda_churn_rate_by_category.png")
    with tabs[2]:
        st.caption("Spearman correlation between numeric predictors – used to find duplicate variables.")
        show_image(f"{OUT}/eda_correlation.png")
    with tabs[3]:
        num = [c for c in X_train.columns if not cp.is_categorical(X_train[c])]
        means = X_train[num].groupby(y_train.values).mean().T
        means.columns = ["retained_mean", "churned_mean"]
        means["difference_%"] = (means.churned_mean / means.retained_mean - 1) * 100
        st.dataframe(means.sort_values("difference_%", key=abs, ascending=False)
                     .style.format("{:,.2f}").background_gradient(subset=["difference_%"], cmap="RdBu_r",
                                                                   vmin=-25, vmax=25))
        st.caption("Training set. Large differences (satisfaction, support tickets, recency) point to "
                   "useful predictors; values near 0% point to variables unrelated to churn.")


def page_feature_selection(scores, selected):
    st.title("Feature selection")
    st.markdown(
        "Computed on the **training set only**. A variable is kept when the Chi-square test is significant "
        f"(p < {cp.ALPHA}), the effect size is meaningful (Cramér's V ≥ {cp.V_MIN}), Information Gain ≥ "
        f"{cp.IG_MIN}, and it is not a duplicate of a stronger variable.")
    st.success(f"Selected {len(selected)} variables: " + ", ".join(f"`{v}`" for v in selected))

    st.subheader("Information Gain (bits)")
    chart = scores.set_index("variable")[["info_gain"]].sort_values("info_gain", ascending=False)
    st.bar_chart(chart, height=380, horizontal=True)

    st.subheader("All scores")
    only_kept = st.toggle("Show selected variables only", value=False)
    show = scores[scores.decision == "KEEP"] if only_kept else scores
    st.dataframe(
        show[["variable", "type", "levels", "info_gain", "gain_ratio", "chi2", "dof", "p_value",
              "cramers_v", "decision"]]
        .style.format({"info_gain": "{:.5f}", "gain_ratio": "{:.5f}", "chi2": "{:,.1f}",
                       "p_value": "{:.2e}", "cramers_v": "{:.4f}"}), hide_index=True)
    with st.expander("What do these measures mean?"):
        st.markdown(
            "- **Information Gain** – how much knowing the variable reduces uncertainty about churn.\n"
            "- **Gain ratio** – Information Gain adjusted for the number of levels.\n"
            "- **Chi-square / p-value** – tests whether the variable and churn are independent.\n"
            "- **Cramér's V** – strength of the association (0 = none, 1 = perfect); needed because with "
            "40,000 rows even tiny effects are 'significant'.")


def page_model_performance(probs, y_test):
    st.title("Model performance – test set")
    st.caption("All numbers come from the 10,000-customer test set, which played no part in feature "
               "selection or tuning.")

    rows = []
    for name, p in probs.items():
        m = metrics_at(y_test, p, 0.50)
        rows.append({"model": name, "roc_auc": roc_auc_score(y_test, p), **{k: m[k] for k in METRIC_LABELS}})
    res = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)

    st.subheader("Metrics at threshold 0.50")
    st.dataframe(res.style.format({c: "{:.3f}" for c in res.columns if c != "model"})
                 .highlight_max(subset=["roc_auc", "sensitivity", "specificity", "precision", "accuracy", "f1"],
                                color="#C8E6C9")
                 .highlight_min(subset=["fpr"], color="#C8E6C9"), hide_index=True)
    st.caption("Green = best in column. SVM probabilities come from Platt scaling, so its 0.50 row can differ "
               "slightly from the SVM's built-in cut-off.")

    metric = st.selectbox("Compare models on", list(METRIC_LABELS) + ["roc_auc"],
                          format_func=lambda k: METRIC_LABELS.get(k, "ROC-AUC"))
    st.bar_chart(res.set_index("model")[[metric]], height=300)

    left, right = st.columns(2)
    with left:
        st.subheader("ROC curves")
        fig, ax = plt.subplots(figsize=(6, 5))
        for name in res.model:
            fpr, tpr, _ = roc_curve(y_test, probs[name])
            ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC {roc_auc_score(y_test, probs[name]):.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (0.500)")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("Sensitivity")
        ax.legend(loc="lower right", fontsize=8)
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.subheader("Confusion matrix")
        name = st.selectbox("Model", list(res.model), key="cm_model")
        st.pyplot(confusion_fig(metrics_at(y_test, probs[name], 0.50), f"{name} – threshold 0.50"))
        plt.close("all")

    cv = read_csv_if_exists(f"{OUT}/cv_summary.csv")
    if cv is not None:
        st.subheader("Tuning on the training set (Step 4)")
        fast = getattr(cp, "FAST_MODE", None)
        mode = "" if fast is None else (" · FAST mode (smaller grids)" if fast else " · FULL grids")
        st.caption(f"{getattr(cp, 'CV_FOLDS', 5)}-fold stratified cross-validation, metric ROC-AUC{mode}.")
        left, right = st.columns([3, 2])
        with left:
            st.dataframe(cv[["model", "baseline_cv_auc", "tuned_cv_auc", "tuned_cv_std", "best_params"]]
                         .style.format({"baseline_cv_auc": "{:.4f}", "tuned_cv_auc": "{:.4f}",
                                        "tuned_cv_std": "{:.4f}"}), hide_index=True)
            test_auc = {n: roc_auc_score(y_test, p) for n, p in probs.items()}
            check = cv[["model", "tuned_cv_auc"]].assign(test_auc=cv.model.map(test_auc))
            check["gap"] = check.test_auc - check.tuned_cv_auc
            st.dataframe(check.style.format({"tuned_cv_auc": "{:.4f}", "test_auc": "{:.4f}", "gap": "{:+.4f}"}),
                         hide_index=True)
            st.caption("Gap = test ROC-AUC − tuned CV ROC-AUC. Close to 0 = no overfitting.")
        with right:
            show_image(f"{OUT}/cv_model_comparison.png")


def page_threshold(probs, y_test):
    st.title("Threshold explorer")
    st.markdown("A customer is flagged as a churner when their churn probability is **at or above** the "
                "threshold. Lower thresholds catch more churners but flag more loyal customers by mistake.")

    c1, c2 = st.columns([1, 2])
    with c1:
        name = st.selectbox("Model", MODEL_NAMES, index=MODEL_NAMES.index("Random Forest"))
        t = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.01)
    p = probs[name]
    m = metrics_at(y_test, p, t)
    base = metrics_at(y_test, p, 0.50)

    cols = st.columns(6)
    for col, key in zip(cols, METRIC_LABELS):
        delta = m[key] - base[key]
        col.metric(METRIC_LABELS[key], pct(m[key]), f"{delta:+.1%} vs 0.50" if abs(delta) > 1e-9 else None,
                   delta_color="inverse" if key == "fpr" else "normal")
    st.caption(f"Flagged as churners: {m['flagged_%']:.1f}% of test customers "
               f"({m['TP'] + m['FP']:,} of {len(y_test):,}).")

    sweep = pd.DataFrame([metrics_at(y_test, p, x) for x in THRESHOLDS])
    left, right = st.columns([2, 1])
    with left:
        fig, ax = plt.subplots(figsize=(8, 4.6))
        colors = {"sensitivity": "#2E7D32", "specificity": "#1565C0", "precision": "#EF6C00",
                  "accuracy": "#6A1B9A", "f1": "#000000", "fpr": "#C62828"}
        for key, color in colors.items():
            ax.plot(sweep.threshold, sweep[key], color=color, lw=2 if key == "f1" else 1.5,
                    label=METRIC_LABELS[key])
        ax.axvline(t, color="#555555", ls="--", lw=1.5, label=f"Selected ({t:.2f})")
        ax.set_xlabel("Threshold")
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1, 0.5))
        ax.set_title(f"{name} – metrics across thresholds (test set)", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.pyplot(confusion_fig(m, f"{name} – threshold {t:.2f}"))
        plt.close("all")

    chosen = read_csv_if_exists(f"{OUT}/threshold_comparison.csv")
    if chosen is not None:
        with st.expander("Thresholds chosen on the training set (Step 5c)"):
            st.dataframe(chosen[chosen.model == name][["rule", "threshold", "test_sensitivity", "test_specificity",
                                                        "test_precision", "test_f1", "train_oof_sensitivity",
                                                        "sensitivity_gap"]]
                         .style.format({c: "{:.3f}" for c in chosen.columns if c not in ("model", "rule")}), hide_index=True)

    st.subheader("Cost / benefit calculator")
    st.caption("Set your own business numbers to find the threshold that earns the most. "
               "Assumes every flagged customer receives one retention offer.")
    a, b, c = st.columns(3)
    value = a.number_input("Value of keeping one customer (₹)", min_value=0.0, value=5000.0, step=500.0)
    cost = b.number_input("Cost of one retention offer (₹)", min_value=0.0, value=500.0, step=50.0)
    success = c.slider("Share of contacted churners who stay", 0.0, 1.0, 0.30, 0.05)
    sweep["net_value"] = sweep.TP * success * value - (sweep.TP + sweep.FP) * cost
    best = sweep.loc[sweep.net_value.idxmax()]
    now = sweep.loc[(sweep.threshold - t).abs().idxmin()]
    k = st.columns(3)
    k[0].metric("Net value at selected threshold", f"₹{now.net_value:,.0f}")
    k[1].metric("Best threshold for these costs", f"{best.threshold:.2f}")
    k[2].metric("Net value at best threshold", f"₹{best.net_value:,.0f}")
    st.line_chart(sweep.set_index("threshold")[["net_value"]], height=260)
    st.caption("Net value = churners caught × success rate × customer value − offers sent × offer cost "
               f"(on the {len(y_test):,} test customers).")


def page_insights(selected):
    st.title("Model insights")
    tabs = st.tabs(["Decision tree", "Variable importance", "Logistic Regression odds ratios", "Calibration"])

    with tabs[0]:
        st.markdown("Top levels of the tuned Decision Tree. Each box: the condition, training customers in the "
                    "branch and their churn rate (green = low churn, red = high churn).")
        show_image(f"{OUT}/decision_tree.png")
        rules = f"{OUT}/decision_tree_rules.txt"
        if os.path.exists(rules):
            with st.expander("Full decision tree rules"):
                st.code(open(rules).read(), language=None)

    with tabs[1]:
        imp = read_csv_if_exists(f"{OUT}/permutation_importance.csv")
        if imp is not None:
            st.markdown("**Permutation importance** – drop in test ROC-AUC when a variable is shuffled.")
            table = imp.pivot(index="variable", columns="model", values="auc_drop")
            table = table.sort_values(table.columns[0], ascending=False)
            st.bar_chart(table, height=380, horizontal=True, stack=False)
            st.dataframe(table.style.format("{:.4f}"))
        else:
            show_image(f"{OUT}/permutation_importance.png")

    with tabs[2]:
        odds = read_csv_if_exists(f"{OUT}/logistic_regression_odds_ratios.csv")
        if odds is not None:
            st.markdown("Odds ratio > 1 raises churn odds, < 1 lowers them. Numeric variables: per 1 standard "
                        "deviation increase. Coefficient 0 = removed by the L1 penalty.")
            st.dataframe(odds.style.format({"coefficient": "{:.3f}", "odds_ratio": "{:.3f}"})
                         .background_gradient(subset=["coefficient"], cmap="RdYlGn_r", vmin=-0.8, vmax=0.8), hide_index=True)

    with tabs[3]:
        st.markdown("If a model says '60% churn', do about 60% of those customers really churn? "
                    "Points on the diagonal = trustworthy probabilities. Lower Brier score = better.")
        show_image(f"{OUT}/calibration.png")


def page_predict(models, X_train, y_train, selected):
    st.title("Predict churn")
    tab1, tab2 = st.tabs(["One customer", "Upload a CSV"])

    with tab1:
        with st.form("customer"):
            cols = st.columns(4)
            inputs = {}
            for i, var in enumerate(selected):
                s = X_train[var]
                with cols[i % 4]:
                    if cp.is_categorical(s):
                        levels = sorted(s.unique())
                        inputs[var] = st.selectbox(var, levels, index=levels.index(s.mode()[0]))
                    else:
                        if pd.api.types.is_integer_dtype(s):
                            lo, hi, med, step = int(s.min()), int(s.max()), int(s.median()), 1
                        else:
                            lo, hi, med, step = float(s.min()), float(s.max()), float(s.median()), 0.01
                        inputs[var] = st.number_input(var, min_value=lo, max_value=hi, value=med, step=step)
            t = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.01, key="pred_t")
            submitted = st.form_submit_button("Predict", type="primary")
        if submitted:
            row = pd.DataFrame([inputs])[selected]
            p = predict_proba_all(models, row, y_train)
            cols = st.columns(len(p))
            for col, (name, prob) in zip(cols, p.items()):
                flag = "🔴 Likely to churn" if prob[0] >= t else "🟢 Likely to stay"
                col.metric(name, pct(prob[0]), flag, delta_color="off")
            avg = float(np.mean([v[0] for v in p.values()]))
            st.progress(min(max(avg, 0.0), 1.0), text=f"Average churn probability across models: {avg:.1%}")

    with tab2:
        st.markdown("The file needs these columns (others are ignored): " + ", ".join(f"`{v}`" for v in selected))
        up = st.file_uploader("Customer CSV", type="csv")
        if up is not None:
            new = pd.read_csv(up)
            missing = [c for c in selected if c not in new.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
                return
            X_new = cp.fill_missing_categories(new[selected])
            name = st.selectbox("Model", MODEL_NAMES, index=MODEL_NAMES.index("Random Forest"), key="batch_model")
            t = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.01, key="batch_t")
            prob = predict_proba_all({name: models[name]}, X_new, y_train)[name]
            out = new.copy()
            out["churn_probability"] = prob.round(4)
            out["predicted_churn"] = (prob >= t).astype(int)
            k = st.columns(3)
            k[0].metric("Customers scored", f"{len(out):,}")
            k[1].metric("Flagged as churners", f"{out.predicted_churn.sum():,}")
            k[2].metric("Share flagged", pct(out.predicted_churn.mean()))
            st.dataframe(out.sort_values("churn_probability", ascending=False).head(200), hide_index=True)
            st.download_button("Download predictions (CSV)", out.to_csv(index=False).encode(),
                               file_name="churn_predictions.csv", mime="text/csv")


# =============================================================================
# App
# =============================================================================
def main():
    X_all, y_all, X_train, X_test, y_train, y_test = load_data()
    scores, selected = load_scores()
    models = load_models()
    probs = test_probabilities(models, X_test, y_train, selected)

    st.sidebar.title("Churn dashboard")
    page = st.sidebar.radio("Page", ["Overview", "Data & EDA", "Feature selection", "Model performance",
                                     "Threshold explorer", "Model insights", "Predict churn"])
    st.sidebar.divider()
    st.sidebar.caption(f"{len(X_train):,} training / {len(X_test):,} test customers\n\n"
                       f"Models: {', '.join(MODEL_NAMES)}")

    if page == "Overview":
        page_overview(X_all, y_all, X_test, y_test, probs, selected)
    elif page == "Data & EDA":
        page_data(X_train, X_test, y_all, y_train, y_test)
    elif page == "Feature selection":
        page_feature_selection(scores, selected)
    elif page == "Model performance":
        page_model_performance(probs, y_test)
    elif page == "Threshold explorer":
        page_threshold(probs, y_test)
    elif page == "Model insights":
        page_insights(selected)
    else:
        page_predict(models, X_train, y_train, selected)


main()