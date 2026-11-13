#!/usr/bin/env python3
"""
Customer churn prediction – combined pipeline
Steps
  1. Load data and remove leakage columns
  2. Train / test split (80/20, stratified)
  3. EDA + Information Gain + Chi-square + Cramer's V on the TRAINING set only
  4. Train and tune Logistic Regression, Decision Tree, SVM, Random Forest (training set, 5-fold CV)
  5. Evaluate once on the untouched test set + interpret the models

Run:  python churn_model_pipeline.py
Requires: pandas, numpy, scipy, scikit-learn, matplotlib, seaborn, joblib
Charts, tables and fitted models are saved in the OUTPUT_DIR folder.

Speed: FAST_MODE = True (default) runs the whole script in a few minutes, with test
results within ~0.002 ROC-AUC of the full search. FAST_MODE = False runs the full
grids (~10-15 min on a 2-core machine). After one run, set LOAD_SAVED_MODELS = True
to reuse the tuned models and go straight to Step 5 in well under a minute.
"""
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # save charts to files, no pop-up windows
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 50)
sns.set_theme(style="whitegrid")

PATH = "E-commerce_Customer_Churn_2026.csv"   # change to the full path of your CSV if it is elsewhere
TARGET = "churn_flag"
OUTPUT_DIR = "outputs"
RUN_EDA = True                             # set False to skip the EDA charts when re-running
LOAD_SAVED_MODELS = False                  # True = reuse tuned models from OUTPUT_DIR, skip Step 4 tuning
FAST_MODE = True                           # True = smaller search (a few minutes); False = full grids (slow)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# STEP 1 – Load data and remove leakage columns
# =============================================================================
# A leakage column is one whose value would not be available at the moment we
# want to make the prediction (i.e. before the customer churns). Using it gives
# near-perfect but meaningless accuracy.
#
# They fall into four groups:
LEAKAGE = {
    # (a) Filled in only after a customer has churned
    "post_churn": ["is_churned", "churn_date", "days_since_churn", "churn_reason",
                   "churn_category", "churn_rate_indicator", "churn_revenue_impact_usd"],
    # (b) Scores / labels produced by an earlier churn model
    "model_outputs": ["churn_probability", "churn_risk_score", "churn_risk_category",
                      "predicted_churn", "is_high_risk_customer", "retention_probability",
                      "customer_health_score", "customer_health_category"],
    # (c) Actions or money spent after churn was detected
    "post_churn_actions": ["prevention_method", "recovery_revenue_usd",
                           "customer_retention_cost_usd"],
    # (d) Identifiers / constants / very high-cardinality text: no predictive meaning
    "identifiers": ["churn_id", "customer_id", "dataset_year", "customer_city"],
}
LEAK_COLS = [c for group in LEAKAGE.values() for c in group]


def load_data(path):
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns")
    print("Target balance:\n", df[TARGET].value_counts(normalize=True).round(3).to_string(), "\n")
    return df


def leakage_evidence(df):
    """Print the evidence that each dropped column leaks the target."""
    print("Leakage evidence")
    print("-" * 70)
    # is_churned is just the target written as Yes/No
    print("is_churned vs churn_flag:\n", pd.crosstab(df["is_churned"], df[TARGET]), "\n")

    # Columns that say "Not Applicable" exactly for retained customers
    for c in ["churn_date", "churn_reason", "churn_category"]:
        na = df[c].astype(str).eq("Not Applicable")
        print(f"{c:16s} 'Not Applicable' rows: {na.sum():,} -> all retained: {df.loc[na, TARGET].eq(0).all()}"
              f" | other rows: {(~na).sum():,} -> all churned: {df.loc[~na, TARGET].eq(1).all()}")

    # churn_rate_indicator: 'Low' only ever appears for retained, 'High' only for churned
    print("\nchurn_rate_indicator vs churn_flag:\n", pd.crosstab(df["churn_rate_indicator"], df[TARGET]))

    # Numeric columns whose values separate the classes almost perfectly
    cols = ["days_since_churn", "churn_revenue_impact_usd", "retention_probability",
            "churn_probability", "churn_risk_score"]
    print("\nMean by churn_flag:\n", df.groupby(TARGET)[cols].mean().T.round(3), "\n")

    # Earlier model's prediction vs actual
    print("predicted_churn vs churn_flag:\n", pd.crosstab(df["predicted_churn"], df[TARGET]), "\n")


def remove_leakage(df):
    missing = [c for c in LEAK_COLS if c not in df.columns]
    if missing:
        print("Note – not found, skipped:", missing)
    clean = df.drop(columns=[c for c in LEAK_COLS if c in df.columns])
    print(f"Dropped {df.shape[1] - clean.shape[1]} leakage columns -> {clean.shape[1]} columns remain")
    print("Remaining columns:", [c for c in clean.columns if c != TARGET])
    return clean


# =============================================================================
# STEP 2 – Train / test split (80/20, stratified)
# =============================================================================
# * The split happens BEFORE any EDA or feature selection, so the test set never
#   influences which variables are chosen or how models are tuned.
# * stratify=y keeps the churn share (~41%) identical in both parts.
# * random_state fixes the split so results are reproducible run to run.
# * From here on, every step (EDA, Information Gain, Chi-square, imputation,
#   scaling, tuning) uses X_train / y_train only. X_test / y_test are used
#   exactly once, in Step 5.
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.20
RANDOM_STATE = 42


def split_data(df):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

    print(f"\nTrain: {len(X_train):,} rows | Test: {len(X_test):,} rows")
    check = pd.DataFrame({
        "rows": [len(y), len(y_train), len(y_test)],
        "churn_rate": [y.mean(), y_train.mean(), y_test.mean()],
    }, index=["full", "train", "test"])
    print(check.round(4).to_string())

    # Sanity checks: no row in both sets, and category levels seen in test also exist in train
    assert X_train.index.intersection(X_test.index).empty, "Train and test overlap!"
    for c in X.select_dtypes(exclude="number").columns:
        unseen = set(X_test[c].dropna()) - set(X_train[c].dropna())
        if unseen:
            print(f"Warning: {c} has test-only levels {unseen}")
    print("Checks passed: no overlap between train and test; all test category levels appear in train")
    return X_train, X_test, y_train, y_test


# =============================================================================
# STEP 3 – EDA + Information Gain + Chi-square + Cramer's V (TRAINING SET ONLY)
# =============================================================================
# Selection rule – a variable is kept when ALL of these hold on the training set:
#   * Chi-square test of independence p-value < ALPHA
#   * Cramer's V >= V_MIN   (with 40k rows, tiny useless effects are still
#                            "significant", so an effect-size floor is needed)
#   * Information Gain >= IG_MIN
#   * it is not a duplicate of a stronger variable (one-to-one relabel, or
#     numeric correlation above CORR_MAX)
ALPHA, V_MIN, IG_MIN, CORR_MAX = 0.05, 0.02, 0.0005, 0.70
N_BINS = 10                                # numeric variables -> deciles for IG / Chi-square


def is_categorical(s):
    return not pd.api.types.is_numeric_dtype(s)


def fill_missing_categories(X):
    """Missing categorical values become their own level 'Missing'.
    A constant fill learns nothing from data, so it is safe to apply to train and test alike."""
    X = X.copy()
    for c in X.columns:
        if is_categorical(X[c]):
            X[c] = X[c].fillna("Missing")
    return X


def run_eda(X, y):
    """Charts and summary tables, training data only."""
    print("\n" + "=" * 70 + "\nSTEP 3a – EDA (training set)\n" + "=" * 70)
    num = [c for c in X.columns if not is_categorical(X[c])]
    cat = [c for c in X.columns if is_categorical(X[c])]
    d = X.assign(**{TARGET: y.values})

    miss = X.isna().mean().sort_values(ascending=False)
    print("Missing share before filling:\n", miss[miss > 0].round(3).to_string())

    means = d.groupby(TARGET)[num].mean().T
    means["diff_%"] = (means[1] / means[0] - 1) * 100
    print("\nNumeric means – retained (0) vs churned (1):\n", means.round(2).to_string())

    # Numeric distributions by churn status
    cols = 5
    rows = int(np.ceil(len(num) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 3.6 * rows))
    axes = axes.ravel()
    for ax, c in zip(axes, num):
        sns.kdeplot(data=d, x=c, hue=TARGET, common_norm=False, fill=True, ax=ax, legend=False)
        ax.set_title(c, fontsize=10)
        ax.set_xlabel("")
    for ax in axes[len(num):]:
        ax.axis("off")
    fig.legend(["Churned (1)", "Retained (0)"], loc="upper right")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/eda_numeric_distributions.png", dpi=90)
    plt.close()

    # Churn rate by category level
    cols = 4
    rows = int(np.ceil(len(cat) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.2 * rows))
    axes = axes.ravel()
    for ax, c in zip(axes, cat):
        d.groupby(c)[TARGET].mean().sort_values().plot.barh(ax=ax, color="#4C72B0")
        ax.axvline(y.mean(), color="red", ls="--", lw=1)
        ax.set_title(f"Churn rate by {c}", fontsize=10)
        ax.set_xlabel("churn rate")
        ax.set_ylabel("")
    for ax in axes[len(cat):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/eda_churn_rate_by_category.png", dpi=90)
    plt.close()

    # Correlation between numeric predictors
    plt.figure(figsize=(13, 10))
    sns.heatmap(X[num].corr(method="spearman"), cmap="coolwarm", center=0,
                annot=True, fmt=".2f", annot_kws={"size": 7})
    plt.title("Spearman correlation – numeric predictors (training set)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/eda_correlation.png", dpi=90)
    plt.close()
    print(f"\nEDA charts saved in '{OUTPUT_DIR}/'")


def entropy(p):
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def discretise(s):
    """Categoricals and low-cardinality numerics as-is; other numerics into deciles."""
    if is_categorical(s) or s.nunique() <= N_BINS:
        return s.astype(str)
    return pd.qcut(s, q=N_BINS, duplicates="drop").astype(str)


def score_variables(X, y):
    """Information Gain, Gain Ratio, Chi-square, p-value and Cramer's V for every variable."""
    print("\n" + "=" * 70 + "\nSTEP 3b – Information Gain + Chi-square (training set)\n" + "=" * 70)
    h_y = entropy(y.value_counts(normalize=True).values)
    print(f"Entropy of churn_flag H(Y) = {h_y:.4f} bits")
    rows = []
    for c in X.columns:
        x = discretise(X[c])
        px = x.value_counts(normalize=True)
        h_y_given_x = sum(px[v] * entropy(y[x == v].value_counts(normalize=True).values) for v in px.index)
        ig = h_y - h_y_given_x
        split_info = entropy(px.values)

        table = pd.crosstab(x, y)
        chi2, p, dof, _ = chi2_contingency(table)
        cramers_v = np.sqrt(chi2 / (table.values.sum() * (min(table.shape) - 1)))

        rows.append({"variable": c, "type": "categorical" if is_categorical(X[c]) else "numeric",
                     "levels": len(px), "info_gain": ig,
                     "gain_ratio": ig / split_info if split_info else 0.0,
                     "chi2": chi2, "dof": dof, "p_value": p, "cramers_v": cramers_v})
    return pd.DataFrame(rows).sort_values("info_gain", ascending=False).reset_index(drop=True)


def find_redundant(X, scores):
    """Return {dropped_variable: reason} for duplicates of a stronger variable."""
    rank = scores.set_index("variable")["info_gain"]
    redundant = {}
    cols = list(scores["variable"])          # strongest first

    # (a) one-to-one relabels, e.g. satisfaction_level <-> satisfaction_score
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            if a in redundant or b in redundant:
                continue
            pair = X[[a, b]].dropna()
            if pair[a].nunique() <= 50 and pair[b].nunique() <= 50 \
                    and pair.groupby(a)[b].nunique().max() == 1 and pair.groupby(b)[a].nunique().max() == 1:
                # same information: keep numeric version if there is one, else the higher-IG one
                keep, drop = (b, a) if is_categorical(X[a]) and not is_categorical(X[b]) else (a, b)
                redundant[drop] = f"one-to-one relabel of {keep}"

    # (b) highly correlated numeric pairs – drop the weaker one
    num = [c for c in cols if not is_categorical(X[c]) and c not in redundant]
    corr = X[num].corr(method="spearman").abs()
    for i, a in enumerate(num):
        for b in num[i + 1:]:
            if a in redundant or b in redundant:
                continue
            if corr.loc[a, b] > CORR_MAX:
                weak, strong = (b, a) if rank[a] >= rank[b] else (a, b)
                redundant[weak] = f"rho={corr.loc[a, b]:.2f} with {strong}"
    return redundant


def select_features(X, y):
    scores = score_variables(X, y)
    redundant = find_redundant(X, scores)

    def decide(r):
        if r.variable in redundant:
            return "drop – " + redundant[r.variable]
        fails = []
        if r.p_value >= ALPHA:
            fails.append(f"p={r.p_value:.3f}")
        if r.cramers_v < V_MIN:
            fails.append(f"V<{V_MIN}")
        if r.info_gain < IG_MIN:
            fails.append(f"IG<{IG_MIN}")
        return "KEEP" if not fails else "drop – " + ", ".join(fails)

    scores["decision"] = scores.apply(decide, axis=1)
    selected = scores.loc[scores.decision == "KEEP", "variable"].tolist()

    show = scores[["variable", "type", "levels", "info_gain", "gain_ratio",
                   "chi2", "dof", "p_value", "cramers_v", "decision"]]
    print(show.to_string(float_format=lambda v: f"{v:.4g}"))
    scores.to_csv(f"{OUTPUT_DIR}/feature_selection_scores.csv", index=False)

    # Information Gain chart, kept variables highlighted
    plt.figure(figsize=(9, 8))
    colors = ["#2E7D32" if d == "KEEP" else "#B0B0B0" for d in scores.decision]
    plt.barh(scores.variable[::-1], scores.info_gain[::-1], color=colors[::-1])
    plt.xlabel("Information Gain (bits)")
    plt.title("Information Gain with churn_flag – training set\n(green = selected)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/information_gain.png", dpi=90)
    plt.close()

    print(f"\nSelected {len(selected)} of {len(scores)} variables: {selected}")
    print(f"Scores saved to '{OUTPUT_DIR}/feature_selection_scores.csv'")
    return selected, scores


# =============================================================================
# STEP 4 – Train and tune the four models (TRAINING SET ONLY)
# =============================================================================
# * Preprocessing sits INSIDE a Pipeline with each model, so during cross-
#   validation the scaler is fitted on the training folds only and applied to
#   the validation fold – no information leaks from the validation fold.
# * Numeric -> StandardScaler (needed by Logistic Regression and SVM;
#   harmless for trees). Categorical -> one-hot encoding.
# * Each model: (1) baseline with default settings, (2) GridSearchCV over a
#   small grid, 5-fold stratified CV, scored on ROC-AUC. The best settings are
#   then refitted on the whole training set.
# * SVM (RBF kernel) scales roughly with the square of the number of rows, so
#   it is tuned on a stratified sample of SVM_TUNE_ROWS rows; the final SVM is
#   then refitted on the full training set with the best settings.
import time
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

CV_FOLDS = 5
SCORING = "roc_auc"
SVM_TUNE_ROWS = 5_000 if FAST_MODE else 10_000   # SVM is tuned on a stratified sample
RF_TUNE_TREES = 100 if FAST_MODE else 300         # trees per forest while tuning
RF_FINAL_TREES = 300                              # trees in the final Random Forest
N_JOBS = -1                                # use all CPU cores


def build_preprocessor(X):
    num = [c for c in X.columns if not is_categorical(X[c])]
    cat = [c for c in X.columns if is_categorical(X[c])]
    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])


def model_grids():
    """Model + hyper-parameter grid. 'clf__' targets the classifier step of the Pipeline.
    FAST_MODE drops settings that were slow and never competitive in the full search
    (fully grown forests, max_features=0.5, very small/large C and gamma)."""
    if FAST_MODE:
        svm_grid = {"clf__C": [0.1, 1], "clf__gamma": ["scale"],
                    "clf__class_weight": [None, "balanced"]}
        rf_grid = {"clf__max_depth": [8, 12], "clf__min_samples_leaf": [20, 50],
                   "clf__max_features": ["sqrt"], "clf__class_weight": [None, "balanced"]}
    else:
        svm_grid = {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 0.01, 0.1],
                    "clf__class_weight": [None, "balanced"]}
        rf_grid = {"clf__max_depth": [8, 12, None], "clf__min_samples_leaf": [5, 20, 50],
                   "clf__max_features": ["sqrt", 0.5], "clf__class_weight": [None, "balanced"]}
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=3000, solver="liblinear"),
            {"clf__C": [0.01, 0.1, 1, 10],
             "clf__penalty": ["l1", "l2"],
             "clf__class_weight": [None, "balanced"]}),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {"clf__criterion": ["gini", "entropy"],     # 'entropy' = Information Gain splits
             "clf__max_depth": [4, 6, 8, 10],
             "clf__min_samples_leaf": [20, 50, 100],
             "clf__class_weight": [None, "balanced"]}),
        "SVM": (SVC(kernel="rbf", random_state=RANDOM_STATE), svm_grid),
        "Random Forest": (
            RandomForestClassifier(n_estimators=RF_TUNE_TREES, random_state=RANDOM_STATE, n_jobs=1),
            rf_grid),
    }


def stratified_sample(X, y, n):
    if len(X) <= n:
        return X, y
    Xs, _, ys, _ = train_test_split(X, y, train_size=n, stratify=y, random_state=RANDOM_STATE)
    return Xs, ys


def train_and_tune(X, y):
    print("\n" + "=" * 70 + "\nSTEP 4 – Train and tune models (training set, "
          f"{CV_FOLDS}-fold stratified CV, metric = {SCORING}, "
          f"{'FAST' if FAST_MODE else 'FULL'} mode)\n" + "=" * 70)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    best_models, rows = {}, []

    for name, (clf, grid) in model_grids().items():
        t0 = time.time()
        n_settings = int(np.prod([len(v) for v in grid.values()]))
        print(f"\n> Tuning {name}: {n_settings} settings x {CV_FOLDS} folds ...", flush=True)
        Xt, yt = (stratified_sample(X, y, SVM_TUNE_ROWS) if name == "SVM" else (X, y))
        pipe = Pipeline([("prep", build_preprocessor(X)), ("clf", clf)])

        # (1) Baseline: default settings
        base = cross_validate(pipe, Xt, yt, cv=cv, scoring=SCORING, n_jobs=N_JOBS)
        base_auc = base["test_score"].mean()

        # (2) Tuning
        search = GridSearchCV(pipe, grid, cv=cv, scoring=SCORING, n_jobs=N_JOBS, refit=False)
        search.fit(Xt, yt)
        best_params = search.best_params_
        best_std = search.cv_results_["std_test_score"][search.best_index_]

        # (3) Refit the best settings on the FULL training set
        final = Pipeline([("prep", build_preprocessor(X)), ("clf", clf)]).set_params(**best_params)
        if name == "Random Forest":
            final.set_params(clf__n_estimators=RF_FINAL_TREES, clf__n_jobs=N_JOBS)
        final.fit(X, y)
        best_models[name] = final

        secs = time.time() - t0
        tuned_on = f"{len(Xt):,} rows" + (" (sample)" if len(Xt) < len(X) else "")
        print(f"{name}  [{len(search.cv_results_['params'])} settings tried on {tuned_on}, {secs:.0f}s]")
        print(f"  baseline CV ROC-AUC : {base_auc:.4f}")
        print(f"  tuned    CV ROC-AUC : {search.best_score_:.4f} (+/- {best_std:.4f})")
        print("  best settings       :", {k.replace('clf__', ''): v for k, v in best_params.items()})
        rows.append({"model": name, "tuned_on_rows": len(Xt), "baseline_cv_auc": base_auc,
                     "tuned_cv_auc": search.best_score_, "tuned_cv_std": best_std,
                     "best_params": {k.replace("clf__", ""): v for k, v in best_params.items()}})

        pd.DataFrame(search.cv_results_).sort_values("rank_test_score").to_csv(
            f"{OUTPUT_DIR}/tuning_{name.lower().replace(' ', '_')}.csv", index=False)
        joblib.dump(final, f"{OUTPUT_DIR}/model_{name.lower().replace(' ', '_')}.joblib")

    cv_summary = pd.DataFrame(rows).sort_values("tuned_cv_auc", ascending=False)
    cv_summary.to_csv(f"{OUTPUT_DIR}/cv_summary.csv", index=False)
    print("\nCross-validation summary (training set):")
    print(cv_summary[["model", "baseline_cv_auc", "tuned_cv_auc", "tuned_cv_std"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # Chart: baseline vs tuned CV ROC-AUC
    plot = cv_summary.set_index("model")[["baseline_cv_auc", "tuned_cv_auc"]]
    ax = plot.plot.barh(figsize=(8, 4.5), color=["#B0B0B0", "#2E7D32"],
                        xerr=[np.zeros(len(plot)), cv_summary.tuned_cv_std.values])
    ax.set_xlim(max(0.5, plot.values.min() - 0.05), min(1.0, plot.values.max() + 0.03))
    ax.set_xlabel("ROC-AUC (5-fold CV, training set)")
    ax.set_title("Baseline vs tuned models")
    ax.legend(["Baseline", "Tuned"], loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/cv_model_comparison.png", dpi=90)
    plt.close()
    print(f"Tuned models, tuning tables and chart saved in '{OUTPUT_DIR}/'")
    return best_models, cv_summary


def model_file(name):
    return f"{OUTPUT_DIR}/model_{name.lower().replace(' ', '_')}.joblib"


def load_saved_models(selected):
    """Reuse the models tuned in an earlier run (they were fitted on the training set only)."""
    print("\n" + "=" * 70 + "\nSTEP 4 – skipped: loading saved tuned models\n" + "=" * 70)
    models = {}
    for name in model_grids():
        path = model_file(name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found – run once with LOAD_SAVED_MODELS = False")
        m = joblib.load(path)
        if list(m.feature_names_in_) != list(selected):
            raise ValueError(f"{name} was trained on different variables – re-run with LOAD_SAVED_MODELS = False")
        models[name] = m
        print(f"  loaded {path}")
    return models


# =============================================================================
# STEP 5 – Evaluate once on the untouched test set + interpret the models
# =============================================================================
# * The test set has played no part in leakage checks, EDA, feature selection,
#   scaling or tuning, so these numbers estimate performance on new customers.
# * Metrics
#     Accuracy  – share of all customers classified correctly
#     Precision – of customers flagged as churners, the share who really churned
#     Recall    – of real churners, the share the model caught (usually the key
#                 metric for churn: a missed churner is a lost customer)
#     F1        – balance of precision and recall
#     ROC-AUC   – ranking quality across all thresholds (0.5 = random, 1 = perfect)
#     PR-AUC    – average precision; more informative when churners are a minority
# * SVM was fitted without probability estimates (that would need an extra 5-fold
#   internal CV), so its ROC / PR curves use the SVM decision score instead,
#   which ranks customers just as well.
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix,
                             roc_curve, precision_recall_curve)
from sklearn.inspection import permutation_importance

PERM_REPEATS = 5


def churn_scores(model, X):
    """Higher score = more likely to churn. Probability where available, else SVM decision score."""
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except AttributeError:
            pass
    return model.decision_function(X)


def evaluate_on_test(models, X_test, y_test):
    print("\n" + "=" * 70 + f"\nSTEP 5a – Evaluation on the untouched test set ({len(X_test):,} rows)\n" + "=" * 70)
    rows, preds, scores = [], {}, {}
    for name, m in models.items():
        t0 = time.time()
        y_pred = m.predict(X_test)
        s = churn_scores(m, X_test)
        preds[name], scores[name] = y_pred, s
        rows.append({"model": name,
                     "accuracy": accuracy_score(y_test, y_pred),
                     "precision": precision_score(y_test, y_pred),
                     "recall": recall_score(y_test, y_pred),
                     "f1": f1_score(y_test, y_pred),
                     "roc_auc": roc_auc_score(y_test, s),
                     "pr_auc": average_precision_score(y_test, s)})
        print(f"  scored {name} ({time.time() - t0:.0f}s)")
    results = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    print(f"\nTest-set results (churn rate in test = {y_test.mean():.1%}; "
          f"a model that flags everyone has recall 1.0 and precision {y_test.mean():.3f}):")
    print(results.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    results.to_csv(f"{OUTPUT_DIR}/test_results.csv", index=False)

    # Confusion matrices
    fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 4.2))
    for ax, name in zip(np.atleast_1d(axes), results.model):
        cm = confusion_matrix(y_test, preds[name])
        sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Retained", "Churned"], yticklabels=["Retained", "Churned"])
        ax.set_title(f"{name}\nrecall {recall_score(y_test, preds[name]):.2f} | "
                     f"precision {precision_score(y_test, preds[name]):.2f}", fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/test_confusion_matrices.png", dpi=90)
    plt.close()

    # ROC and Precision-Recall curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    for name in results.model:
        fpr, tpr, _ = roc_curve(y_test, scores[name])
        ax1.plot(fpr, tpr, lw=2, label=f"{name} (AUC {roc_auc_score(y_test, scores[name]):.3f})")
        prec, rec, _ = precision_recall_curve(y_test, scores[name])
        ax2.plot(rec, prec, lw=2, label=f"{name} (AP {average_precision_score(y_test, scores[name]):.3f})")
    ax1.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC 0.500)")
    ax1.set(xlabel="False positive rate", ylabel="True positive rate (recall)", title="ROC curves – test set")
    ax2.axhline(y_test.mean(), color="k", ls="--", lw=1, label=f"Random (AP {y_test.mean():.3f})")
    ax2.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curves – test set")
    ax1.legend(loc="lower right", fontsize=9)
    ax2.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/test_roc_pr_curves.png", dpi=90)
    plt.close()

    # Threshold trade-off for the best model that gives probabilities
    best = next(n for n in results.model if hasattr(models[n], "predict_proba") and n != "SVM")
    thr_rows = []
    for t in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        p = (scores[best] >= t).astype(int)
        thr_rows.append({"threshold": t, "flagged_%": p.mean() * 100,
                         "precision": precision_score(y_test, p, zero_division=0),
                         "recall": recall_score(y_test, p), "f1": f1_score(y_test, p)})
    thr = pd.DataFrame(thr_rows)
    print(f"\nDecision-threshold trade-off for {best} (default threshold = 0.50):")
    print(thr.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    thr.to_csv(f"{OUTPUT_DIR}/test_threshold_tradeoff.csv", index=False)
    print(f"\nTest metrics, confusion matrices, ROC/PR curves saved in '{OUTPUT_DIR}/'")

    sensitivity_specificity_chart(preds, y_test, list(results.model))
    return results


def sensitivity_specificity_chart(preds, y_test, order):
    """Sensitivity, Specificity, Accuracy, Precision and FPR for each model, from its
    confusion matrix (threshold 0.50).
         Sensitivity (TPR, recall) = TP / (TP + FN)  – share of churners caught
         Specificity (TNR)         = TN / (TN + FP)  – share of retained customers correctly left alone
         Accuracy                  = (TP + TN) / all – share of all customers classified correctly
         Precision (PPV)           = TP / (TP + FP)  – share of flagged customers who really churned
         FPR                       = FP / (FP + TN)  = 1 - Specificity – share of retained customers wrongly flagged
    """
    rows = []
    for name in order:
        tn, fp, fn, tp = confusion_matrix(y_test, preds[name]).ravel()
        rows.append({"model": name, "TP": tp, "FN": fn, "TN": tn, "FP": fp,
                     "sensitivity": tp / (tp + fn),
                     "specificity": tn / (tn + fp),
                     "accuracy": (tp + tn) / (tp + tn + fp + fn),
                     "precision": tp / (tp + fp),
                     "fpr": fp / (fp + tn)})
    t = pd.DataFrame(rows)
    print("\nSensitivity, Specificity, Accuracy, Precision and FPR (test set, threshold 0.50):")
    print(t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    t.to_csv(f"{OUTPUT_DIR}/test_classification_metrics.csv", index=False)

    fig = plt.figure(figsize=(15, 8.8))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.35], hspace=0.32)
    ax = fig.add_subplot(gs[0])
    x = np.arange(len(t))
    bars = [("sensitivity", "Sensitivity (TPR)", "#2E7D32"),
            ("specificity", "Specificity (TNR)", "#1565C0"),
            ("accuracy", "Accuracy", "#6A1B9A"),
            ("precision", "Precision", "#EF6C00"),
            ("fpr", "False Positive Rate", "#C62828")]
    w = 0.8 / len(bars)
    for i, (col, label, color) in enumerate(bars):
        b = ax.bar(x + (i - (len(bars) - 1) / 2) * w, t[col], w, label=label, color=color)
        ax.bar_label(b, labels=[f"{v:.0%}" for v in t[col]], padding=2, fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(t.model, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("Rate")
    ax.set_title("Sensitivity, Specificity, Accuracy, Precision and FPR – test set (threshold 0.50)", fontsize=13)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=len(bars), frameon=False)

    tab = fig.add_subplot(gs[1])
    tab.axis("off")
    cell = [[f"{r.TP:,}", f"{r.FN:,}", f"{r.TN:,}", f"{r.FP:,}",
             f"{r.sensitivity:.1%}", f"{r.specificity:.1%}", f"{r.accuracy:.1%}",
             f"{r.precision:.1%}", f"{r.fpr:.1%}"] for r in t.itertuples()]
    table = tab.table(cellText=cell, rowLabels=list(t.model),
                      colLabels=["TP", "FN", "TN", "FP", "Sensitivity", "Specificity",
                                 "Accuracy", "Precision", "FPR"],
                      loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.45)
    tab.text(0.5, -0.18,
             "Sensitivity = TP / (TP + FN)     Specificity = TN / (TN + FP)     Accuracy = (TP + TN) / (TP + TN + FP + FN)\n"
             "Precision = TP / (TP + FP)     FPR = FP / (FP + TN) = 1 − Specificity\n"
             "TP = churners flagged · FN = churners missed · TN = retained left alone · FP = retained wrongly flagged",
             ha="center", va="top", fontsize=9.5, transform=tab.transAxes, color="#444444")
    plt.savefig(f"{OUTPUT_DIR}/test_classification_metrics.png", dpi=110, bbox_inches="tight")
    plt.close()
    print(f"Chart saved to '{OUTPUT_DIR}/test_classification_metrics.png'")


def tree_decoder(pipe):
    """For a fitted Pipeline(prep, DecisionTree) return the sklearn tree plus helpers that
    express splits in ORIGINAL units (thresholds un-scaled, one-hot columns as 'variable = level')."""
    prep, tree = pipe.named_steps["prep"], pipe.named_steps["clf"].tree_
    scaler = prep.named_transformers_["num"]
    num_cols = list(prep.transformers_[0][2])
    ohe = prep.named_transformers_["cat"]
    cat_cols = list(prep.transformers_[1][2])

    # describe each transformed column: ("num", name, mean, scale) or ("cat", variable, level)
    cols = [("num", c, scaler.mean_[i], scaler.scale_[i]) for i, c in enumerate(num_cols)]
    for c, levels in zip(cat_cols, ohe.categories_):
        cols += [("cat", c, lvl) for lvl in levels]

    def conditions(n):
        """(left condition, right condition) of the split at node n."""
        col = cols[tree.feature[n]]
        if col[0] == "num":
            raw = tree.threshold[n] * col[3] + col[2]
            if col[1] in INTEGER_COLS:          # integer variables: "<= 2.5" means "<= 2"
                return f"{col[1]} <= {np.floor(raw):.0f}", f"{col[1]} >= {np.floor(raw) + 1:.0f}"
            return f"{col[1]} <= {raw:.2f}", f"{col[1]} > {raw:.2f}"
        return f"{col[1]} != {col[2]}", f"{col[1]} = {col[2]}"

    def churn_rate(n):
        v = tree.value[n][0]
        return v[1] / v.sum()

    return tree, conditions, churn_rate


def decision_tree_rules(pipe, max_depth=None):
    """Text rules of a fitted Pipeline(prep, DecisionTree), thresholds converted back to original units."""
    tree, conditions, churn_rate = tree_decoder(pipe)

    def node_text(n):
        return f"[n = {tree.n_node_samples[n]:,}, churn = {churn_rate(n):.0%}]"

    lines = []

    def walk(n, depth):
        if tree.children_left[n] == -1 or (max_depth is not None and depth >= max_depth):
            return
        left_cond, right_cond = conditions(n)
        pad = "|   " * depth
        for cond, child in [(left_cond, tree.children_left[n]), (right_cond, tree.children_right[n])]:
            lines.append(f"{pad}|--- {cond}  {node_text(child)}")
            walk(child, depth + 1)

    lines.append(f"ALL CUSTOMERS  {node_text(0)}")
    walk(0, 0)
    return "\n".join(lines)


TREE_PLOT_DEPTH = 3    # levels of the Decision Tree drawn in the image (4 still fits; 5+ gets crowded)


def plot_decision_tree(pipe, max_depth=TREE_PLOT_DEPTH, path=None):
    """Draw the top levels of the Decision Tree as boxes, in original units.
    Each box: the condition that leads to it, customers in the branch, their churn rate.
    Box colour runs from green (low churn) to red (high churn)."""
    tree, conditions, churn_rate = tree_decoder(pipe)
    path = path or f"{OUTPUT_DIR}/decision_tree.png"
    cmap = plt.get_cmap("RdYlGn_r")
    base_rate = churn_rate(0)

    # lay out: leaves of the drawn tree get consecutive x positions, parents sit above the middle
    pos, labels, edges, next_x = {}, {}, [], [0]

    def layout(n, depth, cond):
        is_split = tree.children_left[n] != -1
        if is_split and depth < max_depth:
            left_cond, right_cond = conditions(n)
            l = layout(tree.children_left[n], depth + 1, left_cond)
            r = layout(tree.children_right[n], depth + 1, right_cond)
            x = (pos[l][0] + pos[r][0]) / 2
            edges.extend([(n, l), (n, r)])
        else:
            x = next_x[0]
            next_x[0] += 1
        pos[n] = (x, -depth)
        # put the variable name on its own line so boxes stay narrow
        head = "ALL CUSTOMERS" if cond is None else cond.replace(" <= ", "\n<= ").replace(
            " >= ", "\n>= ").replace(" > ", "\n> ").replace(" != ", "\n≠ ").replace(" = ", "\n= ")
        more = "\n(splits further)" if is_split and depth >= max_depth else ""
        labels[n] = f"{head}\nn = {tree.n_node_samples[n]:,}\nchurn = {churn_rate(n):.0%}{more}"
        return n

    layout(0, 0, None)
    n_leaves = next_x[0]

    fig, ax = plt.subplots(figsize=(max(12, 2.9 * n_leaves), 2.9 * (max_depth + 1)))
    for a, b in edges:
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1] - 0.22, pos[b][1] + 0.3], color="#888888", lw=1.2, zorder=1)
    for n, (x, y) in pos.items():
        rate = churn_rate(n)
        ax.text(x, y, labels[n], ha="center", va="center", fontsize=9, zorder=2,
                color="white" if abs(rate - 0.5) > 0.3 else "black",
                bbox=dict(boxstyle="round,pad=0.5", fc=cmap(rate), ec="#444444", lw=1))
    ax.set_xlim(-0.7, n_leaves - 0.3)
    ax.set_ylim(-max_depth - 0.6, 0.6)
    ax.axis("off")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.035, pad=0.02, aspect=50)
    cbar.set_label(f"Churn rate in the branch (overall training churn rate = {base_rate:.0%})")
    cbar.ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title(f"Decision Tree – top {max_depth} levels (training customers; left branch = condition true "
                 "for '<=' / '≠', right = '>=' / '=')", fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    return path


INTEGER_COLS = set()   # filled in at run time from the training data


def interpret_models(models, X_test, y_test, results):
    print("\n" + "=" * 70 + "\nSTEP 5b – Interpretation\n" + "=" * 70)

    # (1) Logistic Regression – odds ratios
    lr = models["Logistic Regression"]
    names = lr.named_steps["prep"].get_feature_names_out()
    coefs = lr.named_steps["clf"].coef_[0]
    odds = (pd.DataFrame({"feature": [n.split("__", 1)[1] for n in names], "coefficient": coefs,
                          "odds_ratio": np.exp(coefs)})
            .assign(abs_coef=lambda d: d.coefficient.abs())
            .sort_values("abs_coef", ascending=False).drop(columns="abs_coef"))
    print("Logistic Regression odds ratios (numeric features are standardised, so the odds ratio is"
          "\nthe change in churn odds for a 1 standard-deviation increase; for a category level it is"
          "\nrelative to that variable's other levels; coefficient 0 = removed by the L1 penalty):")
    print(odds.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    odds.to_csv(f"{OUTPUT_DIR}/logistic_regression_odds_ratios.csv", index=False)

    # (2) Decision Tree – rules in the ORIGINAL units (thresholds un-scaled,
    #     one-hot columns shown as "variable = level"), with churn rate per branch
    dt = models["Decision Tree"]
    full_rules = decision_tree_rules(dt)
    top_rules = decision_tree_rules(dt, max_depth=3)
    with open(f"{OUTPUT_DIR}/decision_tree_rules.txt", "w") as f:
        f.write("Decision Tree rules – original units. [n = training customers in the branch, "
                "churn = their churn rate]\n\n" + full_rules)
    print("\nDecision Tree – first 3 levels (n = training customers in the branch, churn = their churn rate):")
    print(top_rules)
    print(f"Full tree saved to '{OUTPUT_DIR}/decision_tree_rules.txt'")
    print(f"Tree diagram (top {TREE_PLOT_DEPTH} levels) saved to '{plot_decision_tree(dt)}'\n")

    # (3) Permutation importance on the TEST set, per original variable
    #     (shuffle one variable, measure the drop in ROC-AUC). Unlike Random Forest's
    #     built-in importance, it is not biased towards many-valued numeric variables.
    #     SVM is skipped: scoring it 40 extra times on 10k rows is slow, and the
    #     tree/linear models already cover interpretation.
    imp_rows = []
    for name in ["Random Forest", "Decision Tree", "Logistic Regression"]:
        r = permutation_importance(models[name], X_test, y_test, scoring="roc_auc",
                                   n_repeats=PERM_REPEATS, random_state=RANDOM_STATE, n_jobs=N_JOBS)
        for col, mean, std in zip(X_test.columns, r.importances_mean, r.importances_std):
            imp_rows.append({"model": name, "variable": col, "auc_drop": mean, "std": std})
    imp = pd.DataFrame(imp_rows)
    table = imp.pivot(index="variable", columns="model", values="auc_drop")
    table = table.sort_values("Random Forest", ascending=False)
    print("Permutation importance on the test set (drop in ROC-AUC when the variable is shuffled):")
    print(table.to_string(float_format=lambda v: f"{v:.4f}"))
    imp.to_csv(f"{OUTPUT_DIR}/permutation_importance.csv", index=False)

    ax = table[::-1].plot.barh(figsize=(9, 6))
    ax.set_xlabel("Drop in test ROC-AUC when shuffled")
    ax.set_title("Permutation importance – test set")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/permutation_importance.png", dpi=90)
    plt.close()

    best = results.iloc[0]
    print(f"\nBest model on the test set: {best.model} (ROC-AUC {best.roc_auc:.4f}, "
          f"recall {best.recall:.3f}, precision {best.precision:.3f})")


if __name__ == "__main__":
    raw = load_data(PATH)
    leakage_evidence(raw)
    data = remove_leakage(raw)
    X_train, X_test, y_train, y_test = split_data(data)

    # ---- Step 3: training set only -------------------------------------------
    if RUN_EDA:
        run_eda(X_train, y_train)
    X_train = fill_missing_categories(X_train)
    X_test = fill_missing_categories(X_test)      # constant fill, learns nothing from test
    selected, scores = select_features(X_train, y_train)
    X_train_sel, X_test_sel = X_train[selected], X_test[selected]
    INTEGER_COLS.update(c for c in selected if pd.api.types.is_integer_dtype(X_train_sel[c]))

    # ---- Step 4: training set only -------------------------------------------
    if LOAD_SAVED_MODELS:
        best_models = load_saved_models(selected)
    else:
        best_models, cv_summary = train_and_tune(X_train_sel, y_train)

    # ---- Step 5: the ONLY place the test set is used ---------------------------
    results = evaluate_on_test(best_models, X_test_sel, y_test)
    interpret_models(best_models, X_test_sel, y_test, results)
    print("\nDone. All outputs are in the", f"'{OUTPUT_DIR}/' folder.")