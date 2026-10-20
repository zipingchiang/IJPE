import os
import random
import warnings
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend: save figures without opening GUI windows.
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit, OptimizeWarning

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree

try:
    from gplearn.genetic import SymbolicRegressor
    GPLEARN_AVAILABLE = True
except ImportError:
    SymbolicRegressor = None
    GPLEARN_AVAILABLE = False

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Global settings
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

FILE_DATA = "./content/Kinmen_PCI.csv"

# All figures are saved automatically; no GUI windows are opened.
FIGURE_DIR = "./content/figures"
os.makedirs(FIGURE_DIR, exist_ok=True)

def auto_figure_path(title, suffix=".png"):
    """Create a filesystem-safe output path for plots that have no explicit save path."""
    safe = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in str(title))
    safe = "_".join(part for part in safe.split("_") if part)
    if not safe:
        safe = "figure"
    return os.path.join(FIGURE_DIR, safe + suffix)

# Five physical predictors used by the revised models.
# D is used only to separate pavement groups and in the pooled nonlinear model.
FEATURE_COLS = ["Mr", "SN", "ALT", "ESALs", "AGE"]
TARGET_COL = "PCI"
RAW_FEATURE_COLS = FEATURE_COLS.copy()
GROUP_COL = "D"

# Shared hold-out protocol for the machine-learning / XAI comparisons.
# The same deterministic 70:30 train:test split (SEED=42) is used for
# Direct NN, transfer learning, symbolic regression and decision tree models.
TRAIN_RATIO = 0.70
TEST_RATIO = 0.30

# Symbolic-regression interpretability controls.
# The final reported symbolic equation is selected using TRAINING data only
# and is guaranteed not to exceed SYMBOLIC_MAX_NODES nodes. The 30% test set
# remains untouched until the final performance evaluation.
SYMBOLIC_MAX_NODES = 30
SYMBOLIC_PARSIMONY = 0.03
SYMBOLIC_FUNCTION_SET = ("add", "sub", "mul", "div")


# ============================================================
# Shared utilities
# ============================================================

def ensure_input_files_exist():
    if not os.path.exists(FILE_DATA):
        raise FileNotFoundError(
            f"Missing required input file: {FILE_DATA}\n"
            "Please place Kinmen_PCI.csv in ./content before running the script."
        )


def load_and_clean_csv(file_path, dropna_for_model=False):
    """Load the unified dataset.

    Accepted columns: Mr, SN, ALT, ESAL (or ESALs), AGE, D, PCI.
    AGE may be missing in the raw descriptive dataset. For modelling, rows with
    missing AGE are excluded rather than imputed.
    """
    df = pd.read_csv(file_path)
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")].copy()

    if "ESAL" in df.columns and "ESALs" not in df.columns:
        df = df.rename(columns={"ESAL": "ESALs"})

    required = ["Mr", "SN", "ALT", "ESALs", "AGE", "D", "PCI"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{file_path} is missing required columns {missing}. "
            f"Current columns: {list(df.columns)}"
        )

    df = df[required].copy()
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    invalid_d = df.loc[df["D"].notna() & ~df["D"].isin([0, 1]), "D"]
    if len(invalid_d):
        raise ValueError(f"D must contain only 0/1. Invalid values: {sorted(invalid_d.unique())}")

    if dropna_for_model:
        # No AGE imputation: use complete cases for Age-inclusive modelling.
        df = df.dropna(subset=FEATURE_COLS + [TARGET_COL, GROUP_COL]).reset_index(drop=True)

    return df


def descriptive_stats(df):
    stats = pd.DataFrame(
        {
            "Count": df.count(),
            "Mean": df.mean(),
            "Std. Dev.": df.std(ddof=1),
            "Min.": df.min(),
            "Max.": df.max(),
        }
    ).T
    return stats


def format_stats_table(stats):
    out = stats.copy()
    if "Count" in out.index:
        out.loc["Count"] = out.loc["Count"].astype(int)

    for row in ["Mean", "Std. Dev.", "Min.", "Max."]:
        if row in out.index:
            out.loc[row] = out.loc[row].astype(float)
    return out


def calc_metrics(y_true, y_pred):
    residuals = y_true - y_pred
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    metrics = {
        "R2": float(r2),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "Residual_Mean": float(np.mean(residuals)),
        "Residual_STD": float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0,
        "Residual_Min": float(np.min(residuals)),
        "Residual_Max": float(np.max(residuals)),
    }
    return metrics, residuals



def print_metrics(metrics, title="Metrics"):
    print(f"\n{title}")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")


def save_prediction_csv(y_true, y_pred, out_path):
    pred_df = pd.DataFrame(
        {
            "Observed_PCI": y_true,
            "Predicted_PCI": y_pred,
            "Residual": y_true - y_pred,
        }
    )
    pred_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


def plot_pred_vs_true(y_true, y_pred, title, save_path=None):
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, alpha=0.8)
    mn = min(np.min(y_true), np.min(y_pred))
    mx = max(np.max(y_true), np.max(y_pred))
    plt.plot([mn, mx], [mn, mx], "k--")
    plt.xlabel("Observed PCI")
    plt.ylabel("Predicted PCI")
    plt.title(title)
    plt.tight_layout()
    if save_path is None:
        save_path = auto_figure_path(title)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def plot_residuals(y_true, y_pred, title, save_path=None):
    residuals = y_true - y_pred
    plt.figure(figsize=(5, 4))
    plt.scatter(y_pred, residuals, alpha=0.8)
    plt.axhline(0, color="k", linestyle="--")
    plt.xlabel("Predicted PCI")
    plt.ylabel("Residual")
    plt.title(title)
    plt.tight_layout()
    if save_path is None:
        save_path = auto_figure_path(title)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def split_df_train_test(df, test_size=TEST_RATIO, random_state=SEED):
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    return train_df, test_df


# ============================================================
# Model explanation diagram utilities
# ============================================================

MODEL_DIAGRAM_DIR = "./content/model_diagrams"


def ensure_model_diagram_dir():
    os.makedirs(MODEL_DIAGRAM_DIR, exist_ok=True)


def safe_filename(text):
    return (
        str(text)
        .replace(" ", "")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "")
        .replace("→", "to")
        .replace("-", "_")
    )


def save_workflow_diagram(steps, title, out_path, figsize=(14, 3.2)):
    """Save a simple left-to-right workflow diagram."""
    ensure_model_diagram_dir()

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    n = len(steps)
    xs = np.linspace(0.08, 0.92, n)
    y = 0.50

    for i, (x, step) in enumerate(zip(xs, steps)):
        ax.text(
            x,
            y,
            textwrap.fill(step, width=18),
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="black"),
        )

        if i < n - 1:
            ax.annotate(
                "",
                xy=(xs[i + 1] - 0.065, y),
                xytext=(x + 0.065, y),
                arrowprops=dict(arrowstyle="->", linewidth=1.2),
            )

    ax.set_title(title, fontsize=12, pad=15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def save_residual_block_diagram(out_path=None):
    """Save a schematic of the residual 1D convolution block."""
    ensure_model_diagram_dir()

    if out_path is None:
        out_path = os.path.join(MODEL_DIAGRAM_DIR, "residual_block_1d.png")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")

    steps = [
        "Input",
        "Conv1D",
        "BatchNorm1D",
        "ReLU",
        "Dropout",
        "Conv1D",
        "BatchNorm1D",
        "Add skip connection",
        "ReLU",
        "Output",
    ]

    xs = np.linspace(0.06, 0.94, len(steps))
    y = 0.50

    for i, (x, step) in enumerate(zip(xs, steps)):
        ax.text(
            x,
            y,
            textwrap.fill(step, width=13),
            ha="center",
            va="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="black"),
        )

        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(xs[i + 1] - 0.035, y),
                xytext=(x + 0.035, y),
                arrowprops=dict(arrowstyle="->", linewidth=1.1),
            )

    ax.annotate(
        "",
        xy=(xs[7], y + 0.12),
        xytext=(xs[0], y + 0.12),
        arrowprops=dict(
            arrowstyle="->",
            linewidth=1.2,
            connectionstyle="arc3,rad=-0.25",
        ),
    )
    ax.text(
        (xs[0] + xs[7]) / 2,
        y + 0.25,
        "Identity skip connection",
        ha="center",
        va="center",
        fontsize=9,
    )

    ax.set_title("Residual 1D convolution block", fontsize=12, pad=15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def save_resnet_1dcnn_architecture(input_len, out_path=None):
    """Save a schematic of the ResNet-1D-CNN architecture used for PCI prediction."""
    ensure_model_diagram_dir()

    if out_path is None:
        out_path = os.path.join(MODEL_DIAGRAM_DIR, "resnet_1dcnn_architecture.png")

    steps = [
        "Observed variables\nMr, SN, ALT, ESALs, AGE",
        "Second-order\npolynomial features",
        "Standardization",
        f"Input tensor\n1 x {input_len}",
        "Conv1D + BatchNorm + ReLU",
        "Residual block 1",
        "Residual block 2",
        "Flatten",
        "Fully connected layers\n64 -> 32 -> 1",
        "Predicted PCI",
    ]

    save_workflow_diagram(
        steps=steps,
        title="ResNet-1D-CNN architecture for PCI prediction",
        out_path=out_path,
        figsize=(18, 3.6),
    )


def save_transfer_learning_workflow(out_path=None):
    """Save the transfer learning workflow from Type 1 to Type 2 pavements."""
    ensure_model_diagram_dir()

    if out_path is None:
        out_path = os.path.join(MODEL_DIAGRAM_DIR, "transfer_learning_workflow.png")

    steps = [
        "Type 1 observed data\nwithout SAMI",
        "Feature transformation\nand standardization",
        "Pre-train ResNet-1D-CNN",
        "Transfer learned weights",
        "Type 2 observed data\nwith SAMI",
        "Fine-tune model",
        "Evaluate on held-out\nType 2 test set",
    ]

    save_workflow_diagram(
        steps=steps,
        title="Transfer learning workflow for data-scarce Type 2 pavement prediction",
        out_path=out_path,
        figsize=(15, 3.5),
    )


def save_symbolic_regression_workflow(out_path=None):
    """Save a conceptual workflow diagram of symbolic regression."""
    ensure_model_diagram_dir()

    if out_path is None:
        out_path = os.path.join(MODEL_DIAGRAM_DIR, "symbolic_regression_workflow.png")

    steps = [
        "Scaled observed inputs\nMr, SN, ALT, ESALs, AGE",
        "Initial population\nof equations",
        "Fitness evaluation\nby MSE",
        "Genetic operations\ncrossover and mutation",
        "Best symbolic expression",
        "Interpretable PCI prediction",
    ]

    save_workflow_diagram(
        steps=steps,
        title="Symbolic regression workflow",
        out_path=out_path,
        figsize=(14, 3.5),
    )


def save_decision_tree_workflow(out_path=None):
    """Save a conceptual workflow diagram of decision-tree regression."""
    ensure_model_diagram_dir()

    if out_path is None:
        out_path = os.path.join(MODEL_DIAGRAM_DIR, "decision_tree_workflow.png")

    steps = [
        "Scaled observed inputs\nMr, SN, ALT, ESALs, AGE",
        "Candidate tree models",
        "GridSearchCV\nhyperparameter tuning",
        "Best decision tree",
        "Rule extraction",
        "Interpretable PCI prediction",
    ]

    save_workflow_diagram(
        steps=steps,
        title="Decision-tree regression workflow",
        out_path=out_path,
        figsize=(14, 3.5),
    )


def save_text_figure(text, title, out_path, figsize=(13, 4)):
    """Save a text-based figure, useful for symbolic equations and tree rules."""
    ensure_model_diagram_dir()

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    wrapped_text = "\n".join(
        textwrap.wrap(str(text), width=120, replace_whitespace=False)
    )

    ax.text(
        0.01,
        0.98,
        wrapped_text,
        ha="left",
        va="top",
        fontsize=9,
        family="monospace",
    )

    ax.set_title(title, fontsize=12, pad=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def save_dataframe_as_table_png(df, title, out_path, figsize=None):
    """Save a pandas DataFrame as a table image."""
    ensure_model_diagram_dir()

    if figsize is None:
        figsize = (12, max(2.5, 0.45 * len(df) + 1.5))

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.25)

    ax.set_title(title, fontsize=12, pad=15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


# ============================================================
# PART A. DESCRIPTIVE STATISTICS
# ============================================================

def run_part_a():
    print("\n" + "=" * 100)
    print("PART A. DESCRIPTIVE STATISTICS")
    print("=" * 100)

    df_raw = load_and_clean_csv(FILE_DATA, dropna_for_model=False)
    df0_raw = df_raw[df_raw["D"] == 0].copy()
    df1_raw = df_raw[df_raw["D"] == 1].copy()

    print("\n=== Basic file check ===")
    print(f"{FILE_DATA}: shape = {df_raw.shape}")
    print(f"Without SAMI (D=0): n = {len(df0_raw)}")
    print(f"With SAMI (D=1): n = {len(df1_raw)}")

    print("\nMissing values in full dataset:")
    print(df_raw.isna().sum())
    print("\nMissing AGE by group:")
    print(df_raw.groupby("D")["AGE"].apply(lambda s: int(s.isna().sum())))

    stat_cols = ["Mr", "SN", "ALT", "ESALs", "AGE", "PCI"]
    stats0 = descriptive_stats(df0_raw[stat_cols])
    stats1 = descriptive_stats(df1_raw[stat_cols])

    print("\n=== Descriptive statistics: without SAMI ===")
    print(format_stats_table(stats0))
    print("\n=== Descriptive statistics: with SAMI ===")
    print(format_stats_table(stats1))

    combined = pd.DataFrame(index=["Count", "Mean", "Std. Dev.", "Min.", "Max."])
    for col in stat_cols:
        combined[("Without SAMI", col)] = stats0.loc[combined.index, col].values
    for col in stat_cols:
        combined[("With SAMI", col)] = stats1.loc[combined.index, col].values
    combined.columns = pd.MultiIndex.from_tuples(combined.columns)

    combined_round = combined.copy()
    for idx in ["Mean", "Std. Dev.", "Min.", "Max."]:
        for col in combined_round.columns:
            variable = col[1]
            digits = 0 if variable == "ESALs" else 2
            combined_round.loc[idx, col] = round(float(combined_round.loc[idx, col]), digits)

    print("\n=== Rounded table for manuscript comparison ===")
    print(combined_round)

    # Correlation matrix on rows with complete AGE.
    corr_cols = ["Mr", "SN", "ALT", "ESALs", "AGE", "PCI"]
    corr_complete = df_raw.dropna(subset=["AGE"])[corr_cols]
    pearson = corr_complete.corr(method="pearson")
    spearman = corr_complete.corr(method="spearman")
    print("\n=== Pearson correlation (AGE-complete rows) ===")
    print(pearson.round(4))
    print("\n=== Spearman correlation (AGE-complete rows) ===")
    print(spearman.round(4))

    # Verify the field-data rule supplied by the study: the same ESAL should map to the same AGE.
    tmp = df_raw.dropna(subset=["AGE"]).groupby("ESALs")["AGE"].nunique()
    conflicts = tmp[tmp > 1]
    print(f"\nSame-ESAL / different-AGE conflicts: {len(conflicts)}")

    dup_all = df_raw.duplicated().sum()
    print(f"Exact duplicate rows in unified dataset: {dup_all}")

    stats0.to_csv("./content/descriptive_stats_without_SAMI.csv")
    stats1.to_csv("./content/descriptive_stats_with_SAMI.csv")
    combined.to_csv("./content/descriptive_stats_combined.csv")
    combined_round.to_csv("./content/descriptive_stats_combined_rounded.csv")
    pearson.to_csv("./content/correlation_pearson.csv")
    spearman.to_csv("./content/correlation_spearman.csv")

    return df_raw


# ============================================================
# PART B. BASELINE NONLINEAR REGRESSION WITH AGE
#
# Three nonlinear AGE specifications are compared on the same complete cases:
#
# (1) No-Age comparator
#     PCI = 100 - 60*(ESALs/rho)^beta
#
# (2) Additive AGE term
#     PCI = 100 - 60*(ESALs/rho)^beta - p9*AGE
#
# (3) Multiplicative AGE modifier (NEW)
#     PCI = 100 - 60*(ESALs/rho)^beta * (AGE/AGE_REF)^p9
#
# AGE_REF = 1 year makes the AGE ratio dimensionless.  With the engineering
# constraint p9 >= 0, older pavement can only maintain or increase the
# deterioration amplitude.  p9 = 0 exactly reduces the multiplicative model
# to the No-Age comparator.
#
# rho  = p1 + p2*Mr + p3*SN + p4*ALT
# beta = p5*Mr^p6*SN^p7*ALT^p8
#
# In the pooled models, D is allowed to affect both rho and beta, following
# the reviewer suggestion.
# ============================================================

def pci_eq1_no_age(X, p1, p2, p3, p4, p5, p6, p7, p8):
    Mr, SN, ALT, ESALs = X
    rho = p1 + p2 * Mr + p3 * SN + p4 * ALT
    beta = p5 * (Mr ** p6) * (SN ** p7) * (ALT ** p8)
    rho = np.maximum(rho, 1e-12)
    beta = np.maximum(beta, 1e-12)
    return 100 - 60 * ((ESALs / rho) ** beta)


def pci_eq1_age(X, p1, p2, p3, p4, p5, p6, p7, p8, p9):
    Mr, SN, ALT, ESALs, AGE = X
    rho = p1 + p2 * Mr + p3 * SN + p4 * ALT
    beta = p5 * (Mr ** p6) * (SN ** p7) * (ALT ** p8)
    rho = np.maximum(rho, 1e-12)
    beta = np.maximum(beta, 1e-12)
    return 100 - 60 * ((ESALs / rho) ** beta) - p9 * AGE


def pci_eq1_age_dummy_both(X, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11):
    Mr, SN, ALT, ESALs, AGE, D = X
    rho = p1 + p2 * Mr + p3 * SN + p4 * ALT + p10 * D
    beta_scale = np.maximum(p5 + p11 * D, 1e-12)
    beta = beta_scale * (Mr ** p6) * (SN ** p7) * (ALT ** p8)
    rho = np.maximum(rho, 1e-12)
    beta = np.maximum(beta, 1e-12)
    return 100 - 60 * ((ESALs / rho) ** beta) - p9 * AGE


AGE_REF = 1.0  # year; used only to make the multiplicative AGE ratio dimensionless


def pci_eq1_age_multiplicative(X, p1, p2, p3, p4, p5, p6, p7, p8, p9):
    """AGE multiplies the deterioration amplitude: AGE_REF=1 year, p9>=0."""
    Mr, SN, ALT, ESALs, AGE = X
    rho = p1 + p2 * Mr + p3 * SN + p4 * ALT
    beta = p5 * (Mr ** p6) * (SN ** p7) * (ALT ** p8)
    rho = np.maximum(rho, 1e-12)
    beta = np.maximum(beta, 1e-12)
    age_ratio = np.maximum(AGE / AGE_REF, 1e-12)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        damage = 60.0 * ((ESALs / rho) ** beta) * (age_ratio ** p9)
    return 100.0 - damage


def pci_eq1_age_multiplicative_dummy_both(
    X, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11
):
    """Pooled multiplicative-AGE model with D modifying both rho and beta."""
    Mr, SN, ALT, ESALs, AGE, D = X
    rho = p1 + p2 * Mr + p3 * SN + p4 * ALT + p10 * D
    beta_scale = np.maximum(p5 + p11 * D, 1e-12)
    beta = beta_scale * (Mr ** p6) * (SN ** p7) * (ALT ** p8)
    rho = np.maximum(rho, 1e-12)
    beta = np.maximum(beta, 1e-12)
    age_ratio = np.maximum(AGE / AGE_REF, 1e-12)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        damage = 60.0 * ((ESALs / rho) ** beta) * (age_ratio ** p9)
    return 100.0 - damage


def _fit_curve_multistart(func, X, y, p0, bounds, maxfev=1500000, n_starts=12):
    """Small deterministic multi-start wrapper to reduce local-solution sensitivity."""
    rng = np.random.default_rng(SEED)
    lo = np.asarray(bounds[0], dtype=float)
    hi = np.asarray(bounds[1], dtype=float)
    p0 = np.asarray(p0, dtype=float)
    starts = [p0]
    for _ in range(max(0, n_starts - 1)):
        # Perturb positive-scale parameters logarithmically where possible.
        z = p0.copy()
        for j in range(len(z)):
            if np.isfinite(lo[j]) and np.isfinite(hi[j]):
                span = hi[j] - lo[j]
                if span <= 0:
                    continue
                jitter = rng.normal(0, 0.08) * span
                z[j] = np.clip(z[j] + jitter, lo[j] + 1e-12, hi[j] - 1e-12)
        starts.append(z)

    best = None
    best_rss = np.inf
    for start in starts:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                params, pcov = curve_fit(
                    func, X, y, p0=start, bounds=bounds, maxfev=maxfev
                )
            pred = func(X, *params)
            rss = float(np.sum((y - pred) ** 2))
            if np.isfinite(rss) and rss < best_rss:
                best = (params, pcov)
                best_rss = rss
        except Exception:
            pass
    if best is None:
        raise RuntimeError("Nonlinear regression failed for all starting values.")
    return best


def information_criteria(y_true, y_pred, k):
    n = len(y_true)
    rss = float(np.sum((np.asarray(y_true) - np.asarray(y_pred)) ** 2))
    rss = max(rss, 1e-12)
    aic = n * np.log(rss / n) + 2 * k
    bic = n * np.log(rss / n) + k * np.log(n)
    return float(aic), float(bic)


def fit_group_nonlinear(df, group_name="Type 1", use_age=True):
    y = df["PCI"].values
    if use_age:
        X = tuple(df[c].values for c in ["Mr", "SN", "ALT", "ESALs", "AGE"])
        func = pci_eq1_age
        p0 = [1.65e6, 3500, 10000, 1200, 0.015, 0.20, 0.25, 0.65, 0.25]
        lo = [1e3, 0, 0, 0, 1e-8, 0, 0, 0, 0]
        hi = [1e8, 1e5, 1e5, 1e5, 10, 5, 5, 5, 10]
        names = ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9_Age"]
    else:
        X = tuple(df[c].values for c in ["Mr", "SN", "ALT", "ESALs"])
        func = pci_eq1_no_age
        p0 = [1.65e6, 3500, 10000, 1200, 0.015, 0.20, 0.25, 0.65]
        lo = [1e3, 0, 0, 0, 1e-8, 0, 0, 0]
        hi = [1e8, 1e5, 1e5, 1e5, 10, 5, 5, 5]
        names = ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"]

    params, pcov = _fit_curve_multistart(func, X, y, p0, (lo, hi))
    pred = func(X, *params)
    metrics, residuals = calc_metrics(y, pred)
    aic, bic = information_criteria(y, pred, len(params))
    metrics["AIC"] = aic
    metrics["BIC"] = bic
    metrics["N"] = len(y)

    print("\n" + "=" * 90)
    print(f"{group_name} | {'Age-inclusive' if use_age else 'No-Age comparator'}")
    print("=" * 90)
    if use_age:
        print("PCI = 100 - 60*(ESALs/rho)^beta - p9*AGE")
        print(f"Age coefficient p9 = {params[8]:.8f} PCI/year")
    else:
        print("PCI = 100 - 60*(ESALs/rho)^beta")
    print(f"rho  = {params[0]:.6f} + {params[1]:.6f}*Mr + {params[2]:.6f}*SN + {params[3]:.6f}*ALT")
    print(f"beta = {params[4]:.8f} * Mr^{params[5]:.6f} * SN^{params[6]:.6f} * ALT^{params[7]:.6f}")
    print_metrics(metrics, "All-data calibration metrics")

    result_df = df.copy()
    result_df["PCI_pred"] = pred
    result_df["Residual"] = residuals
    return {
        "params": params, "pcov": pcov, "metrics": metrics,
        "pred": pred, "y": y, "result_df": result_df,
        "param_table": pd.DataFrame({"Parameter": names, "Value": params})
    }


def fit_group_nonlinear_age_multiplicative(df, group_name="Type 1"):
    """Full-data nonlinear calibration with multiplicative AGE modifier."""
    y = df["PCI"].values
    X = tuple(df[c].values for c in ["Mr", "SN", "ALT", "ESALs", "AGE"])
    p0 = [1.65e6, 3500, 10000, 1200, 0.015, 0.20, 0.25, 0.65, 0.10]
    lo = [1e3, 0, 0, 0, 1e-8, 0, 0, 0, 0]
    hi = [1e8, 1e5, 1e5, 1e5, 10, 5, 5, 5, 5]
    names = ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9_AgeExponent"]

    params, pcov = _fit_curve_multistart(
        pci_eq1_age_multiplicative, X, y, p0, (lo, hi), maxfev=2000000
    )
    pred = pci_eq1_age_multiplicative(X, *params)
    metrics, residuals = calc_metrics(y, pred)
    aic, bic = information_criteria(y, pred, len(params))
    metrics.update({"AIC": aic, "BIC": bic, "N": len(y)})

    print("\n" + "=" * 90)
    print(f"{group_name} | Multiplicative-Age nonlinear model")
    print("=" * 90)
    print("PCI = 100 - 60*(ESALs/rho)^beta*(AGE/1 year)^p9")
    print(f"Age exponent p9 = {params[8]:.8f} (dimensionless)")
    print(f"rho  = {params[0]:.6f} + {params[1]:.6f}*Mr + {params[2]:.6f}*SN + {params[3]:.6f}*ALT")
    print(f"beta = {params[4]:.8f} * Mr^{params[5]:.6f} * SN^{params[6]:.6f} * ALT^{params[7]:.6f}")
    print_metrics(metrics, "All-data calibration metrics")

    result_df = df.copy()
    result_df["PCI_pred"] = pred
    result_df["Residual"] = residuals
    return {
        "params": params, "pcov": pcov, "metrics": metrics,
        "pred": pred, "y": y, "result_df": result_df,
        "param_table": pd.DataFrame({"Parameter": names, "Value": params})
    }


def fit_unified_age_multiplicative_model(df):
    X = tuple(df[c].values for c in ["Mr", "SN", "ALT", "ESALs", "AGE", "D"])
    y = df["PCI"].values
    p0 = [1.65e6, 3500, 10000, 1200, 0.015, 0.20, 0.25, 0.65, 0.10, 0.0, 0.0]
    lo = [1e3, 0, 0, 0, 1e-8, 0, 0, 0, 0, -1e8, -10]
    hi = [1e8, 1e5, 1e5, 1e5, 10, 5, 5, 5, 5, 1e8, 10]
    params, pcov = _fit_curve_multistart(
        pci_eq1_age_multiplicative_dummy_both, X, y, p0, (lo, hi),
        maxfev=3000000
    )
    pred = pci_eq1_age_multiplicative_dummy_both(X, *params)
    metrics, residuals = calc_metrics(y, pred)
    aic, bic = information_criteria(y, pred, len(params))
    metrics.update({"AIC": aic, "BIC": bic, "N": len(y)})
    names = [
        "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8",
        "p9_AgeExponent", "p10_D_rho", "p11_D_beta"
    ]

    print("\n" + "=" * 90)
    print("Unified Multiplicative-Age model with D in rho and beta")
    print("=" * 90)
    print("PCI = 100 - 60*(ESALs/rho)^beta*(AGE/1 year)^p9")
    print(f"rho = {params[0]:.6f} + {params[1]:.6f}*Mr + {params[2]:.6f}*SN + {params[3]:.6f}*ALT + {params[9]:.6f}*D")
    print(f"beta = ({params[4]:.8f} + {params[10]:.8f}*D) * Mr^{params[5]:.6f} * SN^{params[6]:.6f} * ALT^{params[7]:.6f}")
    print(f"Age exponent p9 = {params[8]:.8f} (dimensionless)")
    print_metrics(metrics, "All-data calibration metrics")

    result_df = df.copy()
    result_df["PCI_pred"] = pred
    result_df["Residual"] = residuals
    return {
        "params": params, "pcov": pcov, "metrics": metrics,
        "pred": pred, "y": y, "result_df": result_df,
        "param_table": pd.DataFrame({"Parameter": names, "Value": params})
    }


def fit_unified_age_model(df):
    X = tuple(df[c].values for c in ["Mr", "SN", "ALT", "ESALs", "AGE", "D"])
    y = df["PCI"].values
    p0 = [1.65e6, 3500, 10000, 1200, 0.015, 0.20, 0.25, 0.65, 0.25, 0.0, 0.0]
    lo = [1e3, 0, 0, 0, 1e-8, 0, 0, 0, 0, -1e8, -10]
    hi = [1e8, 1e5, 1e5, 1e5, 10, 5, 5, 5, 10, 1e8, 10]
    params, pcov = _fit_curve_multistart(pci_eq1_age_dummy_both, X, y, p0, (lo, hi), maxfev=2500000)
    pred = pci_eq1_age_dummy_both(X, *params)
    metrics, residuals = calc_metrics(y, pred)
    aic, bic = information_criteria(y, pred, len(params))
    metrics.update({"AIC": aic, "BIC": bic, "N": len(y)})
    names = ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9_Age", "p10_D_rho", "p11_D_beta"]

    print("\n" + "=" * 90)
    print("Unified Age-inclusive model with D in rho and beta")
    print("=" * 90)
    print("PCI = 100 - 60*(ESALs/rho)^beta - p9*AGE")
    print(f"rho = {params[0]:.6f} + {params[1]:.6f}*Mr + {params[2]:.6f}*SN + {params[3]:.6f}*ALT + {params[9]:.6f}*D")
    print(f"beta = ({params[4]:.8f} + {params[10]:.8f}*D) * Mr^{params[5]:.6f} * SN^{params[6]:.6f} * ALT^{params[7]:.6f}")
    print(f"Age coefficient p9 = {params[8]:.8f} PCI/year")
    print_metrics(metrics, "All-data calibration metrics")

    result_df = df.copy()
    result_df["PCI_pred"] = pred
    result_df["Residual"] = residuals
    return {
        "params": params, "pcov": pcov, "metrics": metrics,
        "pred": pred, "y": y, "result_df": result_df,
        "param_table": pd.DataFrame({"Parameter": names, "Value": params})
    }


def run_part_b():
    print("\n" + "=" * 100)
    print("PART B. BASELINE MODEL - AGE-INCLUSIVE FULL-DATA CALIBRATION")
    print("=" * 100)

    df = load_and_clean_csv(FILE_DATA, dropna_for_model=True)
    df_type1 = df[df["D"] == 0].copy().reset_index(drop=True)
    df_type2 = df[df["D"] == 1].copy().reset_index(drop=True)

    print(f"Model-complete cases: total={len(df)}, Type 1={len(df_type1)}, Type 2={len(df_type2)}")

    # Same-row comparison isolates the incremental contribution of AGE.
    noage_t1 = fit_group_nonlinear(df_type1, "Type 1 (without SAMI)", use_age=False)
    age_t1 = fit_group_nonlinear(df_type1, "Type 1 (without SAMI)", use_age=True)
    multage_t1 = fit_group_nonlinear_age_multiplicative(df_type1, "Type 1 (without SAMI)")

    noage_t2 = fit_group_nonlinear(df_type2, "Type 2 (with SAMI)", use_age=False)
    age_t2 = fit_group_nonlinear(df_type2, "Type 2 (with SAMI)", use_age=True)
    multage_t2 = fit_group_nonlinear_age_multiplicative(df_type2, "Type 2 (with SAMI)")

    unified = fit_unified_age_model(df)
    unified_multage = fit_unified_age_multiplicative_model(df)

    rows = []
    for group, model, res in [
        ("Type 1", "No Age", noage_t1),
        ("Type 1", "Additive Age", age_t1),
        ("Type 1", "Multiplicative Age", multage_t1),
        ("Type 2", "No Age", noage_t2),
        ("Type 2", "Additive Age", age_t2),
        ("Type 2", "Multiplicative Age", multage_t2),
        ("Unified", "Additive Age + D in rho,beta", unified),
        ("Unified", "Multiplicative Age + D in rho,beta", unified_multage),
    ]:
        rows.append({"Group": group, "Model": model, **res["metrics"]})
    summary = pd.DataFrame(rows)
    print("\n=== Nonlinear regression comparison ===")
    print(summary.to_string(index=False))

    age_t1["result_df"].to_csv("./content/baseline_age_type1_results.csv", index=False)
    age_t2["result_df"].to_csv("./content/baseline_age_type2_results.csv", index=False)
    unified["result_df"].to_csv("./content/baseline_age_unified_results.csv", index=False)
    multage_t1["result_df"].to_csv("./content/baseline_multage_type1_results.csv", index=False)
    multage_t2["result_df"].to_csv("./content/baseline_multage_type2_results.csv", index=False)
    unified_multage["result_df"].to_csv("./content/baseline_multage_unified_results.csv", index=False)

    age_t1["param_table"].to_csv("./content/baseline_age_type1_parameters.csv", index=False)
    age_t2["param_table"].to_csv("./content/baseline_age_type2_parameters.csv", index=False)
    unified["param_table"].to_csv("./content/baseline_age_unified_parameters.csv", index=False)
    multage_t1["param_table"].to_csv("./content/baseline_multage_type1_parameters.csv", index=False)
    multage_t2["param_table"].to_csv("./content/baseline_multage_type2_parameters.csv", index=False)
    unified_multage["param_table"].to_csv("./content/baseline_multage_unified_parameters.csv", index=False)
    summary.to_csv("./content/baseline_age_summary.csv", index=False)

    plot_pred_vs_true(age_t1["y"], age_t1["pred"], "Type 1 - Additive Age nonlinear calibration", "./content/baseline_age_type1_fit.png")
    plot_pred_vs_true(age_t2["y"], age_t2["pred"], "Type 2 - Additive Age nonlinear calibration", "./content/baseline_age_type2_fit.png")
    plot_pred_vs_true(unified["y"], unified["pred"], "Unified - Additive Age + D nonlinear calibration", "./content/baseline_age_unified_fit.png")
    plot_pred_vs_true(multage_t1["y"], multage_t1["pred"], "Type 1 - Multiplicative Age nonlinear calibration", "./content/baseline_multage_type1_fit.png")
    plot_pred_vs_true(multage_t2["y"], multage_t2["pred"], "Type 2 - Multiplicative Age nonlinear calibration", "./content/baseline_multage_type2_fit.png")
    plot_pred_vs_true(unified_multage["y"], unified_multage["pred"], "Unified - Multiplicative Age + D nonlinear calibration", "./content/baseline_multage_unified_fit.png")

    return df_type1, df_type2


# ============================================================
# PART C. RESNET + 1D-CNN + TRANSFER LEARNING
# ============================================================

class PCIDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].unsqueeze(0), self.y[idx]


class ResidualBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=3, dropout=0.1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.block(x)
        out = out + identity
        out = self.relu(out)
        return out


class ResNet1DCNN(nn.Module):
    def __init__(self, input_len, base_channels=16, dropout=0.15):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
        )

        self.res1 = ResidualBlock1D(base_channels, kernel_size=3, dropout=dropout)
        self.res2 = ResidualBlock1D(base_channels, kernel_size=3, dropout=dropout)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * input_len, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.head(x)
        return x


def print_model_summary(model, input_size, device=device):
    summary_rows = []
    hooks = []

    def register_hook(module):
        def hook(module, inputs, outputs):
            class_name = module.__class__.__name__

            if isinstance(outputs, (list, tuple)):
                output_shape = [list(o.shape) for o in outputs if hasattr(o, "shape")]
            elif hasattr(outputs, "shape"):
                output_shape = list(outputs.shape)
            else:
                output_shape = str(type(outputs))

            params = 0
            trainable = 0
            for p in module.parameters(recurse=False):
                params += p.numel()
                if p.requires_grad:
                    trainable += p.numel()

            if params > 0:
                summary_rows.append(
                    {
                        "Layer": class_name,
                        "Output Shape": output_shape,
                        "Param #": params,
                        "Trainable #": trainable,
                    }
                )

        if not isinstance(module, nn.Sequential) and module != model:
            hooks.append(module.register_forward_hook(hook))

    model.apply(register_hook)

    model.eval()
    x = torch.zeros(*input_size).to(device)
    with torch.no_grad():
        _ = model(x)

    for h in hooks:
        h.remove()

    summary_df = pd.DataFrame(summary_rows)
    total_params = int(summary_df["Param #"].sum()) if not summary_df.empty else 0
    trainable_params = int(summary_df["Trainable #"].sum()) if not summary_df.empty else 0
    non_trainable_params = total_params - trainable_params

    print("\n" + "=" * 90)
    print("MODEL SUMMARY")
    print("=" * 90)
    if not summary_df.empty:
        print(summary_df.to_string(index=False))
    else:
        print("No parameterized layers found.")
    print("-" * 90)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Non-trainable params: {non_trainable_params:,}")
    print("=" * 90)

    return summary_df


def get_loss_fn():
    """MSE loss used for all direct and transfer-learning neural networks."""
    return nn.MSELoss()


def train_model(
    model,
    train_loader,
    lr=1e-3,
    max_epochs=400,
    verbose_every=20,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = get_loss_fn()

    history = {"train_loss": []}

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        history["train_loss"].append(train_loss)

        if epoch % verbose_every == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{max_epochs} | train_loss={train_loss:.6f}")

    return model, history


def predict_model(model, loader):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy().ravel()
            preds.extend(pred.tolist())
            trues.extend(yb.numpy().ravel().tolist())
    return np.array(trues), np.array(preds)


def build_features_train_test(df, test_size=TEST_RATIO, random_state=SEED):
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_raw)
    X_test_poly = poly.transform(X_test_raw)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_poly)
    X_test = scaler.transform(X_test_poly)

    return {
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "poly": poly,
        "scaler": scaler,
        "feature_names": poly.get_feature_names_out(FEATURE_COLS),
    }


def make_loaders(X_train, y_train, X_val, y_val, batch_size=16):
    train_ds = PCIDataset(X_train, y_train)
    val_ds = PCIDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def plot_history(history, title, save_path=None):
    plt.figure(figsize=(6, 4))
    plt.plot(history["train_loss"], label="Train")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()

    if save_path is None:
        save_path = auto_figure_path(title)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")

    plt.close()


def run_direct_model(data_dict, group_name="Type 1", max_epochs=300):
    input_len = data_dict["X_train"].shape[1]

    train_loader = DataLoader(
        PCIDataset(data_dict["X_train"], data_dict["y_train"]),
        batch_size=16,
        shuffle=True,
    )

    test_loader = DataLoader(
        PCIDataset(data_dict["X_test"], data_dict["y_test"]),
        batch_size=32,
        shuffle=False,
    )

    model = ResNet1DCNN(input_len=input_len, base_channels=16, dropout=0.15).to(device)

    print(f"\nModel summary for {group_name}:")
    print_model_summary(model, input_size=(1, 1, input_len), device=device)

    print(f"\nTraining direct model for {group_name} with MSE loss...")
    model, history = train_model(
        model,
        train_loader,
        lr=1e-3,
        max_epochs=max_epochs,
        verbose_every=20,
    )

    y_true, y_pred = predict_model(model, test_loader)
    metrics, _ = calc_metrics(y_true, y_pred)

    print_metrics(metrics, f"Direct model metrics for {group_name} with MSE loss")

    return {
        "model": model,
        "history": history,
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def run_transfer_t1_to_t2(
    data_t1,
    data_t2,
    pretrain_epochs=300,
    finetune_epochs=250,
):
    input_len = data_t1["X_train"].shape[1]

    src_train_loader = DataLoader(
        PCIDataset(data_t1["X_train"], data_t1["y_train"]),
        batch_size=16,
        shuffle=True,
    )

    model = ResNet1DCNN(input_len=input_len, base_channels=16, dropout=0.15).to(device)

    print("\nModel summary for transfer-learning model:")
    print_model_summary(model, input_size=(1, 1, input_len), device=device)

    print("\nPretraining on Type 1 with MSE loss...")
    model, src_history = train_model(
        model,
        src_train_loader,
        lr=1e-3,
        max_epochs=pretrain_epochs,
        verbose_every=20,
    )

    poly_src = data_t1["poly"]
    scaler_src = data_t1["scaler"]

    X2_train_raw = data_t2["X_train_raw"]
    X2_test_raw = data_t2["X_test_raw"]
    y2_train = data_t2["y_train"]
    y2_test = data_t2["y_test"]

    X2_train_poly = poly_src.transform(X2_train_raw)
    X2_test_poly = poly_src.transform(X2_test_raw)

    X2_train = scaler_src.transform(X2_train_poly)
    X2_test = scaler_src.transform(X2_test_poly)

    tgt_train_loader = DataLoader(
        PCIDataset(X2_train, y2_train),
        batch_size=8,
        shuffle=True,
    )

    tgt_test_loader = DataLoader(
        PCIDataset(X2_test, y2_test),
        batch_size=32,
        shuffle=False,
    )

    print("\nFine-tuning on Type 2 with MSE loss...")
    model, tgt_history = train_model(
        model,
        tgt_train_loader,
        lr=5e-4,
        max_epochs=finetune_epochs,
        verbose_every=20,
    )

    y_true, y_pred = predict_model(model, tgt_test_loader)
    metrics, _ = calc_metrics(y_true, y_pred)

    print_metrics(metrics, "Transfer learning metrics (Type 1 -> Type 2) with MSE loss")

    return {
        "model": model,
        "pretrain_history": src_history,
        "finetune_history": tgt_history,
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
    }


# ============================================================
# PART D. EXPLAINABLE AI
# ============================================================

def sensitivity_check(model, base_point, feature_names, step=0.05):
    expected = {
        "Mr": "positive",
        "SN": "positive",
        "ALT": "positive",
        "ESALs": "negative",
        "AGE": "negative",
    }

    rows = []
    base_pred = model.predict(base_point.reshape(1, -1))[0]

    for i, feat in enumerate(feature_names):
        x2 = base_point.copy()
        x2[i] = min(1.0, x2[i] + step)
        pred2 = model.predict(x2.reshape(1, -1))[0]
        delta = pred2 - base_pred

        if expected[feat] == "positive":
            consistent = delta > 0
        else:
            consistent = delta < 0

        rows.append(
            {
                "Feature": feat,
                "BaseValue_norm": float(base_point[i]),
                "Step_norm": step,
                "DeltaPCI": float(delta),
                "ExpectedDirection": expected[feat],
                "DirectionConsistent": bool(consistent),
            }
        )

    return pd.DataFrame(rows)



def _iter_symbolic_programs(symbolic_model):
    """Yield unique gplearn programs retained across generations."""
    seen = set()

    for generation in getattr(symbolic_model, "_programs", []) or []:
        if generation is None:
            continue
        for program in generation:
            if program is None:
                continue
            key = id(program)
            if key not in seen:
                seen.add(key)
                yield program

    final_program = getattr(symbolic_model, "_program", None)
    if final_program is not None and id(final_program) not in seen:
        yield final_program


def select_symbolic_program_with_node_cap(
    symbolic_model,
    X_train,
    y_train,
    max_nodes=SYMBOLIC_MAX_NODES,
):
    """Select the best TRAINING-MSE program subject to a hard node cap.

    The final 30% test set must never be supplied to this function. Candidate
    symbolic expressions are generated and ranked using training data only.
    The selected program replaces symbolic_model._program so that subsequent
    symbolic_model.predict(...) calls use the node-limited equation.
    """
    candidates = []
    total_finite = 0

    for program in _iter_symbolic_programs(symbolic_model):
        try:
            length = int(program.length_)
            pred = np.asarray(program.execute(X_train), dtype=float).ravel()
        except Exception:
            continue

        if pred.shape[0] != len(y_train) or not np.all(np.isfinite(pred)):
            continue

        total_finite += 1
        if length > max_nodes:
            continue

        mse = float(mean_squared_error(y_train, pred))
        candidates.append((mse, length, program))

    if not candidates:
        raise RuntimeError(
            f"No finite symbolic-regression candidate with <= {max_nodes} nodes was found. "
            "Increase SYMBOLIC_MAX_NODES slightly or reduce initial-tree complexity."
        )

    # Primary criterion: training MSE. Secondary criterion: shorter equation.
    candidates.sort(key=lambda item: (item[0], item[1]))
    best_mse, best_length, best_program = candidates[0]
    symbolic_model._program = best_program

    train_pred = symbolic_model.predict(X_train)
    train_metrics, _ = calc_metrics(y_train, train_pred)

    return {
        "MaxNodes": int(max_nodes),
        "ProgramLength": int(best_length),
        "Training_MSE": float(best_mse),
        "Training_R2": float(train_metrics["R2"]),
        "EligibleCandidateCount": int(len(candidates)),
        "FiniteCandidateCount": int(total_finite),
    }

def run_explainable_models(df, group_name="Type 1", test_size=TEST_RATIO, random_state=SEED):
    print("\n" + "=" * 100)
    print(f"Explainable AI analysis for {group_name}")
    print("=" * 100)

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    # Shared 70:30 train/test split. The held-out 30% test set is never used
    # to select the symbolic equation or tune the decision tree.
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    # ------------------------------------------------------------------
    # Symbolic regression
    # ------------------------------------------------------------------
    sym = None
    sym_metrics = None
    sym_sens = pd.DataFrame()
    y_pred_sym = None
    symbolic_selection = None

    if GPLEARN_AVAILABLE:
        print("\nRunning symbolic regression with a <=30-node final equation...")

        # No separate validation subset is used because the target-domain sample
        # is small. Symbolic evolution and <=30-node model selection use only
        # the 70% training subset; the 30% test subset remains untouched.
        sr_scaler = MinMaxScaler()
        X_sr_train = sr_scaler.fit_transform(X_train_raw)
        X_sr_test = sr_scaler.transform(X_test_raw)
        y_sr_train = y_train
        sr_base_point = np.mean(X_sr_train, axis=0)

        sym = SymbolicRegressor(
            population_size=4000,
            generations=40,
            tournament_size=20,
            stopping_criteria=0.001,
            const_range=(-5.0, 5.0),
            init_depth=(2, 3),
            init_method="half and half",
            function_set=SYMBOLIC_FUNCTION_SET,
            metric="mse",
            parsimony_coefficient=SYMBOLIC_PARSIMONY,
            p_crossover=0.65,
            p_subtree_mutation=0.10,
            p_hoist_mutation=0.15,
            p_point_mutation=0.05,
            p_point_replace=0.05,
            max_samples=0.9,
            feature_names=FEATURE_COLS,
            verbose=1,
            random_state=random_state,
            n_jobs=1,
            low_memory=False,
        )
        sym.fit(X_sr_train, y_sr_train)

        symbolic_selection = select_symbolic_program_with_node_cap(
            sym,
            X_sr_train,
            y_sr_train,
            max_nodes=SYMBOLIC_MAX_NODES,
        )

        # Final test evaluation occurs only after training-only equation selection.
        y_pred_sym = sym.predict(X_sr_test)
        sym_metrics, _ = calc_metrics(y_test, y_pred_sym)

        print("\nSelected symbolic regression equation:")
        print(sym._program)
        print(
            f"Program length: {symbolic_selection['ProgramLength']} nodes "
            f"(hard cap = {symbolic_selection['MaxNodes']})"
        )
        print(
            f"Training MSE: {symbolic_selection['Training_MSE']:.6f}; "
            f"Training R2: {symbolic_selection['Training_R2']:.6f}; "
            f"eligible candidates: {symbolic_selection['EligibleCandidateCount']}"
        )
        print_metrics(sym_metrics, "Symbolic regression TEST metrics")

        sym_sens = sensitivity_check(sym, sr_base_point, FEATURE_COLS, step=0.05)
        print("\nSymbolic regression sensitivity check")
        print(sym_sens.to_string(index=False))

        plot_pred_vs_true(y_test, y_pred_sym, f"{group_name} - Symbolic Regression")
        plot_residuals(y_test, y_pred_sym, f"{group_name} - Symbolic Regression Residuals")
    else:
        print("\n[gplearn not installed] Symbolic regression skipped in this environment.")

    # ------------------------------------------------------------------
    # Decision tree (keeps the original outer train/test protocol)
    # ------------------------------------------------------------------
    tree_scaler = MinMaxScaler()
    X_train = tree_scaler.fit_transform(X_train_raw)
    X_test = tree_scaler.transform(X_test_raw)
    tree_base_point = np.mean(X_train, axis=0)

    print("\nRunning decision tree regression...")
    tree = DecisionTreeRegressor(random_state=random_state)

    cv_folds = min(5, len(X_train))
    if cv_folds < 2:
        raise ValueError(f"Not enough samples for cross-validation in {group_name}.")

    param_grid = {
        "max_depth": [2, 3, 4, 5, 6],
        "min_samples_leaf": [2, 3, 5, 8],
        "criterion": ["squared_error", "absolute_error"],
    }

    grid = GridSearchCV(
        tree,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=cv_folds,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)

    best_tree = grid.best_estimator_
    y_pred_tree = best_tree.predict(X_test)
    tree_metrics, _ = calc_metrics(y_test, y_pred_tree)

    print("\nBest decision tree parameters:")
    print(grid.best_params_)
    print_metrics(tree_metrics, "Decision tree metrics")

    print("\nDecision tree rules:")
    tree_rules = export_text(best_tree, feature_names=list(FEATURE_COLS))
    print(tree_rules)

    tree_sens = sensitivity_check(best_tree, tree_base_point, FEATURE_COLS, step=0.05)
    print("\nDecision tree sensitivity check")
    print(tree_sens.to_string(index=False))

    plot_pred_vs_true(y_test, y_pred_tree, f"{group_name} - Decision Tree")
    plot_residuals(y_test, y_pred_tree, f"{group_name} - Decision Tree Residuals")

    plt.figure(figsize=(16, 8))
    plot_tree(best_tree, feature_names=FEATURE_COLS, filled=True, rounded=True, fontsize=9)
    plt.title(f"{group_name} - Decision Tree")
    tree_figure_path = auto_figure_path(f"{group_name} - Decision Tree Structure")
    plt.tight_layout()
    plt.savefig(tree_figure_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {tree_figure_path}")
    plt.close()

    summary_rows = []
    if sym_metrics is not None:
        summary_rows.append(
            {
                "Group": group_name,
                "Model": "SymbolicRegression",
                **sym_metrics,
                "ProgramLength": symbolic_selection["ProgramLength"],
                "MaxNodes": symbolic_selection["MaxNodes"],
                "Training_MSE": symbolic_selection["Training_MSE"],
                "Training_R2": symbolic_selection["Training_R2"],
            }
        )
    summary_rows.append(
        {
            "Group": group_name,
            "Model": "DecisionTree",
            **tree_metrics,
            "ProgramLength": np.nan,
            "MaxNodes": np.nan,
            "Training_MSE": np.nan,
            "Training_R2": np.nan,
        }
    )
    summary = pd.DataFrame(summary_rows)

    sym_pred_df = None
    if y_pred_sym is not None:
        sym_pred_df = pd.DataFrame(
            {"Observed_PCI": y_test, "Predicted_PCI": y_pred_sym, "Residual": y_test - y_pred_sym}
        )

    tree_pred_df = pd.DataFrame(
        {
            "Observed_PCI": y_test,
            "Predicted_PCI": y_pred_tree,
            "Residual": y_test - y_pred_tree,
        }
    )

    safe_group = group_name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")

    summary.to_csv(f"./content/{safe_group}_xai_summary.csv", index=False)
    if sym_pred_df is not None:
        sym_pred_df.to_csv(f"./content/{safe_group}_symbolic_predictions.csv", index=False)
    tree_pred_df.to_csv(f"./content/{safe_group}_tree_predictions.csv", index=False)
    if not sym_sens.empty:
        sym_sens.to_csv(f"./content/{safe_group}_symbolic_sensitivity.csv", index=False)
    tree_sens.to_csv(f"./content/{safe_group}_tree_sensitivity.csv", index=False)

    if sym is not None:
        with open(f"./content/{safe_group}_symbolic_equation.txt", "w") as f:
            f.write(str(sym._program))

        selection_df = pd.DataFrame(
            [
                {
                    "Group": group_name,
                    "Train_N": len(y_sr_train),
                    "Test_N": len(y_test),
                    "Train_Ratio": TRAIN_RATIO,
                    "Test_Ratio": TEST_RATIO,
                    **symbolic_selection,
                    "Equation": str(sym._program),
                }
            ]
        )
        selection_df.to_csv(
            f"./content/{safe_group}_symbolic_selection_diagnostics.csv",
            index=False,
        )

    with open(f"./content/{safe_group}_tree_rules.txt", "w") as f:
        f.write(tree_rules)

    print(f"\nSaved XAI outputs for {group_name}")

    return {
        "summary": summary,
        "symbolic_model": sym,
        "tree_model": best_tree,
        "symbolic_metrics": sym_metrics,
        "tree_metrics": tree_metrics,
        "symbolic_equation": str(sym._program) if sym is not None else None,
        "symbolic_selection": symbolic_selection,
        "tree_rules": tree_rules,
        "symbolic_sensitivity": sym_sens,
        "tree_sensitivity": tree_sens,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    ensure_input_files_exist()

    run_part_a()
    df_type1, df_type2 = run_part_b()

    print("\n" + "=" * 100)
    print("PART C. RESNET + 1D-CNN + TRANSFER LEARNING")
    print("=" * 100)

    data_t1 = build_features_train_test(df_type1, test_size=TEST_RATIO, random_state=SEED)
    data_t2 = build_features_train_test(df_type2, test_size=TEST_RATIO, random_state=SEED)

    print(
        f"Shared hold-out protocol: train={TRAIN_RATIO:.0%}, test={TEST_RATIO:.0%}, random_state={SEED}"
    )
    print(
        f"Type 1 split: train={len(data_t1['y_train'])}, test={len(data_t1['y_test'])}, total={len(df_type1)}"
    )
    print(
        f"Type 2 split: train={len(data_t2['y_train'])}, test={len(data_t2['y_test'])}, total={len(df_type2)}"
    )

    input_len = data_t1["X_train"].shape[1]

    save_residual_block_diagram(
        out_path="./content/model_diagrams/residual_block_1d.png",
    )

    save_resnet_1dcnn_architecture(
        input_len=input_len,
        out_path="./content/model_diagrams/resnet_1dcnn_architecture.png",
    )

    save_transfer_learning_workflow(
        out_path="./content/model_diagrams/transfer_learning_workflow.png",
    )

    save_symbolic_regression_workflow(
        out_path="./content/model_diagrams/symbolic_regression_workflow.png",
    )

    save_decision_tree_workflow(
        out_path="./content/model_diagrams/decision_tree_workflow.png",
    )

    print("Expanded feature dimension for Type 1:", data_t1["X_train"].shape[1])
    print("Expanded feature dimension for Type 2:", data_t2["X_train"].shape[1])

    results_summary = []
    saved_predictions = {}

    # Direct ResNet-1D-CNN models: MSE only.
    res_t1_direct = run_direct_model(
        data_t1,
        group_name="Type 1",
        max_epochs=300,
    )
    plot_history(res_t1_direct["history"], "Type 1 direct - MSE loss")
    plot_pred_vs_true(
        res_t1_direct["y_true"],
        res_t1_direct["y_pred"],
        "Type 1 direct - MSE loss",
    )
    results_summary.append(
        {"Model": "Type1_direct", "Loss": "MSE", **res_t1_direct["metrics"]}
    )
    saved_predictions["Type1_direct_mse"] = (
        res_t1_direct["y_true"],
        res_t1_direct["y_pred"],
    )

    res_t2_direct = run_direct_model(
        data_t2,
        group_name="Type 2",
        max_epochs=300,
    )
    plot_history(res_t2_direct["history"], "Type 2 direct - MSE loss")
    plot_pred_vs_true(
        res_t2_direct["y_true"],
        res_t2_direct["y_pred"],
        "Type 2 direct - MSE loss",
    )
    results_summary.append(
        {"Model": "Type2_direct", "Loss": "MSE", **res_t2_direct["metrics"]}
    )
    saved_predictions["Type2_direct_mse"] = (
        res_t2_direct["y_true"],
        res_t2_direct["y_pred"],
    )

    # Type 1 -> Type 2 transfer learning: MSE only.
    res_tf = run_transfer_t1_to_t2(
        data_t1,
        data_t2,
        pretrain_epochs=300,
        finetune_epochs=250,
    )
    plot_history(res_tf["pretrain_history"], "Pretraining on Type 1 - MSE loss")
    plot_history(res_tf["finetune_history"], "Fine-tuning on Type 2 - MSE loss")
    plot_pred_vs_true(
        res_tf["y_true"],
        res_tf["y_pred"],
        "Transfer T1→T2 - MSE loss",
    )
    results_summary.append(
        {"Model": "Transfer_T1_to_T2", "Loss": "MSE", **res_tf["metrics"]}
    )
    saved_predictions["Transfer_T1_to_T2_mse"] = (
        res_tf["y_true"],
        res_tf["y_pred"],
    )

    summary_dl = pd.DataFrame(results_summary)
    print("\n" + "=" * 100)
    print("Model comparison summary")
    print("=" * 100)
    print(summary_dl)

    summary_dl.to_csv("./content/resnet_1dcnn_transfer_summary.csv", index=False)
    print("\nSaved: ./content/resnet_1dcnn_transfer_summary.csv")

    y_true, y_pred = saved_predictions["Type2_direct_mse"]
    save_prediction_csv(y_true, y_pred, "./content/type2_direct_mse_predictions.csv")

    y_true, y_pred = saved_predictions["Transfer_T1_to_T2_mse"]
    save_prediction_csv(y_true, y_pred, "./content/transfer_t1_to_t2_mse_predictions.csv")

    df_direct = pd.read_csv("./content/type2_direct_mse_predictions.csv")
    df_transfer = pd.read_csv("./content/transfer_t1_to_t2_mse_predictions.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    ax = axes[0]
    ax.scatter(df_direct["Observed_PCI"], df_direct["Predicted_PCI"], alpha=0.8)
    mn = min(df_direct["Observed_PCI"].min(), df_direct["Predicted_PCI"].min())
    mx = max(df_direct["Observed_PCI"].max(), df_direct["Predicted_PCI"].max())
    ax.plot([mn, mx], [mn, mx], "k--")
    ax.set_xlabel("Observed PCI")
    ax.set_ylabel("Predicted PCI")
    ax.set_title("Direct Type 2 model - MSE")

    ax = axes[1]
    ax.scatter(df_transfer["Observed_PCI"], df_transfer["Predicted_PCI"], alpha=0.8)
    mn = min(df_transfer["Observed_PCI"].min(), df_transfer["Predicted_PCI"].min())
    mx = max(df_transfer["Observed_PCI"].max(), df_transfer["Predicted_PCI"].max())
    ax.plot([mn, mx], [mn, mx], "k--")
    ax.set_xlabel("Observed PCI")
    ax.set_ylabel("Predicted PCI")
    ax.set_title("Transfer learning - MSE")

    plt.tight_layout()
    plt.savefig("./content/transfer_fit_combined.png", dpi=300, bbox_inches="tight")
    print("Saved: ./content/transfer_fit_combined.png")
    plt.close()

    plot_pred_vs_true(
        df_direct["Observed_PCI"].values,
        df_direct["Predicted_PCI"].values,
        "Direct Type 2 model - MSE",
        save_path="./content/type2_direct_mse_fit.png",
    )

    plot_pred_vs_true(
        df_transfer["Observed_PCI"].values,
        df_transfer["Predicted_PCI"].values,
        "Transfer learning - MSE",
        save_path="./content/transfer_t1_to_t2_mse_fit.png",
    )

    print("\n" + "=" * 100)
    print("PART D. EXPLAINABLE AI")
    print("=" * 100)

    res_xai_t1 = run_explainable_models(
        df_type1, group_name="Type 1 (without SAMI)", test_size=TEST_RATIO, random_state=SEED
    )
    res_xai_t2 = run_explainable_models(
        df_type2, group_name="Type 2 (with SAMI)", test_size=TEST_RATIO, random_state=SEED
    )

    combined_summary_xai = pd.concat(
        [res_xai_t1["summary"], res_xai_t2["summary"]],
        ignore_index=True,
    )

    print("\n" + "=" * 100)
    print("Combined explainable AI summary")
    print("=" * 100)
    print(combined_summary_xai)

    combined_summary_xai.to_csv("./content/explainable_ai_combined_summary.csv", index=False)
    print("\nSaved: ./content/explainable_ai_combined_summary.csv")

    print("\n" + "=" * 100)
    print("ALL DONE")
    print("=" * 100)

    print("\nMain output files:")
    output_files = [
        "./content/descriptive_stats_without_SAMI.csv",
        "./content/descriptive_stats_with_SAMI.csv",
        "./content/descriptive_stats_combined.csv",
        "./content/descriptive_stats_combined_rounded.csv",
        "./content/baseline_age_type1_results.csv",
        "./content/baseline_age_type2_results.csv",
        "./content/baseline_age_unified_results.csv",
        "./content/baseline_multage_type1_results.csv",
        "./content/baseline_multage_type2_results.csv",
        "./content/baseline_multage_unified_results.csv",
        "./content/baseline_age_summary.csv",
        "./content/baseline_fit_combined.png",
        "./content/baseline_type1_fit.png",
        "./content/baseline_type2_fit.png",
        "./content/resnet_1dcnn_transfer_summary.csv",
        "./content/type2_direct_mse_predictions.csv",
        "./content/transfer_t1_to_t2_mse_predictions.csv",
        "./content/transfer_fit_combined.png",
        "./content/type2_direct_mse_fit.png",
        "./content/transfer_t1_to_t2_mse_fit.png",
        "./content/explainable_ai_combined_summary.csv",
        "./content/Type_1_without_SAMI_symbolic_selection_diagnostics.csv",
        "./content/Type_2_with_SAMI_symbolic_selection_diagnostics.csv",
    ]


    for f in output_files:
        if os.path.exists(f):
            print(f)


if __name__ == "__main__":
    main()
