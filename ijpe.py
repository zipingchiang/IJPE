# ============================================================
# Optional Colab installation
# ============================================================
# !pip install -q scikit-learn==1.5.2 gplearn==0.4.2 graphviz

import os
import random
import warnings
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit, OptimizeWarning

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree

from gplearn.genetic import SymbolicRegressor

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

FILE_TYPE1 = "/content/Kinmen_PCI-0.csv"   # without SAMI
FILE_TYPE2 = "/content/Kinmen_PCI-1.csv"   # with SAMI

FEATURE_COLS = ["Mr", "SN", "ALT", "ESALs"]
TARGET_COL = "PCI"
RAW_FEATURE_COLS = ["Mr", "SN", "ALT", "ESALs"]


# ============================================================
# Shared utilities
# ============================================================

def ensure_input_files_exist():
    missing = [p for p in [FILE_TYPE1, FILE_TYPE2] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing required input file(s):\n"
            + "\n".join(missing)
            + "\n\nPlease upload them to /content in Colab first."
        )


def load_and_clean_csv(file_path, rename=True, dropna=True):
    df = pd.read_csv(file_path)
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]

    cols_raw = ["F1", "F2", "F3", "F4", "PCI"]
    cols_named = ["Mr", "SN", "ALT", "ESALs", "PCI"]

    if all(c in df.columns for c in cols_raw):
        df = df[cols_raw].copy()
        for c in cols_raw:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        if dropna:
            df = df.dropna().reset_index(drop=True)

        if rename:
            df = df.rename(
                columns={
                    "F1": "Mr",
                    "F2": "SN",
                    "F3": "ALT",
                    "F4": "ESALs",
                }
            )

    elif all(c in df.columns for c in cols_named):
        df = df[cols_named].copy()
        for c in cols_named:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        if dropna:
            df = df.dropna().reset_index(drop=True)

    else:
        raise ValueError(
            f"{file_path} must contain either columns {cols_raw} or {cols_named}. "
            f"Current columns: {list(df.columns)}"
        )

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
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_residuals(y_true, y_pred, title, save_path=None):
    residuals = y_true - y_pred
    plt.figure(figsize=(5, 4))
    plt.scatter(y_pred, residuals, alpha=0.8)
    plt.axhline(0, color="k", linestyle="--")
    plt.xlabel("Predicted PCI")
    plt.ylabel("Residual")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def split_df_train_test(df, test_size=0.2, random_state=SEED):
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

MODEL_DIAGRAM_DIR = "/content/model_diagrams"


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
    plt.show()


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
    plt.show()


def save_resnet_1dcnn_architecture(input_len, out_path=None):
    """Save a schematic of the ResNet-1D-CNN architecture used for PCI prediction."""
    ensure_model_diagram_dir()

    if out_path is None:
        out_path = os.path.join(MODEL_DIAGRAM_DIR, "resnet_1dcnn_architecture.png")

    steps = [
        "Observed variables\nMr, SN, ALT, ESALs",
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
        "Scaled observed inputs\nMr, SN, ALT, ESALs",
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
        "Scaled observed inputs\nMr, SN, ALT, ESALs",
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
    plt.show()


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
    plt.show()


# ============================================================
# PART A. DESCRIPTIVE STATISTICS
# ============================================================

def run_part_a():
    print("\n" + "=" * 100)
    print("PART A. DESCRIPTIVE STATISTICS")
    print("=" * 100)

    df0_raw = load_and_clean_csv(FILE_TYPE1, rename=True, dropna=False)
    df1_raw = load_and_clean_csv(FILE_TYPE2, rename=True, dropna=False)

    print("\n=== Basic file check ===")
    print(f"{FILE_TYPE1}: shape = {df0_raw.shape}")
    print(f"{FILE_TYPE2}: shape = {df1_raw.shape}")

    print("\nMissing values in without SAMI:")
    print(df0_raw.isna().sum())

    print("\nMissing values in with SAMI:")
    print(df1_raw.isna().sum())

    stats0 = descriptive_stats(df0_raw)
    stats1 = descriptive_stats(df1_raw)

    stats0_fmt = format_stats_table(stats0)
    stats1_fmt = format_stats_table(stats1)

    print("\n=== Descriptive statistics: without SAMI ===")
    print(stats0_fmt)

    print("\n=== Descriptive statistics: with SAMI ===")
    print(stats1_fmt)

    combined = pd.DataFrame(index=["Count", "Mean", "Std. Dev.", "Min.", "Max."])

    for col in ["Mr", "SN", "ALT", "ESALs", "PCI"]:
        combined[("Without SAMI", col)] = stats0.loc[
            ["Count", "Mean", "Std. Dev.", "Min.", "Max."], col
        ].values

    for col in ["Mr", "SN", "ALT", "ESALs", "PCI"]:
        combined[("With SAMI", col)] = stats1.loc[
            ["Count", "Mean", "Std. Dev.", "Min.", "Max."], col
        ].values

    combined.columns = pd.MultiIndex.from_tuples(combined.columns)

    print("\n=== Combined descriptive statistics table ===")
    print(combined)

    combined_round = combined.copy()
    for idx in combined_round.index:
        if idx == "Count":
            combined_round.loc[idx] = combined_round.loc[idx].astype(int)
        else:
            combined_round.loc[idx] = combined_round.loc[idx].astype(float)

    for idx in ["Mean", "Std. Dev.", "Min.", "Max."]:
        for col in combined_round.columns:
            variable = col[1]
            if variable == "ESALs":
                combined_round.loc[idx, col] = round(float(combined_round.loc[idx, col]), 0)
            else:
                combined_round.loc[idx, col] = round(float(combined_round.loc[idx, col]), 2)

    print("\n=== Rounded table for manuscript comparison ===")
    print(combined_round)

    dup0 = df0_raw.duplicated().sum()
    dup1 = df1_raw.duplicated().sum()

    print("\n=== Duplicate row check ===")
    print(f"Without SAMI duplicate rows: {dup0}")
    print(f"With SAMI duplicate rows: {dup1}")

    if dup0 > 0:
        print("\nDuplicate rows in without SAMI:")
        print(df0_raw[df0_raw.duplicated(keep=False)].sort_values(list(df0_raw.columns)))

    if dup1 > 0:
        print("\nDuplicate rows in with SAMI:")
        print(df1_raw[df1_raw.duplicated(keep=False)].sort_values(list(df1_raw.columns)))

    stats0.to_csv("/content/descriptive_stats_without_SAMI.csv")
    stats1.to_csv("/content/descriptive_stats_with_SAMI.csv")
    combined.to_csv("/content/descriptive_stats_combined.csv")
    combined_round.to_csv("/content/descriptive_stats_combined_rounded.csv")

    print("\nFiles saved:")
    print("/content/descriptive_stats_without_SAMI.csv")
    print("/content/descriptive_stats_with_SAMI.csv")
    print("/content/descriptive_stats_combined.csv")
    print("/content/descriptive_stats_combined_rounded.csv")


# ============================================================
# PART B. BASELINE MODEL
# ============================================================

def pci_eq1(X, p1, p2, p3, p4, p5, p6, p7, p8):
    Mr, SN, ALT, ESALs = X
    rho = p1 + p2 * Mr + p3 * SN + p4 * ALT
    beta = p5 * (Mr ** p6) * (SN ** p7) * (ALT ** p8)
    rho = np.maximum(rho, 1e-12)
    beta = np.maximum(beta, 1e-12)
    pci = 100 - 60 * ((ESALs / rho) ** beta)
    return pci


def pci_eq1_dummy(X, p1, p2, p3, p4, p5, p9, p6, p7, p8):
    Mr, SN, ALT, ESALs, SAMI = X
    rho = p1 + p2 * Mr + p3 * SN + p4 * ALT
    beta = (p5 + p9 * SAMI) * (Mr ** p6) * (SN ** p7) * (ALT ** p8)
    rho = np.maximum(rho, 1e-12)
    beta = np.maximum(beta, 1e-12)
    pci = 100 - 60 * ((ESALs / rho) ** beta)
    return pci


def fit_eq1_baseline_train_test(
    df,
    group_name="Baseline model",
    test_size=0.2,
    random_state=SEED,
    initial_guess=None,
):
    train_df, test_df = split_df_train_test(df, test_size=test_size, random_state=random_state)

    X_train = (
        train_df["Mr"].values,
        train_df["SN"].values,
        train_df["ALT"].values,
        train_df["ESALs"].values,
    )
    y_train = train_df["PCI"].values

    X_test = (
        test_df["Mr"].values,
        test_df["SN"].values,
        test_df["ALT"].values,
        test_df["ESALs"].values,
    )
    y_test = test_df["PCI"].values

    if initial_guess is None:
        initial_guess = [
            1658000.23,
            3728.61,
            9995.13,
            1287.65,
            0.01553,
            0.2066,
            0.2730,
            0.6887,
        ]

    lower_bounds = [1e3, 0, 0, 0, 1e-8, 0, 0, 0]
    upper_bounds = [1e8, 1e5, 1e5, 1e5, 10, 5, 5, 5]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        params, _ = curve_fit(
            pci_eq1,
            X_train,
            y_train,
            p0=initial_guess,
            bounds=(lower_bounds, upper_bounds),
            maxfev=800000,
        )

    y_pred_train = pci_eq1(X_train, *params)
    train_metrics, train_residuals = calc_metrics(y_train, y_pred_train)

    y_pred_test = pci_eq1(X_test, *params)
    test_metrics, test_residuals = calc_metrics(y_test, y_pred_test)

    param_table = pd.DataFrame(
        {
            "Parameter": ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"],
            "Value": params,
        }
    )

    train_result_df = train_df.copy()
    train_result_df["PCI_pred"] = y_pred_train
    train_result_df["Residual"] = train_residuals
    train_result_df["Split"] = "Train"

    test_result_df = test_df.copy()
    test_result_df["PCI_pred"] = y_pred_test
    test_result_df["Residual"] = test_residuals
    test_result_df["Split"] = "Test"

    result_df = pd.concat([train_result_df, test_result_df], axis=0, ignore_index=True)

    print("\n" + "=" * 90)
    print(f"Baseline model for {group_name}")
    print("=" * 90)
    print("PCI = 100 - 60 * (ESALs / rho)^beta")
    print(
        f"rho  = {params[0]:.6f} + {params[1]:.6f}*Mr "
        f"+ {params[2]:.6f}*SN + {params[3]:.6f}*ALT"
    )
    print(
        f"beta = {params[4]:.8f} * Mr^{params[5]:.6f} "
        f"* SN^{params[6]:.6f} * ALT^{params[7]:.6f}"
    )
    print_metrics(train_metrics, "Train metrics")
    print_metrics(test_metrics, "Test metrics")

    return {
        "model_name": group_name,
        "params": params,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "result_df": result_df,
        "param_table": param_table,
        "y_train_true": y_train,
        "y_train_pred": y_pred_train,
        "y_test_true": y_test,
        "y_test_pred": y_pred_test,
    }


def fit_eq1_baseline_with_dummy_train_test(
    df,
    group_name="Unified model with SAMI dummy",
    test_size=0.2,
    random_state=SEED,
    initial_guess=None,
):
    train_df, test_df = split_df_train_test(df, test_size=test_size, random_state=random_state)

    X_train = (
        train_df["Mr"].values,
        train_df["SN"].values,
        train_df["ALT"].values,
        train_df["ESALs"].values,
        train_df["SAMI"].values,
    )
    y_train = train_df["PCI"].values

    X_test = (
        test_df["Mr"].values,
        test_df["SN"].values,
        test_df["ALT"].values,
        test_df["ESALs"].values,
        test_df["SAMI"].values,
    )
    y_test = test_df["PCI"].values

    if initial_guess is None:
        initial_guess = [
            1658000.23,
            3728.61,
            9995.13,
            1287.65,
            0.01553,
            0.0,
            0.2066,
            0.2730,
            0.6887,
        ]

    lower_bounds = [1e3, 0, 0, 0, 1e-8, -10, 0, 0, 0]
    upper_bounds = [1e8, 1e5, 1e5, 1e5, 10, 10, 5, 5, 5]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        params, _ = curve_fit(
            pci_eq1_dummy,
            X_train,
            y_train,
            p0=initial_guess,
            bounds=(lower_bounds, upper_bounds),
            maxfev=800000,
        )

    y_pred_train = pci_eq1_dummy(X_train, *params)
    train_metrics, train_residuals = calc_metrics(y_train, y_pred_train)

    y_pred_test = pci_eq1_dummy(X_test, *params)
    test_metrics, test_residuals = calc_metrics(y_test, y_pred_test)

    param_table = pd.DataFrame(
        {
            "Parameter": ["p1", "p2", "p3", "p4", "p5", "p9", "p6", "p7", "p8"],
            "Value": params,
        }
    )

    train_result_df = train_df.copy()
    train_result_df["PCI_pred"] = y_pred_train
    train_result_df["Residual"] = train_residuals
    train_result_df["Split"] = "Train"

    test_result_df = test_df.copy()
    test_result_df["PCI_pred"] = y_pred_test
    test_result_df["Residual"] = test_residuals
    test_result_df["Split"] = "Test"

    result_df = pd.concat([train_result_df, test_result_df], axis=0, ignore_index=True)

    print("\n" + "=" * 90)
    print(f"Baseline model for {group_name}")
    print("=" * 90)
    print("PCI = 100 - 60 * (ESALs / rho)^beta")
    print(
        f"rho  = {params[0]:.6f} + {params[1]:.6f}*Mr "
        f"+ {params[2]:.6f}*SN + {params[3]:.6f}*ALT"
    )
    print(
        f"beta = ({params[4]:.8f} + {params[5]:.8f}*SAMI) "
        f"* Mr^{params[6]:.6f} * SN^{params[7]:.6f} * ALT^{params[8]:.6f}"
    )
    print("SAMI dummy: 0 = without SAMI, 1 = with SAMI")
    print_metrics(train_metrics, "Train metrics")
    print_metrics(test_metrics, "Test metrics")

    return {
        "model_name": group_name,
        "params": params,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "result_df": result_df,
        "param_table": param_table,
        "y_train_true": y_train,
        "y_train_pred": y_pred_train,
        "y_test_true": y_test,
        "y_test_pred": y_pred_test,
    }


def run_part_b():
    print("\n" + "=" * 100)
    print("PART B. BASELINE MODEL")
    print("=" * 100)

    df_type1 = load_and_clean_csv(FILE_TYPE1, rename=True, dropna=True)
    df_type2 = load_and_clean_csv(FILE_TYPE2, rename=True, dropna=True)

    df_type1 = df_type1.copy()
    df_type2 = df_type2.copy()
    df_type1["SAMI"] = 0
    df_type2["SAMI"] = 1
    df_combined = pd.concat([df_type1, df_type2], axis=0, ignore_index=True)

    baseline_t1 = fit_eq1_baseline_train_test(
        df_type1,
        "Type 1 (without SAMI)",
        test_size=0.2,
        random_state=SEED,
    )
    baseline_t2 = fit_eq1_baseline_train_test(
        df_type2,
        "Type 2 (with SAMI)",
        test_size=0.2,
        random_state=SEED,
    )
    baseline_dummy = fit_eq1_baseline_with_dummy_train_test(
        df_combined,
        "Unified baseline with SAMI dummy",
        test_size=0.2,
        random_state=SEED,
    )

    summary_baseline = pd.DataFrame(
        [
            {"Group": "Type 1 (without SAMI)", "Split": "Train", **baseline_t1["train_metrics"]},
            {"Group": "Type 1 (without SAMI)", "Split": "Test", **baseline_t1["test_metrics"]},
            {"Group": "Type 2 (with SAMI)", "Split": "Train", **baseline_t2["train_metrics"]},
            {"Group": "Type 2 (with SAMI)", "Split": "Test", **baseline_t2["test_metrics"]},
            {
                "Group": "Unified baseline with SAMI dummy",
                "Split": "Train",
                **baseline_dummy["train_metrics"],
            },
            {
                "Group": "Unified baseline with SAMI dummy",
                "Split": "Test",
                **baseline_dummy["test_metrics"],
            },
        ]
    )

    print("\n" + "=" * 90)
    print("Baseline comparison summary")
    print("=" * 90)
    print(summary_baseline)

    baseline_t1["result_df"].to_csv("/content/baseline_eq1_type1_results.csv", index=False)
    baseline_t2["result_df"].to_csv("/content/baseline_eq1_type2_results.csv", index=False)
    baseline_dummy["result_df"].to_csv("/content/baseline_eq1_dummy_combined_results.csv", index=False)

    baseline_t1["param_table"].to_csv("/content/baseline_eq1_type1_parameters.csv", index=False)
    baseline_t2["param_table"].to_csv("/content/baseline_eq1_type2_parameters.csv", index=False)
    baseline_dummy["param_table"].to_csv("/content/baseline_eq1_dummy_parameters.csv", index=False)

    summary_baseline.to_csv("/content/baseline_eq1_summary.csv", index=False)

    print("\nSaved files:")
    print("/content/baseline_eq1_type1_results.csv")
    print("/content/baseline_eq1_type2_results.csv")
    print("/content/baseline_eq1_dummy_combined_results.csv")
    print("/content/baseline_eq1_type1_parameters.csv")
    print("/content/baseline_eq1_type2_parameters.csv")
    print("/content/baseline_eq1_dummy_parameters.csv")
    print("/content/baseline_eq1_summary.csv")

    plot_pred_vs_true(
        baseline_t1["y_test_true"],
        baseline_t1["y_test_pred"],
        "Type 1 (without SAMI) - Baseline Eq. (1) Test",
        save_path="/content/baseline_type1_fit.png",
    )

    plot_pred_vs_true(
        baseline_t2["y_test_true"],
        baseline_t2["y_test_pred"],
        "Type 2 (with SAMI) - Baseline Eq. (1) Test",
        save_path="/content/baseline_type2_fit.png",
    )

    plot_pred_vs_true(
        baseline_dummy["y_test_true"],
        baseline_dummy["y_test_pred"],
        "Unified baseline with SAMI dummy - Test",
        save_path="/content/baseline_dummy_fit.png",
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.scatter(baseline_t1["y_test_true"], baseline_t1["y_test_pred"], alpha=0.8)
    mn = min(baseline_t1["y_test_true"].min(), baseline_t1["y_test_pred"].min())
    mx = max(baseline_t1["y_test_true"].max(), baseline_t1["y_test_pred"].max())
    ax.plot([mn, mx], [mn, mx], "k--")
    ax.set_xlabel("Observed PCI")
    ax.set_ylabel("Predicted PCI")
    ax.set_title("Type 1 test")

    ax = axes[1]
    ax.scatter(baseline_t2["y_test_true"], baseline_t2["y_test_pred"], alpha=0.8)
    mn = min(baseline_t2["y_test_true"].min(), baseline_t2["y_test_pred"].min())
    mx = max(baseline_t2["y_test_true"].max(), baseline_t2["y_test_pred"].max())
    ax.plot([mn, mx], [mn, mx], "k--")
    ax.set_xlabel("Observed PCI")
    ax.set_ylabel("Predicted PCI")
    ax.set_title("Type 2 test")

    ax = axes[2]
    ax.scatter(baseline_dummy["y_test_true"], baseline_dummy["y_test_pred"], alpha=0.8)
    mn = min(baseline_dummy["y_test_true"].min(), baseline_dummy["y_test_pred"].min())
    mx = max(baseline_dummy["y_test_true"].max(), baseline_dummy["y_test_pred"].max())
    ax.plot([mn, mx], [mn, mx], "k--")
    ax.set_xlabel("Observed PCI")
    ax.set_ylabel("Predicted PCI")
    ax.set_title("Unified dummy test")

    plt.tight_layout()
    plt.savefig("/content/baseline_fit_combined.png", dpi=300, bbox_inches="tight")
    print("Saved: /content/baseline_fit_combined.png")
    plt.show()

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


def get_loss_fn(loss_name="huber"):
    if loss_name.lower() == "mse":
        return nn.MSELoss()
    elif loss_name.lower() == "huber":
        return nn.HuberLoss(delta=1.0)
    else:
        raise ValueError("loss_name must be 'mse' or 'huber'")


def train_model(
    model,
    train_loader,
    loss_name="huber",
    lr=1e-3,
    max_epochs=400,
    verbose_every=20,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = get_loss_fn(loss_name)

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


def build_features_train_test(df, test_size=0.2, random_state=SEED):
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

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()


def run_direct_model(data_dict, group_name="Type 1", loss_name="huber", max_epochs=300):
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

    print(f"\nTraining direct model for {group_name} with {loss_name.upper()} loss...")
    model, history = train_model(
        model,
        train_loader,
        loss_name=loss_name,
        lr=1e-3,
        max_epochs=max_epochs,
        verbose_every=20,
    )

    y_true, y_pred = predict_model(model, test_loader)
    metrics, _ = calc_metrics(y_true, y_pred)

    print_metrics(metrics, f"Direct model metrics for {group_name} with {loss_name.upper()} loss")

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
    loss_name="huber",
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

    print(f"\nPretraining on Type 1 with {loss_name.upper()} loss...")
    model, src_history = train_model(
        model,
        src_train_loader,
        loss_name=loss_name,
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

    print(f"\nFine-tuning on Type 2 with {loss_name.upper()} loss...")
    model, tgt_history = train_model(
        model,
        tgt_train_loader,
        loss_name=loss_name,
        lr=5e-4,
        max_epochs=finetune_epochs,
        verbose_every=20,
    )

    y_true, y_pred = predict_model(model, tgt_test_loader)
    metrics, _ = calc_metrics(y_true, y_pred)

    print_metrics(metrics, f"Transfer learning metrics (Type 1 -> Type 2) with {loss_name.upper()} loss")

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



def run_explainable_models(df, group_name="Type 1", test_size=0.2, random_state=SEED):
    print("\n" + "=" * 100)
    print(f"Explainable AI analysis for {group_name}")
    print("=" * 100)

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    print("\nRunning symbolic regression...")
    sym = SymbolicRegressor(
        population_size=4000,
        generations=40,
        tournament_size=20,
        stopping_criteria=0.001,
        const_range=(-5.0, 5.0),
        init_depth=(2, 5),
        init_method="half and half",
        function_set=("add", "sub", "mul", "div", "log"),
        metric="mse",
        parsimony_coefficient=0.003,
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        p_point_replace=0.05,
        max_samples=0.9,
        feature_names=FEATURE_COLS,
        verbose=1,
        random_state=random_state,
        n_jobs=1,
    )
    sym.fit(X_train, y_train)
    y_pred_sym = sym.predict(X_test)
    sym_metrics, _ = calc_metrics(y_test, y_pred_sym)

    print("\nSymbolic regression equation:")
    print(sym._program)
    print_metrics(sym_metrics, "Symbolic regression metrics")

    base_point = np.mean(X_train, axis=0)
    sym_sens = sensitivity_check(sym, base_point, FEATURE_COLS, step=0.05)
    print("\nSymbolic regression sensitivity check")
    print(sym_sens.to_string(index=False))

    plot_pred_vs_true(y_test, y_pred_sym, f"{group_name} - Symbolic Regression")
    plot_residuals(y_test, y_pred_sym, f"{group_name} - Symbolic Regression Residuals")

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

    tree_sens = sensitivity_check(best_tree, base_point, FEATURE_COLS, step=0.05)
    print("\nDecision tree sensitivity check")
    print(tree_sens.to_string(index=False))

    # PART D one-way sensitivity curve generation has been removed.
    # Therefore, no /content/sensitivity_outputs_xai folder or sensitivity curve figures are generated.

    plot_pred_vs_true(y_test, y_pred_tree, f"{group_name} - Decision Tree")
    plot_residuals(y_test, y_pred_tree, f"{group_name} - Decision Tree Residuals")

    plt.figure(figsize=(16, 8))
    plot_tree(best_tree, feature_names=FEATURE_COLS, filled=True, rounded=True, fontsize=9)
    plt.title(f"{group_name} - Decision Tree")
    plt.show()

    summary = pd.DataFrame(
        [
            {"Group": group_name, "Model": "SymbolicRegression", **sym_metrics},
            {"Group": group_name, "Model": "DecisionTree", **tree_metrics},
        ]
    )

    sym_pred_df = pd.DataFrame(
        {
            "Observed_PCI": y_test,
            "Predicted_PCI": y_pred_sym,
            "Residual": y_test - y_pred_sym,
        }
    )

    tree_pred_df = pd.DataFrame(
        {
            "Observed_PCI": y_test,
            "Predicted_PCI": y_pred_tree,
            "Residual": y_test - y_pred_tree,
        }
    )

    safe_group = group_name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")

    summary.to_csv(f"/content/{safe_group}_xai_summary.csv", index=False)
    sym_pred_df.to_csv(f"/content/{safe_group}_symbolic_predictions.csv", index=False)
    tree_pred_df.to_csv(f"/content/{safe_group}_tree_predictions.csv", index=False)
    sym_sens.to_csv(f"/content/{safe_group}_symbolic_sensitivity.csv", index=False)
    tree_sens.to_csv(f"/content/{safe_group}_tree_sensitivity.csv", index=False)

    with open(f"/content/{safe_group}_symbolic_equation.txt", "w") as f:
        f.write(str(sym._program))

    with open(f"/content/{safe_group}_tree_rules.txt", "w") as f:
        f.write(tree_rules)

    print(f"\nSaved XAI outputs for {group_name}")

    return {
        "summary": summary,
        "symbolic_model": sym,
        "tree_model": best_tree,
        "symbolic_metrics": sym_metrics,
        "tree_metrics": tree_metrics,
        "symbolic_equation": str(sym._program),
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

    data_t1 = build_features_train_test(df_type1, test_size=0.2, random_state=SEED)
    data_t2 = build_features_train_test(df_type2, test_size=0.2, random_state=SEED)

    # ------------------------------------------------------------
    # Save model explanation diagrams for manuscript use
    # ------------------------------------------------------------
    input_len = data_t1["X_train"].shape[1]

    save_residual_block_diagram(
        out_path="/content/model_diagrams/residual_block_1d.png",
    )

    save_resnet_1dcnn_architecture(
        input_len=input_len,
        out_path="/content/model_diagrams/resnet_1dcnn_architecture.png",
    )

    save_transfer_learning_workflow(
        out_path="/content/model_diagrams/transfer_learning_workflow.png",
    )

    save_symbolic_regression_workflow(
        out_path="/content/model_diagrams/symbolic_regression_workflow.png",
    )

    save_decision_tree_workflow(
        out_path="/content/model_diagrams/decision_tree_workflow.png",
    )

    print("Expanded feature dimension for Type 1:", data_t1["X_train"].shape[1])
    print("Expanded feature dimension for Type 2:", data_t2["X_train"].shape[1])

    results_summary = []
    saved_predictions = {}
    transfer_results = {}

    for loss_name in ["mse", "huber"]:
        res_t1_direct = run_direct_model(
            data_t1,
            group_name="Type 1",
            loss_name=loss_name,
            max_epochs=300,
        )
        plot_history(res_t1_direct["history"], f"Type 1 direct - {loss_name.upper()} loss")
        plot_pred_vs_true(
            res_t1_direct["y_true"],
            res_t1_direct["y_pred"],
            f"Type 1 direct - {loss_name.upper()} loss",
        )
        results_summary.append(
            {"Model": "Type1_direct", "Loss": loss_name.upper(), **res_t1_direct["metrics"]}
        )
        saved_predictions[f"Type1_direct_{loss_name}"] = (
            res_t1_direct["y_true"],
            res_t1_direct["y_pred"],
        )

        res_t2_direct = run_direct_model(
            data_t2,
            group_name="Type 2",
            loss_name=loss_name,
            max_epochs=300,
        )
        plot_history(res_t2_direct["history"], f"Type 2 direct - {loss_name.upper()} loss")
        plot_pred_vs_true(
            res_t2_direct["y_true"],
            res_t2_direct["y_pred"],
            f"Type 2 direct - {loss_name.upper()} loss",
        )
        results_summary.append(
            {"Model": "Type2_direct", "Loss": loss_name.upper(), **res_t2_direct["metrics"]}
        )
        saved_predictions[f"Type2_direct_{loss_name}"] = (
            res_t2_direct["y_true"],
            res_t2_direct["y_pred"],
        )

    for loss_name in ["mse", "huber"]:
        res_tf = run_transfer_t1_to_t2(
            data_t1,
            data_t2,
            loss_name=loss_name,
            pretrain_epochs=300,
            finetune_epochs=250,
        )
        transfer_results[loss_name] = res_tf
        plot_history(res_tf["pretrain_history"], f"Pretraining on Type 1 - {loss_name.upper()} loss")
        plot_history(res_tf["finetune_history"], f"Fine-tuning on Type 2 - {loss_name.upper()} loss")
        plot_pred_vs_true(
            res_tf["y_true"],
            res_tf["y_pred"],
            f"Transfer T1→T2 - {loss_name.upper()} loss",
        )
        results_summary.append(
            {"Model": "Transfer_T1_to_T2", "Loss": loss_name.upper(), **res_tf["metrics"]}
        )
        saved_predictions[f"Transfer_T1_to_T2_{loss_name}"] = (
            res_tf["y_true"],
            res_tf["y_pred"],
        )

    summary_dl = pd.DataFrame(results_summary)
    print("\n" + "=" * 100)
    print("Model comparison summary")
    print("=" * 100)
    print(summary_dl)

    summary_dl.to_csv("/content/resnet_1dcnn_transfer_summary.csv", index=False)
    print("\nSaved: /content/resnet_1dcnn_transfer_summary.csv")

    # PART C-1 transfer-learning sensitivity analysis has been removed.
    # Therefore, no /content/sensitivity_outputs_dl folder or
    # /content/transfer_t1_to_t2_huber_sensitivity_summary.csv file is generated.

    y_true, y_pred = saved_predictions["Type2_direct_huber"]
    save_prediction_csv(y_true, y_pred, "/content/type2_direct_huber_predictions.csv")

    y_true, y_pred = saved_predictions["Transfer_T1_to_T2_huber"]
    save_prediction_csv(y_true, y_pred, "/content/transfer_t1_to_t2_huber_predictions.csv")

    df_direct = pd.read_csv("/content/type2_direct_huber_predictions.csv")
    df_transfer = pd.read_csv("/content/transfer_t1_to_t2_huber_predictions.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    ax = axes[0]
    ax.scatter(df_direct["Observed_PCI"], df_direct["Predicted_PCI"], alpha=0.8)
    mn = min(df_direct["Observed_PCI"].min(), df_direct["Predicted_PCI"].min())
    mx = max(df_direct["Observed_PCI"].max(), df_direct["Predicted_PCI"].max())
    ax.plot([mn, mx], [mn, mx], "k--")
    ax.set_xlabel("Observed PCI")
    ax.set_ylabel("Predicted PCI")
    ax.set_title("Direct Type 2 model")

    ax = axes[1]
    ax.scatter(df_transfer["Observed_PCI"], df_transfer["Predicted_PCI"], alpha=0.8)
    mn = min(df_transfer["Observed_PCI"].min(), df_transfer["Predicted_PCI"].min())
    mx = max(df_transfer["Observed_PCI"].max(), df_transfer["Predicted_PCI"].max())
    ax.plot([mn, mx], [mn, mx], "k--")
    ax.set_xlabel("Observed PCI")
    ax.set_ylabel("Predicted PCI")
    ax.set_title("Transfer learning")

    plt.tight_layout()
    plt.savefig("/content/transfer_fit_combined.png", dpi=300, bbox_inches="tight")
    print("Saved: /content/transfer_fit_combined.png")
    plt.show()

    plot_pred_vs_true(
        df_direct["Observed_PCI"].values,
        df_direct["Predicted_PCI"].values,
        "Direct Type 2 model",
        save_path="/content/type2_direct_huber_fit.png",
    )

    plot_pred_vs_true(
        df_transfer["Observed_PCI"].values,
        df_transfer["Predicted_PCI"].values,
        "Transfer learning",
        save_path="/content/transfer_t1_to_t2_huber_fit.png",
    )

    print("\n" + "=" * 100)
    print("PART D. EXPLAINABLE AI")
    print("=" * 100)

    res_xai_t1 = run_explainable_models(df_type1, group_name="Type 1 (without SAMI)")
    res_xai_t2 = run_explainable_models(df_type2, group_name="Type 2 (with SAMI)")

    combined_summary_xai = pd.concat(
        [res_xai_t1["summary"], res_xai_t2["summary"]],
        ignore_index=True,
    )

    print("\n" + "=" * 100)
    print("Combined explainable AI summary")
    print("=" * 100)
    print(combined_summary_xai)

    combined_summary_xai.to_csv("/content/explainable_ai_combined_summary.csv", index=False)
    print("\nSaved: /content/explainable_ai_combined_summary.csv")

    print("\n" + "=" * 100)
    print("ALL DONE")
    print("=" * 100)

    print("\nMain output files:")
    output_files = [
        "/content/descriptive_stats_without_SAMI.csv",
        "/content/descriptive_stats_with_SAMI.csv",
        "/content/descriptive_stats_combined.csv",
        "/content/descriptive_stats_combined_rounded.csv",
        "/content/baseline_eq1_type1_results.csv",
        "/content/baseline_eq1_type2_results.csv",
        "/content/baseline_eq1_summary.csv",
        "/content/baseline_fit_combined.png",
        "/content/baseline_type1_fit.png",
        "/content/baseline_type2_fit.png",
        "/content/resnet_1dcnn_transfer_summary.csv",
        "/content/type2_direct_huber_predictions.csv",
        "/content/transfer_t1_to_t2_huber_predictions.csv",
        "/content/transfer_fit_combined.png",
        "/content/type2_direct_huber_fit.png",
        "/content/transfer_t1_to_t2_huber_fit.png",
        "/content/explainable_ai_combined_summary.csv",
    ]


    for f in output_files:
        if os.path.exists(f):
            print(f)


if __name__ == "__main__":
    main()
