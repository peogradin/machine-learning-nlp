# %%
import os
import json
import glob
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from datasets import load_from_disk

import inspect

def set_plot_style():
    plt.rcParams.update({
        "figure.figsize": (8, 4.5),
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "font.size": 11,
    })

# %% --------------------------------------------
# Training history evaluation and plotting
# -----------------------------------------------

MODEL_DIR_TO_LABEL = {
    "teacher_bert_base": "Teacher (BERT-base)",
    # "student_baseline_distilbert": "Baseline (DistilBERT)",
    "student_baseline_bertmini": "Baseline (BERT-mini)",
    "distilled_model": "Distilled (BERT-mini)",
}

labels = ["joy", "sadness", "anger", "fear", "love", "surprise"]

@dataclass
class RunKey:
    run_id: str
    seed: Optional[int]
    frac: Optional[float]

def parse_run_id(run_id: str) -> RunKey:
    # expects run_id like "seed101_frac0.1" but gracefully handles other IDs
    seed = None
    frac = None
    try:
        parts = run_id.split("_")
        for p in parts:
            if p.startswith("seed"):
                seed = int(p.replace("seed", ""))
            if p.startswith("frac"):
                frac = float(p.replace("frac", ""))
    except Exception:
        pass
    return RunKey(run_id=run_id, seed=seed, frac=frac)

def safe_read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None
    
def find_metrics_files(outputs_dir: str) -> List[str]:
    # Find metrics.json under each model folder inside each run folder
    pattern = os.path.join(outputs_dir, "seed*_frac*", "*", "metrics.json")
    return sorted(glob.glob(pattern))

def model_label_from_path(metrics_path: str) -> Optional[str]:
    # .../seedX_fracY/<model_dir>/metrics.json
    model_dir = os.path.basename(os.path.dirname(metrics_path))
    return MODEL_DIR_TO_LABEL.get(model_dir)

def run_id_from_path(metrics_path: str) -> str:
    # .../outputs/<run_id>/<model_dir>/metrics.json
    return os.path.basename(os.path.dirname(os.path.dirname(metrics_path)))

def extract_metrics(metrics_json: Dict[str, Any], split: str) -> Dict[str, Any]:
    # Your metrics.json is {"val": {...}, "test": {...}}
    out = {}
    d = metrics_json.get(split, {})
    # standard Trainer keys
    out[f"{split}_accuracy"] = d.get("eval_accuracy")
    out[f"{split}_f1"] = d.get("eval_f1")
    out[f"{split}_loss"] = d.get("eval_loss")
    out[f"{split}_runtime"] = d.get("eval_runtime")
    # sometimes useful:
    out[f"{split}_samples_per_sec"] = d.get("eval_samples_per_second")
    return out

def read_log_history_for_curve(model_dir: str) -> pd.DataFrame:
    """
    Reads log_history.json and returns a dataframe with columns:
    epoch, eval_accuracy (if present), loss (if present)
    """
    path = os.path.join(model_dir, "log_history.json")
    data = safe_read_json(path)
    if not data:
        return pd.DataFrame()

    rows = []
    for item in data:
        if not isinstance(item, dict):
            continue
        # Keep entries that have an epoch
        if "epoch" not in item:
            continue
        rows.append({
            "epoch": item.get("epoch"),
            "loss": item.get("loss"),
            "eval_accuracy": item.get("eval_accuracy"),
            "eval_f1": item.get("eval_f1"),
            "eval_loss": item.get("eval_loss"),
        })

    df = pd.DataFrame(rows)
    # Remove NaNs-only columns
    if not df.empty:
        df = df.sort_values("epoch")
    return df

def plot_training_history(outputs_dir: str = "./outputs", results_dir: str = "./results", exclude_teacher: bool = True) -> None:
    metrics_files = find_metrics_files(outputs_dir)
    if not metrics_files:
        raise FileNotFoundError(f"No metrics.json found under {outputs_dir}/seed*_frac*/")

    records = []
    curves = []  # for learning curve plot

    for mp in metrics_files:
        label = model_label_from_path(mp)
        if label is None or (exclude_teacher and label == "Teacher (BERT-base)"):
            continue

        run_id = run_id_from_path(mp)
        runkey = parse_run_id(run_id)

        mj = safe_read_json(mp)
        if not mj:
            continue

        rec = {
            "run_id": run_id,
            "seed": runkey.seed,
            "train_fraction": runkey.frac,
            "model": label,
            "model_dir": os.path.dirname(mp),
        }
        rec.update(extract_metrics(mj, "val"))
        rec.update(extract_metrics(mj, "test"))
        records.append(rec)

        # log history (optional)
        df_curve = read_log_history_for_curve(os.path.dirname(mp))
        if not df_curve.empty:
            df_curve["run_id"] = run_id
            df_curve["train_fraction"] = runkey.frac
            df_curve["model"] = label
            curves.append(df_curve)

    df = pd.DataFrame(records)

    # Sort nicely
    df = df.sort_values(["train_fraction", "model"], ascending=[True, True])

    # Write summary tables
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "summary.csv")
    md_path = os.path.join(results_dir, "summary.md")
    df.to_csv(csv_path, index=False)

    # Markdown table (compact)
    cols = [
        "train_fraction", "model",
        "val_accuracy", "val_f1",
        "test_accuracy", "test_f1",
        "val_runtime",
    ]
    df_md = df[cols].copy()
    df_md.to_markdown(md_path, index=False)

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")

    # ---- Plot: accuracy vs fraction (test) ----
    plt.figure()
    for model in sorted(df["model"].unique()):
        sub = df[df["model"] == model].dropna(subset=["train_fraction", "test_accuracy"])
        if sub.empty:
            continue
        sub = sub.sort_values("train_fraction")
        plt.plot(sub["train_fraction"], sub["test_accuracy"], marker="o", label=model)
    plt.xlabel("Train fraction")
    plt.ylabel("Test accuracy")
    plt.xticks([0.1, 0.25, 0.5, 0.75, 1.0])
    plt.title("Test accuracy vs train fraction")
    plt.legend()
    resultp = os.path.join(results_dir, "plot_accuracy_vs_fraction.png")
    plt.savefig(resultp, dpi=200, bbox_inches="tight")
    print(f"Wrote: {resultp}")

    # ---- Plot: f1 vs fraction (test) ----
    plt.figure()
    for model in sorted(df["model"].unique()):
        sub = df[df["model"] == model].dropna(subset=["train_fraction", "test_f1"])
        if sub.empty:
            continue
        sub = sub.sort_values("train_fraction")
        plt.plot(sub["train_fraction"], sub["test_f1"], marker="o", label=model)
    plt.xlabel("Train fraction")
    plt.ylabel("Test F1 (weighted)")
    plt.title("Test F1 vs train fraction")
    plt.xticks([0.1, 0.25, 0.5, 0.75, 1.0])
    plt.legend()
    resultp = os.path.join(results_dir, "plot_f1_vs_fraction.png")
    plt.savefig(resultp, dpi=200, bbox_inches="tight")
    print(f"Wrote: {resultp}")

    # ---- Plot: learning curves (val accuracy vs epoch) ----
    if curves:
        dfc = pd.concat(curves, ignore_index=True)
        # plot only eval points that exist
        dfc = dfc.dropna(subset=["epoch", "eval_accuracy"])
        if not dfc.empty:
            fracs = sorted(dfc["train_fraction"].dropna().unique())
            models_curves = sorted(dfc["model"].dropna().unique())
            cmap = plt.get_cmap("tab10")
            frac_colors = {frac: cmap(i / max(len(fracs), 1)) for i, frac in enumerate(fracs)}
            alpha_levels = np.linspace(1.0, 0.5, num=len(models_curves)) if models_curves else []
            model_alpha = {model: alpha_levels[i] for i, model in enumerate(models_curves)}

            plt.figure()
            # separate lines by (fraction, model)
            for (frac, model), sub in dfc.groupby(["train_fraction", "model"]):
                sub = sub.sort_values("epoch")
                color = frac_colors.get(frac, None)
                alpha = model_alpha.get(model, 1.0)
                marker = "s" if model == "Baseline (BERT-mini)" else "o"
                plt.plot(sub["epoch"], sub["eval_accuracy"], marker=marker,
                         label=f"{model} (frac={frac})", color=color, alpha=alpha)
            plt.xlabel("Epoch")
            plt.ylabel("Validation accuracy")
            plt.title("Validation accuracy learning curves")
            plt.xticks(np.arange(2, dfc["epoch"].max() + 1, 2))
            plt.legend(fontsize=8)
            resultp = os.path.join(results_dir, "plot_learning_curves.png")
            plt.savefig(resultp, dpi=200, bbox_inches="tight")
            print(f"Wrote: {resultp}")
        else:
            print("No eval_accuracy found in log_history.json files; skipping learning curve plot.")
    else:
        print("No log_history.json files found; skipping learning curve plot.")

# %% --------------------------------------------
# Predictions-based evaluation (no model inference)
# -----------------------------------------------

def find_prediction_files(results_dir: str, split: str = "test") -> List[str]:
    # e.g. .../seed101_frac0.1/<model_dir>/test_predictions.jsonl
    pattern = os.path.join(results_dir, "seed*_frac*", "*", f"{split}_predictions.jsonl")
    return sorted(glob.glob(pattern))

def model_dir_from_pred_path(pred_path: str) -> str:
    # .../<run_id>/<model_dir>/<split>_predictions.jsonl
    return os.path.basename(os.path.dirname(pred_path))

def load_predictions_jsonl(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    # Expected columns: idx, true, pred, logits(optional)
    return df

def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exps = np.exp(logits)
    return exps / exps.sum(axis=1, keepdims=True)


def entropy_from_probs(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(probs, eps, 1.0)
    return -(p * np.log(p)).sum(axis=1)

def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm

def per_class_prf(cm: np.ndarray) -> pd.DataFrame:
    # rows=true, cols=pred
    num_classes = cm.shape[0]
    out = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1c  = (2*prec*rec/(prec+rec)) if (prec+rec) > 0 else 0.0
        support = cm[c, :].sum()
        out.append({"class": labels[c] if c < len(labels) else str(c),
                    "precision": prec, "recall": rec, "f1": f1c, "support": int(support)})
    return pd.DataFrame(out)

def agreement(y_a: np.ndarray, y_b: np.ndarray) -> float:
    return float((y_a == y_b).mean()) if len(y_a) else float("nan")

def kl_divergence_rowwise(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # KL(p || q)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return (p * (np.log(p) - np.log(q))).sum(axis=1)

def deep_predictions_analysis(outputs_dir: str = "./outputs", results_dir: str = "./results", split: str = "test") -> None:
    pred_files = find_prediction_files(outputs_dir, split=split)
    if not pred_files:
        raise FileNotFoundError(f"No {split}_predictions.jsonl found under {outputs_dir}/seed*_frac*/")

    # Collect per-run/model stats
    rows_summary = []

    # We'll also compute teacher comparisons within each run_id (needs teacher present)
    # We'll store per (run_id) teacher df for easy joins
    teacher_by_run: Dict[str, pd.DataFrame] = {}

    # First pass: load all predictions
    preds_by_run_model: Dict[tuple, pd.DataFrame] = {}
    for pth in pred_files:
        run_id = os.path.basename(os.path.dirname(os.path.dirname(pth)))
        model_dir = model_dir_from_pred_path(pth)

        # Skip anything we don't want to include (e.g. distilbert, dataset folders)
        if model_dir not in MODEL_DIR_TO_LABEL:
            continue

        model_label = MODEL_DIR_TO_LABEL[model_dir]

        dfp = load_predictions_jsonl(pth)
        # Ensure ordering by idx for safe alignment
        if "idx" in dfp.columns:
            dfp = dfp.sort_values("idx").reset_index(drop=True)

        preds_by_run_model[(run_id, model_dir)] = dfp

        if model_dir == "teacher_bert_base":
            teacher_by_run[run_id] = dfp

    # Second pass: compute metrics per model
    for (run_id, model_dir), dfp in preds_by_run_model.items():
        runkey = parse_run_id(run_id)
        model_label = MODEL_DIR_TO_LABEL.get(model_dir, model_dir)

        y_true = dfp["true"].to_numpy()
        y_pred = dfp["pred"].to_numpy()
        num_classes = len(labels)

        cm = confusion_matrix_np(y_true, y_pred, num_classes=num_classes)
        df_prf = per_class_prf(cm)

        acc = float((y_true == y_pred).mean())

        # Per-class accuracy (i.e., recall) + support fraction
        support = cm.sum(axis=1).astype(np.int64)              # true counts per class
        tp = np.diag(cm).astype(np.int64)
        per_class_acc = np.divide(tp, support, out=np.zeros_like(tp, dtype=float), where=support > 0)

        total = int(support.sum())
        support_frac = np.divide(support, total, out=np.zeros_like(support, dtype=float), where=total > 0)

        # Add to df_prf for saving/plotting
        df_prf["accuracy"] = per_class_acc
        df_prf["support_frac"] = support_frac

        # Confidence/entropy if logits exist
        mean_conf = None
        mean_entropy = None
        if "logits" in dfp.columns and dfp["logits"].notna().all():
            logits = np.array(dfp["logits"].tolist(), dtype=np.float32)
            probs = softmax_np(logits)
            conf = probs.max(axis=1)
            ent = entropy_from_probs(probs)
            mean_conf = float(conf.mean())
            mean_entropy = float(ent.mean())

        # Agreement vs teacher (if available and same run)
        agree_teacher = None
        teacher_adv = None  # teacher correct when student wrong
        student_adv = None  # student correct when teacher wrong
        mean_kl_to_teacher = None

        if run_id in teacher_by_run and model_dir != "teacher_bert_base":
            dft = teacher_by_run[run_id]
            # align by idx if possible
            if "idx" in dfp.columns and "idx" in dft.columns:
                dft = dft.sort_values("idx").reset_index(drop=True)

            y_t_pred = dft["pred"].to_numpy()
            agree_teacher = agreement(y_pred, y_t_pred)

            teacher_correct = (dft["pred"].to_numpy() == dft["true"].to_numpy())
            student_correct = (y_pred == y_true)
            student_wrong = ~student_correct
            teacher_wrong = ~teacher_correct

            # Of the cases where student is wrong, how often teacher is correct?
            if student_wrong.any():
                teacher_adv = float(teacher_correct[student_wrong].mean())
            else:
                teacher_adv = float("nan")

            # Of the cases where teacher is wrong, how often student is correct?
            if teacher_wrong.any():
                student_adv = float(student_correct[teacher_wrong].mean())
            else:
                student_adv = float("nan")

            # KL(student || teacher) if both have logits
            if ("logits" in dfp.columns and "logits" in dft.columns
                and dfp["logits"].notna().all() and dft["logits"].notna().all()):
                s_logits = np.array(dfp["logits"].tolist(), dtype=np.float32)
                t_logits = np.array(dft["logits"].tolist(), dtype=np.float32)
                s_probs = softmax_np(s_logits)
                t_probs = softmax_np(t_logits)
                kl = kl_divergence_rowwise(s_probs, t_probs)
                mean_kl_to_teacher = float(kl.mean())

        # Save confusion matrix plot
        cm_fig = os.path.join(results_dir, run_id, model_dir, f"{split}_confusion_matrix.png")
        os.makedirs(os.path.dirname(cm_fig), exist_ok=True)
        plt.figure()
        plt.imshow(cm, interpolation="nearest")
        plt.title(f"{model_label} - {split} confusion matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks(range(num_classes), labels, rotation=45, ha="right")
        plt.yticks(range(num_classes), labels)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(cm_fig, dpi=200)
        plt.close()

        # Save per-class accuracy bar plot (with support shown in x tick labels)
        acc_fig = os.path.join(results_dir, run_id, model_dir, f"{split}_per_class_accuracy.png")

        tick_labels = []
        for cls, n, frac in zip(df_prf["class"], support.tolist(), df_prf["support_frac"].tolist()):
            tick_labels.append(f"{cls}\n(n={n}, {100*frac:.1f}%)")

        plt.figure()
        plt.bar(range(num_classes), df_prf["accuracy"].to_numpy())
        plt.ylim(0.0, 1.0)
        plt.grid(True, axis="y", alpha=0.3)
        plt.title(f"{model_label} - {split} per-class accuracy")
        plt.xlabel("Class (support in test/val)")
        plt.ylabel("Accuracy per class")
        plt.xticks(range(num_classes), tick_labels, rotation=0, ha="center")
        plt.tight_layout()
        plt.savefig(acc_fig, dpi=200)
        plt.close()

        # Save per-class table
        prf_csv = os.path.join(results_dir, run_id, model_dir, f"{split}_per_class_metrics.csv")
        df_prf.to_csv(prf_csv, index=False)

        rows_summary.append({
            "run_id": run_id,
            "seed": runkey.seed,
            "train_fraction": runkey.frac,
            "split": split,
            "model_dir": model_dir,
            "model": model_label,
            "acc_from_preds": acc,
            "mean_conf": mean_conf,
            "mean_entropy": mean_entropy,
            "agree_with_teacher": agree_teacher,
            "teacher_correct_when_student_wrong": teacher_adv,
            "student_correct_when_teacher_wrong": student_adv,
            "mean_KL_student_to_teacher": mean_kl_to_teacher,
            "confusion_matrix_png": cm_fig,
            "per_class_accuracy_png": acc_fig,
            "per_class_csv": prf_csv,
        })

    df_sum = pd.DataFrame(rows_summary).sort_values(["train_fraction", "model"])
    out_csv = os.path.join(results_dir, f"{split}_preds_summary.csv")
    df_sum.to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv}")

    # ---- Per-run grouped bar chart: per-class accuracy for Teacher vs Baseline(BERT-mini) vs Distilled(BERT-mini) ----
    # Build from loaded predictions (preds_by_run_model) so we don't need any extra files
    for run_id in sorted({rk[0] for rk in preds_by_run_model.keys()}):
        # only if teacher exists
        if (run_id, "teacher_bert_base") not in preds_by_run_model:
            continue

        # Collect per-class accuracies for the models we care about
        model_order = ["teacher_bert_base", "student_baseline_bertmini", "distilled_model"]
        rows = []
        support = None

        for md in model_order:
            if (run_id, md) not in preds_by_run_model:
                continue
            dfp = preds_by_run_model[(run_id, md)]
            y_true = dfp["true"].to_numpy()
            y_pred = dfp["pred"].to_numpy()
            cm = confusion_matrix_np(y_true, y_pred, num_classes=len(labels))

            sup = cm.sum(axis=1).astype(np.int64)
            tp = np.diag(cm).astype(np.int64)
            acc_c = np.divide(tp, sup, out=np.zeros_like(tp, dtype=float), where=sup > 0)

            if support is None:
                support = sup

            rows.append({
                "model": MODEL_DIR_TO_LABEL[md],
                "acc": acc_c,
            })

        if not rows or support is None:
            continue

        # Plot grouped bars
        fig_path = os.path.join(results_dir, run_id, f"{split}_per_label_accuracy_grouped.png")
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)

        x = np.arange(len(labels))
        width = 0.25 if len(rows) >= 3 else 0.35

        plt.figure(figsize=(10, 4.5))
        for i, r in enumerate(rows):
            plt.bar(x + (i - (len(rows)-1)/2)*width, r["acc"], width=width, label=r["model"])

        # x tick labels include support
        total = int(support.sum())
        tick_labels = []
        for cls, n in zip(labels, support.tolist()):
            frac = (n / total) if total > 0 else 0.0
            tick_labels.append(f"{cls}\n(n={n}, {100*frac:.1f}%)")

        plt.ylim(0.0, 1.2)
        plt.grid(True, axis="y", alpha=0.3)
        plt.xticks(x, tick_labels, rotation=0, ha="center")
        plt.ylabel("Accuracy per label")
        plt.title(f"{split}: Per-label accuracy (Teacher vs Baseline vs Distilled)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close()

        print(f"Wrote: {fig_path}")


    # Aggregate plots across fractions for a few key signals
    # 1) Agreement with teacher vs fraction
    plt.figure()
    for model in sorted(df_sum["model"].unique()):
        sub = df_sum[(df_sum["model"] == model) & (df_sum["agree_with_teacher"].notna())]
        if sub.empty:
            continue
        sub = sub.sort_values("train_fraction")
        plt.plot(sub["train_fraction"], sub["agree_with_teacher"], marker="o", label=model)
    plt.xlabel("Train fraction")
    plt.ylabel("Agreement with teacher (pred label)")
    plt.title(f"{split}: Agreement with teacher vs fraction")
    plt.legend()
    resultp = os.path.join(results_dir, f"plot_{split}_agreement_with_teacher.png")
    plt.savefig(resultp, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Wrote: {resultp}")

    # 2) KL(student||teacher) vs fraction (if logits exist)
    plt.figure()
    any_kl = False
    for model in sorted(df_sum["model"].unique()):
        sub = df_sum[(df_sum["model"] == model) & (df_sum["mean_KL_student_to_teacher"].notna())]
        if sub.empty:
            continue
        any_kl = True
        sub = sub.sort_values("train_fraction")
        plt.plot(sub["train_fraction"], sub["mean_KL_student_to_teacher"], marker="o", label=model)
    if any_kl:
        plt.xlabel("Train fraction")
        plt.ylabel("Mean KL(student || teacher)")
        plt.title(f"{split}: KL divergence to teacher vs fraction")
        plt.legend()
        resultp = os.path.join(results_dir, f"plot_{split}_kl_to_teacher.png")
        plt.savefig(resultp, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Wrote: {resultp}")
    else:
        plt.close()
        print("No KL plots generated (missing logits in predictions).")


# %% --------------------------------------------
# Run evaluation
# -----------------------------------------------
if __name__ == "__main__":
    outputs = "./outputs"
    results = "./results"
    plot_training_history(outputs)
    
    set_plot_style()
    # Deep analysis from saved predictions (no model inference)
    deep_predictions_analysis(outputs, results, split="val")
    deep_predictions_analysis(outputs, results, split="test")
    