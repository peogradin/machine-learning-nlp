import os
import argparse
import evaluate
from datasets import load_dataset
import torch
from transformers import AutoModelForSequenceClassification
import numpy as np
import json

DEFAULT_OUTPUT_DIR = "./outputs"
DEFAULT_NUM_EPOCHS = 2
DEFAULT_SEED = 101

def parse_args():
    parser = argparse.ArgumentParser(description="Distillation training script.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to use (default auto-detect).",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional identifier to group outputs (e.g. seed101_frac0.5). If omitted, uses seed+fraction."
    )
    return parser.parse_args()


def load_emotion_dataset(tokenizer, train_fraction: float = 1.0, seed: int = 1):
    dataset = load_dataset("dair-ai/emotion")

    def tokenize(batch):
        return tokenizer(
            batch["text"], 
            truncation=True,
            padding="max_length",
            max_length=128
            )
    tokenized = dataset.map(tokenize, batched=True)
    tokenized = tokenized.remove_columns(["text"])
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")

    if train_fraction < 1.0:
        train_size = int(train_fraction * dataset["train"].num_rows)

        tokenized["train"] = (
            tokenized["train"]
            .shuffle(seed=seed)
            .select(range(train_size))
        )

    print("\n" + "=" * 30 + " LOADED DATASET " + "=" * 30)
    print("Fraction of training data: ", train_fraction)
    print(
    f"Train: {tokenized['train'].num_rows}, "
    f"Val: {tokenized['validation'].num_rows}, "
    f"Test: {tokenized['test'].num_rows}"
    )
    return tokenized

def get_run_id(args):
    if args.run_id is not None:
        return args.run_id
    else:
        return f"seed{args.seed}_frac{args.train_fraction}"
    
def save_dataset(dataset, path: str):
    os.makedirs(path, exist_ok=True)
    dataset.save_to_disk(path)

def load_saved_dataset(path: str):
    from datasets import load_from_disk
    return load_from_disk(path)

def load_model(model_name: str):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=6,
    )

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1.compute(predictions=preds, references=labels, average="weighted")["f1"],
    }

@torch.no_grad()
def save_split_predictions(trainer, dataset_split, out_path: str, include_logits: bool = True):
    """
    Saves per-example predictions on a split (e.g. test/val) as JSONL.
    Each row: idx, true, pred, (optional) logits
    """
    preds = trainer.predict(dataset_split)
    logits = preds.predictions
    labels = preds.label_ids
    pred = logits.argmax(axis=-1)

    # write jsonl
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for i in range(len(labels)):
            row = {
                "idx": int(i),
                "true": int(labels[i]),
                "pred": int(pred[i]),
            }
            if include_logits:
                # keep it smaller; float16 is enough
                row["logits"] = np.asarray(logits[i], dtype=np.float16).tolist()
            f.write(json.dumps(row) + "\n")
