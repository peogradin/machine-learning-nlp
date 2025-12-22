import argparse
import evaluate
from datasets import load_dataset
import torch
from transformers import AutoModelForSequenceClassification
import numpy as np

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
