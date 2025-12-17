
# %%
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# %%
import numpy as np
import torch
import torch.nn as nn
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# %%
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

teacher_name = "bert-base-uncased"
student_name = "distilbert-base-uncased"
student_bert_mini_name = "prajjwal1/bert-mini"

tokenizer = AutoTokenizer.from_pretrained(teacher_name)

def load_emotion_dataset(train_fraction: float = 1.0, seed: int = 1):
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

        dataset["train"] = (
            dataset["train"]
            .shuffle(seed=seed)
            .select(range(train_size))
        )

    print("\n" + "=" * 30 + " LOADED DATASET " + "=" * 30)
    print("Fraction of training data: ", train_fraction)
    print(
    f"Train: {dataset['train'].num_rows}, "
    f"Val: {dataset['validation'].num_rows}, "
    f"Test: {dataset['test'].num_rows}"
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

def train_model(
    model_name: str,
    dataset,
    output_dir: str,
    epochs=3,
    batch_size=32,
    train_fraction: float = 1.0,
    ):

    model = load_model(model_name).to(DEVICE)

    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="no",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)
    metrics = trainer.evaluate()

    return model, trainer, metrics


if __name__ == "__main__":
    dataset = load_emotion_dataset(train_fraction=1.0)

    print("\n" + "=" * 30 + " Training teacher model " + "=" * 30)
    teacher_model, _, teacher_metrics = train_model(
        model_name=teacher_name,
        dataset=dataset,
        output_dir="./teacher_bert_base",
        epochs=5,
        train_fraction=1.0,
    )

    print("\n" + "=" * 30 + " Training student base model " + "=" * 30)
    student_base_model, _, student_base_metrics = train_model(
        model_name=student_name,
        dataset=dataset,
        output_dir="./student_baseline_distilbert",
        epochs=5,
        train_fraction=1.0,
    )

    print("\n" + "=" * 30 + " Training BERT tiny student base model " + "=" * 30)
    student_base_model, _, student_base_metrics = train_model(
        model_name=student_bert_mini_name,
        dataset=dataset,
        output_dir="./student_baseline_bert-tiny",
        epochs=5,
        train_fraction=1.0,
    )

