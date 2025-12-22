
# %%
import os

from utils import load_emotion_dataset, load_model, compute_metrics, parse_args
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# %%
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, TrainingArguments, Trainer

# %%
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

teacher_name = "bert-base-uncased"
student_name = "distilbert-base-uncased"
student_bert_mini_name = "prajjwal1/bert-mini"


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

    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(teacher_name)
    dataset = load_emotion_dataset(tokenizer, train_fraction=args.train_fraction, seed=args.seed)

    print("\n" + "=" * 30 + " Training teacher model " + "=" * 30)
    teacher_model, _, teacher_metrics = train_model(
        model_name=teacher_name,
        dataset=dataset,
        output_dir=args.output_dir + "/teacher_bert_base",
        epochs=args.num_epochs,
        train_fraction=1.0, # QUESTION: shouldn't we always train teacher on full data?
    )

    print("\n" + "=" * 30 + " Training student base model " + "=" * 30)
    student_base_model, _, student_base_metrics = train_model(
        model_name=student_name,
        dataset=dataset,
        output_dir=args.output_dir + "/student_baseline_distilbert",
        epochs=args.num_epochs,
        train_fraction=args.train_fraction,
    )

    print("\n" + "=" * 30 + " Training BERT tiny student base model " + "=" * 30)
    student_base_model, _, student_base_metrics = train_model(
        model_name=student_bert_mini_name,
        dataset=dataset,
        output_dir=args.output_dir + "/student_baseline_bert-tiny",
        epochs=args.num_epochs,
        train_fraction=args.train_fraction,
    )

