
# %%
import os
import json

from utils import load_emotion_dataset, load_model, compute_metrics, parse_args, get_run_id, save_dataset, save_split_predictions
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# %%
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, TrainingArguments, Trainer

# %%
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train_model(
    model_name: str,
    dataset,
    tokenizer,
    output_dir: str,
    epochs=3,
    batch_size=32,
    ):

    model = load_model(model_name).to(DEVICE)

    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch", # save model at each epoch
        load_best_model_at_end=True, # load best model when finished training
        metric_for_best_model="eval_accuracy", # use accuracy to evaluate best model
        greater_is_better=True, # higher accuracy is better
        save_total_limit=1,
        logging_strategy="epoch", # log at each epoch
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
    val_metrics = trainer.evaluate(eval_dataset=dataset["validation"])
    test_metrics = trainer.evaluate(eval_dataset=dataset["test"])
    metrics = {
        "val": val_metrics,
        "test": test_metrics,
    }

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump({"val": val_metrics, "test": test_metrics}, f, indent=2)

    with open(os.path.join(output_dir, "log_history.json"), "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    save_split_predictions(
        trainer,
        dataset["validation"],
        out_path=os.path.join(output_dir, "val_predictions.jsonl"),
        include_logits=True,
    )
    save_split_predictions(
        trainer,
        dataset["test"],
        out_path=os.path.join(output_dir, "test_predictions.jsonl"),
        include_logits=True,
    )

    return model, trainer, metrics


if __name__ == "__main__":

    args = parse_args()

    model_name = args.model_name 
    is_teacher = args.teacher

    run_dir = os.path.join(args.output_dir, get_run_id(args))
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "run_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_emotion_dataset(tokenizer, train_fraction=args.train_fraction, seed=args.seed)

    run_id = get_run_id(args)

    if is_teacher:
        print("\n" + "=" * 30 + " Training teacher model " + model_name + "=" * 30)
    else:
        student_ds_path = os.path.join(args.output_dir, run_id, "student_dataset_bertmini")
        save_dataset(dataset, student_ds_path)
        print(f"Saved student dataset to:\n  {student_ds_path}")

        print("\n" + "=" * 30 + " Training student model " + model_name + "=" * 30)
        
    teacher_model, _, teacher_metrics = train_model(
        model_name=model_name,
        dataset=dataset,
        tokenizer=tokenizer,
        output_dir=os.path.join(args.output_dir, run_id, "teacher_bert_base" if is_teacher else "student_baseline_bertmini"),
        epochs=args.num_epochs,
    )
