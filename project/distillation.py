import os
import json
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import compute_metrics, parse_args, get_run_id, load_saved_dataset, save_split_predictions

teacher_name = "bert-base-uncased"
student_name = "prajjwal1/bert-mini"

class DistillTrainingArguments(TrainingArguments):
    def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
 
        self.alpha = alpha
        self.temperature = temperature
 
class DistillTrainer(Trainer):
    def __init__(self, *args, teacher=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher
        self.teacher.to(self.model.device)
        for p in self.teacher.parameters():
            p.requires_grad = False
        # freeze teacher weights
        self.teacher.eval()
 
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # student output
        outputs_student = model(**inputs)
        student_loss = outputs_student.loss
        # teacher output
        with torch.no_grad():
          outputs_teacher = self.teacher(**inputs)
 
        # compute distillation loss
        loss_function = nn.KLDivLoss(reduction="batchmean")
        loss_logits = (loss_function(
            F.log_softmax(outputs_student.logits / self.args.temperature, dim=-1),
            F.softmax(outputs_teacher.logits / self.args.temperature, dim=-1)) * (self.args.temperature ** 2))
        # Combined weighted loss
        loss = self.args.alpha * student_loss + (1. - self.args.alpha) * loss_logits
        return (loss, outputs_student) if return_outputs else loss
    
if  __name__ == "__main__":
    args = parse_args()
    with open(os.path.join(args.output_dir, get_run_id(args), "run_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # tokenizer for both teacher and student
    tokenizer = AutoTokenizer.from_pretrained(student_name)
    # training dataset 
    # dataset = load_emotion_dataset(tokenizer, train_fraction=args.train_fraction, seed=args.seed)
    run_id = get_run_id(args)
    student_ds_path = os.path.join(args.output_dir, run_id, "student_dataset_bertmini")
    dataset = load_saved_dataset(student_ds_path)
    print(f"Loaded student dataset from: {student_ds_path}")

    # load teacher from checkpoint
    teacher_path = os.path.join(args.output_dir, f"seed{args.seed}_frac1.0", "teacher_bert_base")
    teacher = AutoModelForSequenceClassification.from_pretrained(teacher_path, num_labels=6).to(args.device)
    # load student distillbert 
    print(f"Loading student model {student_name} for distillation...")
    student = AutoModelForSequenceClassification.from_pretrained(student_name, num_labels=teacher.config.num_labels).to(args.device)

    print("Loaded teacher and student models.")

    # define distillation training arguments
    training_args = DistillTrainingArguments(
        output_dir= os.path.join(args.output_dir, run_id, "distilled_model"),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        alpha=0.7,
        temperature=3.0,
        eval_strategy="epoch",
        logging_strategy="epoch", # log at each epoch
        save_strategy="epoch", # save at each epoch
        load_best_model_at_end=True, # load best model when finished training
        metric_for_best_model="eval_accuracy", # use accuracy to evaluate best model
        greater_is_better=True, # higher accuracy is better
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    # initialize distillation trainer
    trainer = DistillTrainer(
        model=student,
        args=training_args,
        train_dataset=dataset["train"],
        teacher=teacher,
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    # start distillation training
    trainer.train()
    trainer.save_model(training_args.output_dir)
    save_split_predictions(trainer, dataset["test"], os.path.join(training_args.output_dir, "test_predictions.jsonl"))
    save_split_predictions(trainer, dataset["validation"], os.path.join(training_args.output_dir, "val_predictions.jsonl"))
    val_metrics = trainer.evaluate(eval_dataset=dataset["validation"])
    test_metrics = trainer.evaluate(eval_dataset=dataset["test"])
    metrics = {
        "val": val_metrics,
        "test": test_metrics,
    }
    with open(os.path.join(training_args.output_dir, "metrics.json"), "w") as f:
        json.dump({"val": val_metrics, "test": test_metrics}, f, indent=2)

    with open(os.path.join(training_args.output_dir, "log_history.json"), "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    print("Distillation training completed. Evaluation metrics:", metrics)