from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
 
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
        # freeze teacher weights
        self.teacher.eval()
 
    def compute_loss(self, model, inputs, return_outputs=False):
        # student output
        outputs_student = model(**inputs)
        student_loss=outputs_student.loss
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

    # training dataset 
    dataset = load_dataset("dair-ai/emotion", "split")['train']
    # limit dataset size 
    percentage = 0.1
    dataset = dataset.select(range(int(len(dataset)*percentage)))

    # tokenizer for both teacher and student
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize_fn(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    print(f"Using {len(tokenized)} samples for distillation training.")

    # load teacher from checkpoint
    teacher = AutoModelForSequenceClassification.from_pretrained("teacher")
    # load student distillbert 
    student = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=teacher.config.num_labels)

    print("Loaded teacher and student models.")

    # define distillation training arguments
    training_args = DistillTrainingArguments(
        output_dir="./distilled_model",
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        alpha=0.7,
        temperature=3.0,
    )

    # initialize distillation trainer
    trainer = DistillTrainer(
        model=student,
        args=training_args,
        train_dataset=tokenized,
        teacher=teacher,
    )
    # start distillation training
    trainer.train()
