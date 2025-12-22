from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from utils import load_emotion_dataset, load_model, compute_metrics, parse_args

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
    # tokenizer for both teacher and student
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    # training dataset 
    dataset = load_emotion_dataset(tokenizer, train_fraction=args.train_fraction, seed=args.seed)


    # load teacher from checkpoint
    teacher_path = args.output_dir + "/teacher_bert_base"
    teacher = AutoModelForSequenceClassification.from_pretrained(teacher_path, num_labels=6).to(args.device)
    # load student distillbert 
    print(f"Loading student model {student_name} for distillation...")
    student = AutoModelForSequenceClassification.from_pretrained(student_name, num_labels=teacher.config.num_labels).to(args.device)

    print("Loaded teacher and student models.")

    # define distillation training arguments
    training_args = DistillTrainingArguments(
        output_dir= args.output_dir + "/distilled_model",
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        alpha=0.7,
        temperature=3.0,
        eval_strategy="epoch",
        save_strategy="no",
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
    trainer.save_model(args.output_dir + "/distilled_model")
    metrics = trainer.evaluate()
    print("Distillation training completed. Evaluation metrics:", metrics)