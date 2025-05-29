from utils.constants import *
from utils.functions import set_seed, prepare_dataset

set_seed(SEED)

import wandb, huggingface_hub, os
import evaluate
import numpy as np
from transformers import TrainingArguments, Trainer, T5ForConditionalGeneration, T5Tokenizer, AdamW
import torch

# [PREPARING DATASET AND FUNCTIONS]
wandb.login(key=os.getenv("WANDB_API_KEY"))
huggingface_hub.login(token=os.getenv("HUGGINGFACE_TOKEN"))

train_dataset, val_dataset, test_dataset, tokenizer = prepare_dataset(TOKENIZER_T5)

# Define compute_metrics using ROUGE for summarization
rouge = evaluate.load("rouge")

def preprocess_logits_for_metrics(logits, labels):
    # Handle tuple
    if isinstance(logits, tuple):
        logits = logits[0]
    
    # Convert to tensor
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits, device=logits.device if hasattr(logits, 'device') else 'cpu')
    
    # Handle logits
    if logits.ndim == 4:  # (batch_size, num_beams, sequence_length, vocab_size)
        logits = logits.argmax(dim=-1)  # (batch_size, num_beams, sequence_length)
        logits = logits[:, 0, :]  # Select first beam
    elif logits.ndim == 3:  # (batch_size, sequence_length, vocab_size)
        logits = logits.argmax(dim=-1)  # (batch_size, sequence_length)
    elif logits.ndim == 2:  # (batch_size, sequence_length)
        pass
    
    return logits

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Remove padding tokens
    predictions = [[id for id in seq if id != tokenizer.pad_token_id] for seq in predictions.tolist()]
    labels = [[id for id in seq if id != tokenizer.pad_token_id] for seq in labels.tolist()]
    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Compute ROUGE metrics
    return rouge.compute(predictions=decoded_preds, references=decoded_labels)

# [SETTING UP MODEL AND TRAINING ARGUMENTS]
os.makedirs(EXPERIMENT_RESULTS_DIR_T5, exist_ok=True)

def get_last_checkpoint(output_dir):
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint")]
    if checkpoints:
        last_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1]
        return os.path.join(output_dir, last_checkpoint)
    return None

checkpoint = get_last_checkpoint(EXPERIMENT_RESULTS_DIR_T5)
if checkpoint:
    model = T5ForConditionalGeneration.from_pretrained(checkpoint)
else:
    model = T5ForConditionalGeneration.from_pretrained(MODEL_T5)

training_args = TrainingArguments(
    run_name=EXPERIMENT_NAME_T5,
    report_to="wandb",
    evaluation_strategy='steps',
    save_strategy='steps',
    eval_steps=EVAL_STEPS_T5,
    save_steps=SAVE_STEPS_T5,
    per_device_train_batch_size=TRAIN_BATCH_SIZE_T5,
    per_device_eval_batch_size=EVAL_BATCH_SIZE_T5,
    num_train_epochs=NUM_TRAIN_EPOCHS_T5,
    weight_decay=WEIGHT_DECAY_T5,
    learning_rate=LR_T5,
    lr_scheduler_type="linear",
    output_dir=EXPERIMENT_RESULTS_DIR_T5,
    logging_dir=EXPERIMENT_RESULTS_DIR_T5 + "/logs",
    logging_steps=LOGGING_STEPS_T5,
    load_best_model_at_end=True,
    metric_for_best_model="eval_rougeL",
    greater_is_better=True,
    save_total_limit=2,
    fp16=True,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS_T5,
    seed=SEED
)

optimizer = AdamW(model.parameters(), lr=LR_T5)
total_steps = len(train_dataset) * training_args.num_train_epochs

class LinearDecayWithMinLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, min_lr, max_steps, last_epoch=-1):
        self.min_lr = min_lr
        self.max_steps = max_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        lr_decay = max(0, (1 - step / self.max_steps)) * (self.base_lrs[0] - self.min_lr) + self.min_lr
        return [lr_decay] * len(self.base_lrs)

scheduler = LinearDecayWithMinLR(
    optimizer,
    min_lr=1e-6,
    max_steps=total_steps
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, scheduler)
)

# [TRAINING]
if checkpoint:
    trainer.train(resume_from_checkpoint=checkpoint)
else:
    trainer.train()

# [EVALUATING]
torch.cuda.empty_cache()
test_results = trainer.evaluate(test_dataset, metric_key_prefix="test")

# [SAVING THINGS]
model.save_pretrained(EXPERIMENT_RESULTS_DIR_T5)
tokenizer.save_pretrained(EXPERIMENT_RESULTS_DIR_T5)

with open(EXPERIMENT_RESULTS_DIR_T5 + "/training_args.txt", "w") as f:
    f.write(str(training_args))

with open(EXPERIMENT_RESULTS_DIR_T5 + "/test_results.txt", "w") as f:
    f.write(str(test_results))

api = huggingface_hub.HfApi()
api.upload_large_folder(
    folder_path=RESULTS_DIR_T5,
    repo_id="TheSyx/text-summarization",
    repo_type="model",
    private=False
)
