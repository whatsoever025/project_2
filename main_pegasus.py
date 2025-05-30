from utils.constants import *
from utils.functions import set_seed, prepare_dataset

set_seed(SEED)

import numpy as np
import wandb, huggingface_hub, os, evaluate, torch, json
from transformers import TrainingArguments, Trainer, PegasusForConditionalGeneration, PegasusTokenizer, DataCollatorForSeq2Seq

# [PREPARING DATASET AND FUNCTIONS]
wandb.login(key=os.getenv("WANDB_API_KEY"))
huggingface_hub.login(token=os.getenv("HUGGINGFACE_TOKEN"))

train_dataset, val_dataset, test_dataset, tokenizer = prepare_dataset(
    tokenizer_name=TOKENIZER_PEGASUS,
    model_type="pegasus",
    max_input_length=512,
    max_target_length=128,
    save_dir="/kaggle/working/cnn_dailymail_subset_csv"
)

# Define compute_metrics using ROUGE for summarization
rouge = evaluate.load("rouge")

def preprocess_logits_for_metrics(logits, labels):
    """
    Preprocess model outputs to ensure valid token IDs for decoding.
    """
    if isinstance(logits, tuple):
        logits = logits[0]
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits, device=logits.device if hasattr(logits, 'device') else 'cpu')
    
    # Handle logits or token IDs
    if logits.ndim == 4:  # (batch_size, num_beams, sequence_length, vocab_size)
        logits = logits.argmax(dim=-1)[:, 0, :]  # Select first beam
    elif logits.ndim == 3:  # (batch_size, sequence_length, vocab_size)
        logits = logits.argmax(dim=-1)
    elif logits.ndim == 2:  # (batch_size, sequence_length) - already token IDs
        pass
    else:
        raise ValueError(f"Unexpected logits shape: {logits.shape}")

    # Clip token IDs to valid range [0, vocab_size)
    vocab_size = tokenizer.vocab_size
    logits = torch.clamp(logits, min=0, max=vocab_size - 1)
    
    # Replace invalid or negative token IDs with pad_token_id
    logits = torch.where(logits < 0, tokenizer.pad_token_id, logits)
    
    return logits

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Convert to numpy for processing
    predictions = np.array(predictions)
    labels = np.array(labels)
    # Validate token IDs
    vocab_size = tokenizer.vocab_size
    predictions = np.clip(predictions, 0, vocab_size - 1)
    predictions[predictions < 0] = tokenizer.pad_token_id
    labels = np.clip(labels, 0, vocab_size - 1)
    labels[labels < 0] = tokenizer.pad_token_id
    # Remove padding tokens
    predictions = [[int(id) for id in seq if id != tokenizer.pad_token_id] for seq in predictions]
    labels = [[int(id) for id in seq if id != tokenizer.pad_token_id] for seq in labels]
    # Decode predictions and labels
    try:
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    except Exception as e:
        print(f"Decoding error: {e}")
        decoded_preds = [""] * len(predictions)
        decoded_labels = [""] * len(labels)
    # Compute ROUGE metrics
    return rouge.compute(predictions=decoded_preds, references=decoded_labels)

# [SETTING UP MODEL AND TRAINING ARGUMENTS]
os.makedirs(EXPERIMENT_RESULTS_DIR_PEGASUS, exist_ok=True)

def get_last_checkpoint(output_dir):
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint")]
    if checkpoints:
        last_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1]
        return os.path.join(output_dir, last_checkpoint)
    return None

checkpoint = get_last_checkpoint(EXPERIMENT_RESULTS_DIR_PEGASUS)
if checkpoint:
    model = PegasusForConditionalGeneration.from_pretrained(checkpoint)
else:
    model = PegasusForConditionalGeneration.from_pretrained(MODEL_PEGASUS)

training_args = TrainingArguments(
    run_name=EXPERIMENT_NAME_PEGASUS,
    report_to="wandb",
    eval_strategy='steps',
    save_strategy='steps',
    eval_steps=EVAL_STEPS_PEGASUS,
    save_steps=SAVE_STEPS_PEGASUS,
    per_device_train_batch_size=TRAIN_BATCH_SIZE_PEGASUS,
    per_device_eval_batch_size=EVAL_BATCH_SIZE_PEGASUS,
    num_train_epochs=NUM_TRAIN_EPOCHS_PEGASUS,
    weight_decay=WEIGHT_DECAY_PEGASUS,
    learning_rate=LR_PEGASUS,
    lr_scheduler_type="linear",
    output_dir=EXPERIMENT_RESULTS_DIR_PEGASUS,
    logging_dir=EXPERIMENT_RESULTS_DIR_PEGASUS + "/logs",
    logging_steps=LOGGING_STEPS_PEGASUS,
    load_best_model_at_end=True,
    metric_for_best_model="eval_rougeL",
    greater_is_better=True,
    save_total_limit=2,
    fp16=True,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS_PEGASUS,
    seed=SEED
)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR_PEGASUS)
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

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, scheduler),
    data_collator=data_collator
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
model.save_pretrained(EXPERIMENT_RESULTS_DIR_PEGASUS)
tokenizer.save_pretrained(EXPERIMENT_RESULTS_DIR_PEGASUS)

with open(EXPERIMENT_RESULTS_DIR_PEGASUS + "/training_args.txt", "w") as f:
    f.write(str(training_args))

with open(EXPERIMENT_RESULTS_DIR_PEGASUS + "/test_results.json", "w") as f:
    json.dump({"metrics": test_results}, f)

wandb.save(EXPERIMENT_RESULTS_DIR_PEGASUS + "/test_results.json")
wandb.log({"test_metrics": test_results})

api = huggingface_hub.HfApi()
api.upload_large_folder(
    folder_path=RESULTS_DIR_PEGASUS,
    repo_id="TheSyx/text-summarization",
    repo_type="model",
    private=False
)