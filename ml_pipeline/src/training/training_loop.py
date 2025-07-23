import torch
from accelerate import Accelerator
import logging
from transformers import Trainer, TrainingArguments

class TrainingLoop:
    def __init__(self, model, tokenizer, train_dataset, val_dataset, output_dir="outputs",
                 batch_size=4, grad_accum_steps=8, epochs=3, lr=3e-5, mixed_precision="bf16", checkpoint_steps=500):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.accelerator = Accelerator(mixed_precision=mixed_precision)
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.grad_accum_steps = grad_accum_steps
        self.epochs = epochs
        self.lr = lr
        self.checkpoint_steps = checkpoint_steps
        self.training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.grad_accum_steps,
            num_train_epochs=self.epochs,
            learning_rate=self.lr,
            evaluation_strategy="steps",
            eval_steps=self.checkpoint_steps,
            save_steps=self.checkpoint_steps,
            save_total_limit=3,
            fp16=(mixed_precision=="fp16"),
            bf16=(mixed_precision=="bf16"),
            logging_steps=50,
            report_to=["wandb"],
            load_best_model_at_end=True,
        )

    def run(self):
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer,
        )
        self.logger.info("Starting training loop...")
        trainer.train()
        self.logger.info("Training complete. Saving final model...")
        trainer.save_model(self.output_dir) 