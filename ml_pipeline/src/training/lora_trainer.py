import torch
from peft import LoraConfig, get_peft_model, TaskType
import logging

class LoRATrainer:
    def __init__(self, model, target_modules=None, r=16, alpha=32, lora_dropout=0.05, use_qlora=False):
        self.logger = logging.getLogger(self.__class__.__name__)
        if target_modules is None:
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        self.config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            task_type=TaskType.CAUSAL_LM,
            use_qlora=use_qlora
        )
        self.model = get_peft_model(model, self.config)
        self.logger.info(f"Initialized {'QLoRA' if use_qlora else 'LoRA'} with r={r}, alpha={alpha}, targets={target_modules}")

    def get_model(self):
        return self.model 