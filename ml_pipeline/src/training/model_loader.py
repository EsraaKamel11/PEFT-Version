import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import logging

def load_model(model_name: str, quantization: str = '4bit', device: str = None):
    """
    Load a model with quantization (4bit/8bit) using bitsandbytes.
    """
    logger = logging.getLogger("model_loader")
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Loading model {model_name} with {quantization} quantization on {device}")
    
    if quantization == '4bit':
        if device == 'cpu':
            # CPU requires nf4 quantization type
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True, 
                bnb_4bit_compute_dtype=torch.float32,
                bnb_4bit_quant_type="nf4"
            )
        else:
            # GPU can use fp4 or nf4
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True, 
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4"  # Use nf4 for better compatibility
            )
    elif quantization == '8bit':
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        quant_config = None
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='auto' if device == 'cuda' else None,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if device == 'cuda' else torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    return model, tokenizer 