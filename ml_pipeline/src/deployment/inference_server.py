from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from .model_registry import ModelRegistry
from .auth_handler import APIKeyAuth
from .monitoring import Monitoring
import asyncio
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

app = FastAPI()
logger = logging.getLogger("inference_server")
registry = ModelRegistry()
auth = APIKeyAuth()
monitor = Monitoring()

# Add rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

class InferenceRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 32
    adapter_version: str
    base_model: str

class ModelCache:
    def __init__(self):
        self.cache = {}

    async def get_model(self, base_model, adapter_version):
        key = (base_model, adapter_version)
        if key in self.cache:
            return self.cache[key]
        # Load base model with quantization
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=quant_config, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        adapter_info = registry.get_adapter(base_model, adapter_version)
        if not adapter_info:
            raise HTTPException(status_code=404, detail="Adapter version not found")
        model = PeftModel.from_pretrained(model, adapter_info["adapter_path"])
        model.eval()
        self.cache[key] = (model, tokenizer)
        # Warm-up
        _ = model.generate(**tokenizer("Hello", return_tensors="pt").to(model.device), max_new_tokens=1)
        return model, tokenizer

model_cache = ModelCache()

@app.on_event("startup")
async def startup_event():
    logger.info("Inference server started.")

@app.post("/infer")
@limiter.limit("5/minute")
async def infer(request: Request, payload: InferenceRequest, api_key: str = Depends(auth.verify_key)):
    monitor.log_request(request)
    model, tokenizer = await model_cache.get_model(payload.base_model, payload.adapter_version)
    inputs = tokenizer(payload.prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=payload.max_new_tokens)
    result = tokenizer.decode(output[0], skip_special_tokens=True)
    monitor.log_response_time(request)
    return JSONResponse({"result": result})

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/metrics")
async def metrics():
    return monitor.get_metrics() 