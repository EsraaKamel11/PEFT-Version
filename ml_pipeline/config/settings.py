from pydantic import BaseSettings
from pathlib import Path
import yaml
import os
import logging

class Settings(BaseSettings):
    # Example config fields
    model_name: str = "gpt-3"
    train_batch_size: int = 8
    eval_batch_size: int = 8
    learning_rate: float = 3e-5
    use_gpu: bool = True
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Load YAML config and override defaults
config_path = Path(__file__).parent / "config.yaml"
if config_path.exists():
    with open(config_path, "r") as f:
        yaml_config = yaml.safe_load(f)
else:
    yaml_config = {}

settings = Settings(**yaml_config)

# Logging configuration
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s - %(levelname)s - %(name)s - `%(message)s"
)
logger = logging.getLogger("ml_pipeline") 