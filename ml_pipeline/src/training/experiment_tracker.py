import wandb
import logging

class ExperimentTracker:
    def __init__(self, project: str = "llm-finetune", run_name: str = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        wandb.init(project=project, name=run_name)
        self.logger.info(f"Initialized wandb run: {wandb.run.name}")

    def log_metrics(self, metrics: dict, step: int = None):
        wandb.log(metrics, step=step)
        self.logger.info(f"Logged metrics at step {step}: {metrics}")

    def save_checkpoint(self, path: str):
        wandb.save(path)
        self.logger.info(f"Saved checkpoint to wandb: {path}") 