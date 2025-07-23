import logging
from .workflow_dag import build_workflow

def run_pipeline():
    logging.info("Starting full pipeline execution...")
    try:
        build_workflow()
        logging.info("Pipeline execution completed successfully.")
    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}") 