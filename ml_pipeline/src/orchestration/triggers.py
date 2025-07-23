from prefect.deployments import run_deployment
from prefect import flow
import logging
import time

# Manual trigger
def manual_trigger(flow_func):
    logging.info("Manually triggering pipeline...")
    flow_func()

# Scheduled trigger (simple example)
def schedule_trigger(flow_func, interval_seconds=3600):
    logging.info(f"Scheduling pipeline every {interval_seconds} seconds...")
    while True:
        flow_func()
        time.sleep(interval_seconds) 