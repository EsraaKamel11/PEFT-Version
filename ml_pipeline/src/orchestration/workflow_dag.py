from prefect import flow, task
import logging
from slack_sdk import WebClient
import os

@task
def data_collection_task():
    logging.info("Running data collection...")
    # Call your data collection logic here

@task
def data_processing_task():
    logging.info("Running data processing...")
    # Call your data processing logic here

@task
def training_task():
    logging.info("Running model training...")
    # Call your training logic here

@task
def evaluation_task():
    logging.info("Running evaluation...")
    # Call your evaluation logic here

@task
def deployment_task():
    logging.info("Running deployment...")
    # Call your deployment logic here

@task
def slack_alert_task(message: str):
    slack_token = os.environ.get("SLACK_API_TOKEN")
    slack_channel = os.environ.get("SLACK_CHANNEL", "#general")
    if not slack_token:
        logging.error("SLACK_API_TOKEN not set in environment.")
        return
    client = WebClient(token=slack_token)
    try:
        client.chat_postMessage(channel=slack_channel, text=message)
        logging.info(f"Sent Slack alert: {message}")
    except Exception as e:
        logging.error(f"Failed to send Slack alert: {e}")

@flow(name="LLM Fine-tuning Pipeline")
def build_workflow():
    try:
        data_collection_task()
        data_processing_task()
        training_task()
        evaluation_task()
        deployment_task()
    except Exception as e:
        slack_alert_task.submit(f"Pipeline failure: {e}")
        raise 