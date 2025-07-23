# ML Pipeline: End-to-End LLM Fine-Tuning

This project provides a modular, extensible pipeline for end-to-end fine-tuning of large language models (LLMs). It supports local GPU training and CPU inference, with configuration management via Pydantic and YAML, and logging.

## Project Structure

```
ml_pipeline/
├── config/
│   ├── __init__.py
│   ├── settings.py
│   └── config.yaml
├── src/
│   ├── __init__.py
│   ├── data_collection/
│   ├── data_processing/
│   ├── training/
│   ├── evaluation/
│   ├── deployment/
│   └── utils/
├── tests/
├── requirements.txt
├── setup.py
├── README.md
└── docker/
```

## Features
- Modular and extensible
- Pydantic-based config with YAML and environment variable support
- Logging configuration
- Local GPU training and CPU inference

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Edit `config/config.yaml` or set environment variables as needed.
3. Run your pipeline scripts from `src/`.

## Docker
A `docker/` directory is provided for containerization (add your Dockerfiles and scripts).

## License
MIT 